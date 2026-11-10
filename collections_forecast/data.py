"""Read-only SQLite access and explicit vintage-aware input contracts."""
from pathlib import Path
from contextlib import contextmanager
import hashlib
import re
import sqlite3
import numpy as np
import pandas as pd
from .config import month_start, origin, days


def identifier(name):
    if not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", name):
        raise ValueError(f"Unsafe SQL identifier: {name!r}")
    return '"' + name + '"'


def numeric(frame, columns):
    for col in columns:
        parsed = pd.to_numeric(frame[col], errors="coerce")
        bad = parsed.isna() | ~np.isfinite(parsed)
        if bad.any():
            examples = frame.loc[bad, col].head(5).to_dict()
            raise ValueError(f"Missing or non-finite numeric value: {col}; {int(bad.sum())}/{len(frame)} rows; "
                             f"examples (index: value)={examples}. Check NULLs and numeric formatting; values were not replaced with zero.")
        frame[col] = parsed
    return frame


def normalize_segments(frame, segments):
    for col in segments:
        if col not in frame:
            raise ValueError(f"Missing segment column: {col}")
        frame[col] = frame[col].astype("string").fillna("__UNDEFINED__").str.strip()
        frame[col] = frame[col].replace("", "__UNDEFINED__").astype(str)
        if col == "inflow":
            frame[col] = frame[col].replace({"0.0": "0", "1.0": "1"})
    return frame


def parse_dates(values, encoding="iso"):
    if isinstance(values.dtype, pd.DatetimeTZDtype):
        return values.dt.tz_convert("Europe/Moscow").dt.tz_localize(None)
    if pd.api.types.is_datetime64_any_dtype(values.dtype):
        return values
    if encoding == "iso":
        # Naive strings are local Moscow time; offset-aware strings are converted to it.
        parsed = pd.to_datetime(values, format="mixed", errors="raise", utc=True)
        aware = values.astype(str).str.contains(r"(?:Z|[+-]\d{2}:?\d{2})$", regex=True)
        result = parsed.dt.tz_localize(None)
        result.loc[aware] = parsed.loc[aware].dt.tz_convert("Europe/Moscow").dt.tz_localize(None)
        return result
    if encoding == "yyyymmdd":
        return pd.to_datetime(values.astype(str).str.replace(r"\.0$", "", regex=True), format="%Y%m%d")
    return pd.to_datetime(pd.to_numeric(values), unit=encoding, utc=True).dt.tz_convert("Europe/Moscow").dt.tz_localize(None)


def sql_date(column, encoding):
    q = identifier(column)
    if encoding == "iso":
        aware = f"(substr({q},-1)='Z' OR substr({q},-6,1) IN ('+','-') OR substr({q},-5,1) IN ('+','-'))"
        return f"CASE WHEN {aware} THEN date({q}, '+3 hours') ELSE date({q}) END"
    if encoding == "yyyymmdd":
        return f"date(substr({q},1,4)||'-'||substr({q},5,2)||'-'||substr({q},7,2))"
    scale = {"s": 1, "ms": 1000, "us": 1000000, "ns": 1000000000}[encoding]
    return f"date({q} / {scale}.0, 'unixepoch', '+3 hours')"


@contextmanager
def readonly_connection(path):
    p = Path(path).resolve()
    if not p.is_file():
        raise FileNotFoundError(f"SQLite database not found: {p}")
    conn = sqlite3.connect(p.as_uri() + "?mode=ro", uri=True)
    try:
        conn.execute("PRAGMA query_only=ON")
        yield conn
    finally:
        conn.close()


def actual_keys(config):
    return ["report_month", "report_dt", "dt_in", *config.segments]


def with_report_date(frame):
    if "report_dt" not in frame and "date" in frame:
        return frame.rename(columns={"date": "report_dt"})
    return frame


def load_raw(config, before=None, month=None):
    """SQL excludes target facts from the prediction read, not just from fitting."""
    if (before is None) == (month is None):
        raise ValueError("Specify either before or month")
    cols = [*actual_keys(config), "w1", "w2", "cession", "write_off", "target", "acc_count", "avg_dpd", "avg_pvr"]
    if config.raw_available_column:
        cols.append(config.raw_available_column)
    table = identifier(config.raw_table)
    expr = sql_date("report_dt", config.raw_date_encoding)
    with readonly_connection(config.db_path) as conn:
        present = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        missing = set(cols + ["is_forecast"]) - present
        if missing:
            raise ValueError(f"{config.raw_table}: missing columns {sorted(missing)}")
        invalid = conn.execute(
            f"SELECT COUNT(*) FROM {table} WHERE is_forecast=0 AND {expr} IS NULL"
        ).fetchone()[0]
        if invalid:
            raise ValueError(f"{invalid} invalid report_dt values; check raw_date_encoding")
        if before is not None:
            where, params = f"{expr} < ?", [month_start(before).strftime("%Y-%m-%d")]
        else:
            start = month_start(month)
            where = f"{expr} >= ? AND {expr} < ?"
            params = [start.strftime("%Y-%m-%d"), (start + pd.offsets.MonthBegin()).strftime("%Y-%m-%d")]
        frame = pd.read_sql_query(
            f"SELECT {', '.join(identifier(c) for c in cols)} FROM {table} "
            f"WHERE is_forecast=0 AND {where}", conn, params=params)
    return prepare_actual(frame, config, before=before, month=month)


def prepare_actual(frame, config, *, before=None, month=None):
    """Validate raw RR rows from either SQLite or the caller's DataFrame."""
    if (before is None) == (month is None):
        raise ValueError("Specify either before or month")
    frame = with_report_date(frame)
    dims = list(config.segments)
    precomputed = "fact" in frame
    amounts = ["fact"] if precomputed else ["w1", "w2", "cession", "write_off"]
    portfolio = ["acc_count", "avg_dpd", "avg_pvr"]
    if precomputed:
        portfolio.append("w1")
    columns = [*actual_keys(config), *amounts]
    if before is not None:
        columns.append("target")
    # Evaluation uses the frozen forecast plan, not target from the actual export.
    columns += portfolio if before is not None and not precomputed else [c for c in portfolio if c in frame]
    if config.raw_available_column:
        columns.append(config.raw_available_column)
    missing = set(columns) - set(frame)
    if missing:
        raise ValueError(f"Actual frame: missing columns {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"No actual rows found for {'history before ' + str(before) if before else month}")
    if "is_forecast" in frame and not pd.to_numeric(frame.is_forecast, errors="raise").eq(0).all():
        raise ValueError("Actual frames must contain only rows with is_forecast=0")
    frame = frame[columns].copy()
    for col in ["report_dt", "report_month", "dt_in"]:
        frame[col] = parse_dates(frame[col], config.raw_date_encoding).dt.normalize()
        if frame[col].isna().any():
            raise ValueError(f"Missing date in actual frame: {col}")
    if before is not None and frame.report_dt.ge(month_start(before)).any():
        raise ValueError(f"df_history must contain only dates before {month_start(before).date()}")
    if month is not None and not frame.report_dt.dt.to_period("M").eq(pd.Period(month, "M")).all():
        raise ValueError(f"actual_df must contain only dates in {month}")
    frame = normalize_segments(frame, dims)
    if (frame.report_dt.dt.to_period("M") != frame.report_month.dt.to_period("M")).any():
        raise ValueError("report_month and report_dt disagree")
    if not frame.report_month.dt.day.eq(1).all():
        raise ValueError("report_month must be the first day of the month")
    if frame.dt_in.isna().any() or frame.dt_in.gt(frame.report_dt).any():
        raise ValueError("Invalid or future dt_in in an actual row")
    numeric(frame, [c for c in ["fact", "w1", "w2", "cession", "write_off", "target", "acc_count"] if c in frame])
    for col in ["avg_dpd", "avg_pvr"]:
        if col not in frame:
            continue
        frame[col] = pd.to_numeric(frame[col], errors="raise")
        if np.isinf(frame[col]).any():
            raise ValueError(f"Infinite values in {col}")
    if not precomputed:
        frame["fact"] = frame.w1 - frame.w2 - frame.cession - frame.write_off
    # Preserve source values: a snapshot may be filtered repeatedly at different origins.
    # Differencing must occur AFTER selecting the available vintage, never before it.
    frame["_source_fact"] = frame.fact
    if "target" in frame:
        frame["_source_target"] = frame.target
    frame["date"] = frame.report_dt
    frame["month"] = frame.report_dt.dt.strftime("%Y-%m")
    if config.raw_available_column:
        frame["available_at"] = parse_dates(frame[config.raw_available_column])
        if frame.available_at.isna().any():
            raise ValueError("Missing raw availability timestamp")
    key = ["report_month", "date", "dt_in", *dims]
    version_key = key + (["available_at"] if config.raw_available_column else [])
    if frame.duplicated(version_key).any():
        duplicates = frame.loc[frame.duplicated(version_key, keep=False), version_key]
        raise ValueError(f"Duplicate raw grain (including dt_in): {len(duplicates)} rows on keys {version_key}; "
                         f"examples={duplicates.head(3).to_dict('records')}. "
                         "Check joins and source versions; repeated snapshot amounts must not be summed.")
    return frame


def cohort_daily(frame, value, mode, config):
    """Convert stable cohort snapshots without bridging gaps or erasing corrections."""
    if mode == "daily":
        return frame[value].copy()
    result = pd.Series(index=frame.index, dtype=float)
    month_ends = frame.groupby("report_month").date.max()
    keys = ["report_month", "dt_in", *config.segments]
    for _, group in frame.groupby(keys, sort=False, dropna=False):
        group = group.sort_values("date")
        start = max(group.report_month.iloc[0], group.dt_in.iloc[0])
        before_entry = group.date < start
        if group.loc[before_entry, value].ne(0).any():
            raise ValueError(f"Nonzero cumulative {value} before cohort entry")
        active = group.loc[~before_entry]
        end = month_ends.loc[group.report_month.iloc[0]]
        expected = pd.date_range(start, end, freq="D")
        if not pd.DatetimeIndex(active.date).equals(expected):
            missing = expected.difference(pd.DatetimeIndex(active.date))
            raise ValueError(f"Cumulative {value}: missing cohort snapshots from {start.date()} to {end.date()}; "
                             f"key={group[keys].iloc[0].to_dict()}; missing dates={missing.strftime('%Y-%m-%d').tolist()[:10]}. "
                             "If DPD/PVR segments change between snapshots, aggregate to a stable grain before differencing; "
                             "otherwise supply a complete stable cohort series or independently reconciled daily amounts")
        result.loc[group.index[before_entry]] = 0.0
        if not active.empty:
            change = active[value].diff()
            change.iloc[0] = active[value].iloc[0]
            result.loc[active.index] = change
    return result


def raw_asof(raw, config, cutoff):
    f = raw[raw.date <= cutoff].copy()
    if config.raw_available_column:
        f = f[f.available_at <= cutoff]
        key = ["report_month", "date", "dt_in", *config.segments]
        f = f.sort_values("available_at").drop_duplicates(key, keep="last")
    f = f.reset_index(drop=True)
    if "_source_fact" not in f:
        f["_source_fact"] = f.fact
    f["fact"] = cohort_daily(f, "_source_fact", config.fact_mode, config)
    if "target" in f:
        if "_source_target" not in f:
            f["_source_target"] = f.target
        f["target"] = cohort_daily(f, "_source_target", config.plan_mode, config)
    return f


def daily_actual(raw, config):
    raw = raw_asof(raw, config, pd.Timestamp.max - pd.Timedelta(days=1))
    aggregates = {"fact": ("fact", "sum")}
    if "target" in raw:
        aggregates["raw_plan"] = ("target", "sum")
    aggregates["raw_rows"] = ("fact", "size")
    return raw.groupby(["date", *config.segments], as_index=False, dropna=False).agg(**aggregates)


def load_plan(config, month, allow_missing=False):
    """Read target only: target-month fact/debt columns are never selected."""
    table_name = config.plan_table or config.raw_table
    table = identifier(table_name)
    source = [*actual_keys(config), config.plan_value_column]
    if config.plan_available_column:
        source.append(config.plan_available_column)
    start = month_start(month)
    end = start + pd.offsets.MonthBegin()
    date_expr = sql_date("report_dt", config.raw_date_encoding)
    with readonly_connection(config.db_path) as conn:
        present = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
        missing = set([*source, "is_forecast"]) - present
        if missing:
            raise ValueError(f"Plan table {table_name}: missing columns {sorted(missing)}")
        frame = pd.read_sql_query(
            f"SELECT {', '.join(identifier(c) for c in source)} FROM {table} "
            f"WHERE is_forecast=? AND {date_expr} >= ? AND {date_expr} < ?",
            conn, params=[config.plan_is_forecast, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")])
    return aggregate_plan(prepare_plan(frame, config), config, month, allow_missing=allow_missing)


def prepare_plan(frame, config):
    frame = with_report_date(frame)
    columns = [*actual_keys(config), config.plan_value_column]
    rename = {config.plan_value_column: "plan"}
    if config.plan_available_column:
        columns.append(config.plan_available_column)
        rename[config.plan_available_column] = "plan_available_at"
    missing = set(columns) - set(frame)
    if missing:
        raise ValueError(f"Plan frame: missing columns {sorted(missing)}")
    if "is_forecast" in frame:
        kinds = pd.to_numeric(frame.is_forecast, errors="raise")
        if not kinds.isin([0, 1]).all():
            raise ValueError("Plan is_forecast values must be 0 or 1")
        frame = frame.loc[kinds.eq(config.plan_is_forecast)].copy()
    frame = normalize_segments(frame[columns].rename(columns=rename), config.segments)
    for col in ["report_month", "report_dt", "dt_in"]:
        frame[col] = parse_dates(frame[col], config.raw_date_encoding).dt.normalize()
        if frame[col].isna().any():
            raise ValueError(f"Missing date in plan: {col}")
    if not frame.report_month.eq(frame.report_dt.dt.to_period("M").dt.start_time).all():
        raise ValueError("Plan report_month and report_dt disagree")
    if "plan_available_at" in frame:
        frame["plan_available_at"] = parse_dates(frame.plan_available_at)
        if frame.plan_available_at.isna().any():
            raise ValueError("Missing plan availability timestamp")
    return frame


def split_dated_frame(df, config):
    """Adapt date/fact/plan rows to separate history, plan and evaluation inputs."""
    frame = df.copy()
    if "date" not in frame and "report_dt" in frame:
        frame["date"] = frame.report_dt
    if "plan" not in frame and "target" in frame:
        frame["plan"] = frame.target
    if "fact" not in frame and {"w1", "w2", "cession", "write_off"} <= set(frame):
        components = frame[["w1", "w2", "cession", "write_off"]].apply(pd.to_numeric, errors="raise")
        frame["fact"] = components.w1 - components.w2 - components.cession - components.write_off
    missing = {"date", "fact", "plan"} - set(frame)
    if missing:
        raise ValueError(f"run(df) needs columns date, fact, plan. Missing: {sorted(missing)}")
    frame["report_dt"] = parse_dates(frame.date, config.raw_date_encoding).dt.normalize()
    if frame.report_dt.isna().any():
        raise ValueError("Missing date in run(df) input")
    frame["report_month"] = frame.report_dt.dt.to_period("M").dt.start_time
    if "dt_in" not in frame:
        frame["dt_in"] = frame.report_month
    for col in config.segments:
        if col not in frame:
            frame[col] = "0" if col == "inflow" else "ВСЕ"
    frame["target"] = frame.plan
    start = month_start(config.forecast_month)
    end = start + pd.offsets.MonthBegin()
    actual_rows = pd.Series(True, index=frame.index)
    plan_rows = actual_rows.copy()
    if "is_forecast" in frame:
        flags = pd.to_numeric(frame.is_forecast, errors="raise")
        if not flags.isin([0, 1]).all():
            raise ValueError("is_forecast must be 0 or 1")
        actual_rows = flags.eq(0)
        # Historical training plans come from actual snapshots. Target plans use
        # the explicitly configured source; never sum both source kinds together.
        plan_rows = (frame.report_dt.lt(start) & flags.eq(0)) | (frame.report_dt.ge(start) & flags.eq(config.plan_is_forecast))
    history = frame.loc[frame.report_dt.lt(start) & frame.fact.notna() & actual_rows]
    plan_cols = [*actual_keys(config), "target"]
    if config.plan_available_column:
        plan_cols.append(config.plan_available_column)
    plan = frame.loc[frame.report_dt.lt(end) & frame.target.notna() & plan_rows, plan_cols]
    actual = frame.loc[frame.report_dt.ge(start) & frame.report_dt.lt(end) & frame.fact.notna() & actual_rows]
    return history, plan, actual


def aggregate_plan(rows, config, month, allow_missing=False):
    dims = list(config.segments)
    frame = rows.loc[rows.report_month.eq(month_start(month))].copy()
    if frame.empty and not allow_missing:
        raise ValueError(f"No daily plan for {month}")
    frame["date"] = frame.report_dt
    key = ["date", "dt_in", *dims]
    if "plan_available_at" in frame:
        frame = frame[frame.plan_available_at <= origin(month)]
        if frame.empty and not allow_missing:
            raise ValueError(f"No plan version available before {month}")
        if frame.duplicated([*key, "plan_available_at"]).any():
            raise ValueError("Duplicate plan cohort and availability timestamp")
        frame = frame.sort_values("plan_available_at").drop_duplicates(key, keep="last")
    elif frame.duplicated(key).any():
        raise ValueError("Duplicate daily plan cohort; do not sum repeated snapshots")
    if allow_missing:
        # An absent future plan is unknown, never an official zero or a reason to
        # stop the monetary forecast. Expand each cohort before summing segments.
        frame["plan"] = pd.to_numeric(frame.plan, errors="raise")
        if np.isinf(frame.plan).any() or frame.plan.lt(0).any():
            raise ValueError("Plans must be nonnegative and finite when present")
        parts = []
        for _, group in frame.groupby(["dt_in", *dims], sort=False):
            grid = group.set_index("date").plan.reindex(days(month))
            entry = max(month_start(month), group.dt_in.iloc[0])
            grid.loc[grid.index < entry] = 0.0
            if config.plan_mode == "cumulative":
                source = grid.copy()
                grid = source.diff()
                grid.iloc[0] = source.iloc[0]
            part = pd.DataFrame({"date": grid.index, "plan": grid.to_numpy()})
            for dim in dims:
                part[dim] = group[dim].iloc[0]
            parts.append(part)
        if not parts:
            return pd.DataFrame({"date": pd.Series(dtype="datetime64[ns]"),
                                 **{d: pd.Series(dtype=str) for d in dims}, "plan": pd.Series(dtype=float)})
        result = pd.concat(parts).groupby(["date", *dims], as_index=False).plan.agg(lambda s: s.sum(skipna=False))
        if result.plan.lt(0).any():
            raise ValueError("Daily plans must be nonnegative")
        return result.sort_values([*dims, "date"]).reset_index(drop=True)
    numeric(frame, ["plan"])
    frame = frame.reset_index(drop=True)
    frame["plan"] = cohort_daily(frame, "plan", config.plan_mode, config)
    if frame.plan.lt(0).any():
        raise ValueError("Daily plans must be nonnegative; resolve corrections explicitly")
    # dt_in is retained until here: each cohort contributes once to its segment/day.
    result = frame.groupby(["date", *dims], as_index=False).plan.sum()
    first_entry = frame.groupby(dims, as_index=False).dt_in.min()
    expected_days = days(month)
    # Before the first planned inflow cohort exists its plan is structurally zero.
    # Other absent days remain errors, never guessed from payment observations.
    added = []
    for key_values, part in result.groupby(dims, sort=False):
        key_values = key_values if isinstance(key_values, tuple) else (key_values,)
        leaf = dict(zip(dims, key_values))
        missing_days = expected_days.difference(part.date)
        if len(missing_days) and leaf.get("inflow") == "1":
            entry = first_entry
            for col, value in leaf.items():
                entry = entry[entry[col].eq(value)]
            first_day = entry.dt_in.iloc[0]
            structural = missing_days[missing_days < first_day]
            added.extend({"date": d, **leaf, "plan": 0.0} for d in structural)
    if added:
        result = pd.concat([result, pd.DataFrame(added)], ignore_index=True)
    counts = result.groupby(dims).date.nunique()
    if not counts.eq(len(expected_days)).all():
        raise ValueError(f"{month}: every segment needs a plan for every day; missing plan rows are not assumed zero")
    if result.plan.sum() <= 0:
        raise ValueError(f"{month}: the entire monthly plan is zero; efficiency cannot be assessed")
    return result.sort_values([*dims, "date"]).reset_index(drop=True)


def read_input(path, required, config):
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"Input not found: {p}. See input/*_template.csv and README.md")
    f = pd.read_parquet(p) if p.suffix.lower() == ".parquet" else pd.read_csv(p, dtype={c: str for c in config.segments})
    missing = set(required) - set(f)
    if missing:
        raise ValueError(f"{p.name}: missing columns {sorted(missing)}")
    if f.empty:
        raise ValueError(f"{p.name} is empty")
    f = normalize_segments(f, config.segments)
    for c in ["date", "known_at", "snapshot_date"]:
        if c in f:
            f[c] = parse_dates(f[c])
            if f[c].isna().any():
                raise ValueError(f"{p.name}: missing {c}")
            if c != "known_at":
                f[c] = f[c].dt.normalize()
    if "month" in f:
        f["month"] = pd.to_datetime(f["month"], format="mixed", errors="raise").dt.strftime("%Y-%m")
    return f


class Inputs:
    def __init__(self, config, *, plan_df=None):
        self.config = config
        dims = list(config.segments)
        self._plan_cache = {}
        self._plan_rows = None if plan_df is None else prepare_plan(plan_df, config)
        self.portfolio = self._optional(config.portfolio_path,
            ["month", "snapshot_date", "known_at", "debt", "acc_count", *dims])
        self.inflow = self._optional(config.inflow_path,
            ["month", "date", "known_at", "incoming_debt", *dims])
        self.operations = self._optional(config.operations_path, ["date", "known_at", *dims])
        self.reference = self._optional(config.reference_path,
            ["month", "date", "known_at", "forecast_daily", *dims])
        for frame, columns in [(self.portfolio, ["debt", "acc_count"]),
                               (self.inflow, ["incoming_debt"]), (self.reference, ["forecast_daily"])]:
            if frame is not None:
                numeric(frame, columns)
                if "date" in frame and (frame.date.dt.strftime("%Y-%m") != frame.month).any():
                    raise ValueError("Input month and date disagree")
        if self.operations is not None:
            opcols = [c for c in self.operations if c.startswith("op_")]
            if not opcols:
                raise ValueError("Operations input needs numeric op_* columns")
            numeric(self.operations, opcols)
        self.calendar = None
        if config.calendar_path:
            self.calendar = pd.read_csv(config.calendar_path)
            if not {"date", "is_workday"} <= set(self.calendar):
                raise ValueError("Calendar needs date,is_workday columns")
            self.calendar["date"] = parse_dates(self.calendar.date).dt.normalize()
            if self.calendar.date.duplicated().any() or not self.calendar.is_workday.isin([0, 1]).all():
                raise ValueError("Calendar dates must be unique, is_workday must be 0/1")
            if "is_holiday" in self.calendar and not self.calendar.is_holiday.isin([0, 1]).all():
                raise ValueError("Calendar is_holiday must be 0/1")
        elif config.require_calendar_overrides:
            raise ValueError("Provide calendar_path with the approved production calendar, including transfers")

    def _optional(self, path, required):
        return read_input(path, required, self.config) if path else None

    def vintage(self, frame, month, value_columns, extra_keys=()):
        if frame is None:
            return None
        f = frame[(frame.month == str(month)) & (frame.known_at <= origin(month))].copy()
        key = [*self.config.segments, *extra_keys]
        if f.duplicated([*key, "known_at"]).any():
            raise ValueError("Duplicate input grain and known_at; aggregate cohorts before supplying input")
        f = f.sort_values("known_at").drop_duplicates(key, keep="last")
        return f[[*key, *value_columns]]

    def plan(self, month):
        month = str(month)
        if month not in self._plan_cache:
            if self._plan_rows is None:
                self._plan_cache[month] = load_plan(self.config, month, allow_missing=month == self.config.forecast_month)
            else:
                self._plan_cache[month] = aggregate_plan(self._plan_rows, self.config, month,
                                                        allow_missing=month == self.config.forecast_month)
        return self._plan_cache[month].copy()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()
