"""Generated DB-loader adapter; rebuild with tools/build_standalone_notebook.py."""
import numpy as np
import pandas as pd


def days(month):
    p = pd.Period(month, freq="M")
    return pd.date_range(p.start_time, p.end_time.normalize(), freq="D")


def month_start(month):
    return pd.Period(month, freq="M").start_time


def origin(month):
    return month_start(month) - pd.Timedelta(nanoseconds=1)


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


def aggregate_database_upload(frame, *, debt_by_day=None, report_month=None, forecast_model=None):
    """Prepare ONE portfolio snapshot per date for the aggregate dashboard loader.

    Accepts forecast_upload.csv, whose fact/plan are already cumulative.
    Optional debt_by_day is a display denominator, not a model feature. Target
    actual debt must never be fed back into model fitting or origin backtests.
    """
    columns = ["report_month", "report_dt", "dt_in", "subproduct", "dpd_bucket", "pvr_bucket",
               "terbank", "inflow", "dolg", "plan", "fact", "is_forecast", "forecast_date", "forecast_model"]
    missing = set(columns) - set(frame)
    if missing:
        raise ValueError(f"Use forecast_upload.csv (RR cumulative schema); missing columns {sorted(missing)}")
    f = frame[columns].copy()
    if f.empty:
        raise ValueError("Empty forecast upload")
    for col in ["report_month", "report_dt", "forecast_date"]:
        f[col] = pd.to_datetime(f[col], errors="raise").dt.normalize()
        if f[col].isna().any():
            raise ValueError(f"Upload: missing {col}")
    for col in ["report_month", "forecast_date", "forecast_model"]:
        if f[col].isna().any() or f[col].nunique() != 1:
            raise ValueError(f"Aggregate one {col} at a time; do not mix forecast vintages or models")
    month = f.report_month.iloc[0].to_period("M")
    if not f.report_month.eq(month.start_time).all() or not f.report_dt.dt.to_period("M").eq(month).all():
        raise ValueError("Upload report_month/report_dt mismatch")
    if report_month is not None and pd.Period(report_month, "M") != month:
        raise ValueError("CSV month differs from REPORT_MONTH; do not relabel October rows as November")
    dims = ["dt_in", "subproduct", "dpd_bucket", "pvr_bucket", "terbank", "inflow"]
    for col in dims:
        f[col] = f[col].astype("string").fillna("").str.strip()
    f["inflow"] = pd.to_numeric(f.inflow, errors="raise")
    if not f.inflow.isin([0, 1]).all() or not pd.to_numeric(f.is_forecast, errors="raise").eq(1).all():
        raise ValueError("Upload needs is_forecast=1 and inflow=0/1")
    key = ["report_dt", *dims]
    if f.duplicated(key).any():
        raise ValueError("Duplicate forecast segment/date; aggregation would double the money")
    for _, group in f.groupby(dims, dropna=False):
        if not pd.DatetimeIndex(group.report_dt.sort_values()).equals(days(str(month))):
            raise ValueError("Every forecast segment must cover the entire month; segment counts may not change after day 25")
    for col in ["fact", "plan", "dolg"]:
        f[col] = pd.to_numeric(f[col], errors="raise")
        if np.isinf(f[col]).any() or (col == "fact" and f[col].isna().any()):
            raise ValueError(f"Upload: invalid {col}")
    if f.plan.lt(0).any() or f.dolg.lt(0).any():
        raise ValueError("Upload: plan/debt cannot be negative")
    # Sum ACROSS segments on the same date. Never cumsum snapshots again.
    total = f.groupby("report_dt", as_index=False)[["fact", "plan", "dolg"]].agg(lambda s: s.sum(skipna=False))
    if debt_by_day is not None:
        debt = debt_by_day.copy()
        if "dolg" not in debt:
            raise ValueError("debt_by_day needs dolg and report_dt or report_day")
        if "report_dt" in debt:
            debt["report_dt"] = pd.to_datetime(debt.report_dt, errors="raise").dt.normalize()
        elif "report_day" in debt:
            numbers = pd.to_numeric(debt.report_day, errors="raise")
            if not (numbers.notna() & numbers.between(1, month.days_in_month) & numbers.eq(numbers.round())).all():
                raise ValueError("Invalid debt report_day for the target month")
            debt["report_dt"] = month.start_time + pd.to_timedelta(numbers - 1, unit="D")
        else:
            raise ValueError("debt_by_day needs report_dt or report_day")
        if debt.report_dt.isna().any() or debt.report_dt.duplicated().any() or not debt.report_dt.isin(total.report_dt).all():
            raise ValueError("Debt needs one total per date in the target month; aggregate it before mapping")
        debt["dolg"] = pd.to_numeric(debt.dolg, errors="raise")
        if np.isinf(debt.dolg).any() or debt.dolg.lt(0).any():
            raise ValueError("Invalid debt amount")
        # Reindex chronologically BEFORE forward filling, not in segment-row order.
        # Leading unknown debt stays NULL; it is never invented as zero.
        mapped = debt.set_index("report_dt").dolg.reindex(total.report_dt).ffill()
        total["dolg"] = mapped.to_numpy()
    total["report_month"] = month.start_time
    total["dt_in"] = ""
    total["subproduct"] = total["dpd_bucket"] = ""
    total["terbank"] = "МБ"
    total["pvr_bucket"] = "0. [undef]"
    total["inflow"] = 0  # Existing dashboard convention for this ALL-portfolio series.
    total["is_forecast"] = 1
    total["forecast_date"] = f.forecast_date.iloc[0]
    total["forecast_model"] = forecast_model if forecast_model is not None else f.forecast_model.iloc[0]
    if not str(total.forecast_model.iloc[0]).strip():
        raise ValueError("forecast_model cannot be empty")
    return total[columns]


def legacy_dashboard_upload(frame, *, debt_by_day=None, report_month=None, forecast_model=None):
    """Exact aggregate identifiers used by the user's existing uploader.

    forecast_date here is the dashboard slice label (month start). The actual
    information cutoff remains in the prediction origin and run manifest.
    """
    total = aggregate_database_upload(frame, debt_by_day=debt_by_day,
                                      report_month=report_month, forecast_model=forecast_model)
    total["dt_in"] = total.report_month
    total["forecast_date"] = total.report_month
    return total


def filtered_dashboard_upload(total_forecast, layout, config, *, forecast_model="collections.v2.allocated"):
    """Allocate the frozen total to an explicit RR cohort layout for UI filters.

    layout contains COMPLETE cumulative plan/debt snapshots available at the
    forecast origin. Its factual collections are never read. This is allocation,
    not independently validated segment prediction. Output matches the RR schema.
    """
    columns = ["report_month", "report_dt", "dt_in", "subproduct", "dpd_bucket", "terbank",
               "pvr_bucket", "inflow", "dolg", "plan", "fact", "is_forecast", "forecast_date", "forecast_model"]
    dims = ["subproduct", "dpd_bucket", "terbank", "pvr_bucket", "inflow"]
    required = ["dt_in", *dims]
    f = layout.copy()
    for dst, src in [("report_dt", "date"), ("plan", "target"), ("dolg", "w1")]:
        if dst not in f and src in f:
            f[dst] = f[src]
    missing = set(["report_dt", "plan", "dolg", *required]) - set(f)
    if missing:
        raise ValueError(f"Filtered CSV needs real cohort fields, no synthetic labels: {sorted(missing)}")
    f["report_dt"] = parse_dates(f.report_dt, config.raw_date_encoding).dt.normalize()
    target_month = pd.Period(config.forecast_month, "M")
    f = f.loc[f.report_dt.dt.to_period("M").eq(target_month)].copy()
    if "is_forecast" in f:
        flags = pd.to_numeric(f.is_forecast, errors="raise")
        if not flags.isin([0, 1]).all():
            raise ValueError("Layout is_forecast must be 0/1")
        f = f.loc[flags.eq(config.plan_is_forecast)].copy()
    if f.empty:
        raise ValueError("No cohort layout for the target month and selected plan source")
    f["dt_in"] = parse_dates(f.dt_in, config.raw_date_encoding).dt.normalize()
    if f.dt_in.isna().any() or f.dt_in.gt(f.report_dt).any():
        raise ValueError("Layout dt_in must be a real cohort entry date on or before report_dt")
    if "report_month" in f:
        m = parse_dates(f.report_month, config.raw_date_encoding).dt.normalize()
        if not m.eq(target_month.start_time).all():
            raise ValueError("Layout report_month and report_dt disagree")
    f["report_month"] = target_month.start_time
    for col in dims[:-1]:
        if f[col].isna().any():
            raise ValueError(f"Layout {col} is missing; use actual source labels for dashboard filters")
        f[col] = f[col].astype("string").str.strip()
        if f[col].isin(["", "ВСЕ", "__UNDEFINED__"]).any():
            raise ValueError(f"Layout {col} contains aggregate placeholders; real segment labels are required")
    # Excel can silently turn the textual bucket '1-30' into a date. Do not emit
    # date-like buckets and pretend they match the dashboard's category buttons.
    if f.dpd_bucket.str.match(r"^(?:\d{4}-\d{2}-\d{2}|\d{2}\.\d{2}\.\d{4})(?:\s.*)?$").any():
        raise ValueError("DPD bucket looks like an Excel-converted date; use original text labels from the database")
    f["inflow"] = pd.to_numeric(f.inflow, errors="raise")
    if not f.inflow.isin([0, 1]).all():
        raise ValueError("Layout inflow must be 0 or 1")
    f["inflow"] = f.inflow.astype(int)
    key = ["report_dt", "dt_in", *dims]
    available = config.plan_available_column
    if available:
        if available not in f:
            raise ValueError(f"Layout needs the configured plan availability column {available}")
        f[available] = parse_dates(f[available])
        if f[available].isna().any():
            raise ValueError("Missing layout availability timestamp")
        f = f.loc[f[available].le(origin(config.forecast_month))]
        if f.duplicated([*key, available]).any():
            raise ValueError("Duplicate cohort layout vintage")
        f = f.sort_values(available).drop_duplicates(key, keep="last")
    if f.duplicated(key).any():
        raise ValueError("Duplicate cohort/date layout; it would multiply forecast amounts")
    for col in ["plan", "dolg"]:
        f[col] = pd.to_numeric(f[col], errors="raise")
        if np.isinf(f[col]).any() or f[col].lt(0).any() or (col == "dolg" and f[col].isna().any()):
            raise ValueError(f"Layout {col} must contain nonnegative money; debt cannot be unknown for RR filters")
    total = total_forecast.copy()
    if not {"report_dt", "fact", "plan", "forecast_date"} <= set(total):
        raise ValueError("Pass result['dashboard_upload'] as the total forecast")
    total["report_dt"] = pd.to_datetime(total.report_dt).dt.normalize()
    total["forecast_date"] = pd.to_datetime(total.forecast_date, errors="raise").dt.normalize()
    if total.forecast_date.isna().any() or total.forecast_date.nunique() != 1:
        raise ValueError("Allocate one forecast release at a time")
    if total.report_dt.duplicated().any() or not pd.DatetimeIndex(total.report_dt.sort_values()).equals(days(config.forecast_month)):
        raise ValueError("The total forecast must contain exactly one row per target-month date")
    total["fact"] = pd.to_numeric(total.fact, errors="raise")
    if not np.isfinite(total.fact).all():
        raise ValueError("Non-finite total forecast")
    total["plan"] = pd.to_numeric(total.plan, errors="raise")
    totals = f.groupby("report_dt", as_index=False).agg(
        source_plan=("plan", lambda s: s.sum(skipna=False)), source_debt=("dolg", "sum"))
    checked = total.merge(totals, on="report_dt", how="left", validate="one_to_one", indicator=True)
    if checked._merge.ne("both").any():
        raise ValueError("Cohort layout does not cover every forecast date; missing dates will not be fabricated")
    comparable = checked.plan.notna() & checked.source_plan.notna()
    if (checked.plan.isna() != checked.source_plan.isna()).any() or not np.allclose(
            checked.loc[comparable, "plan"], checked.loc[comparable, "source_plan"], atol=.01, rtol=1e-10):
        raise ValueError("Cumulative layout plan totals differ from the frozen forecast plan; check scope, source version and cumulative units")
    f = f.merge(checked[["report_dt", "fact", "forecast_date", "source_plan", "source_debt"]].rename(
        columns={"fact": "_total_fact"}), on="report_dt", how="left", validate="many_to_one", suffixes=("_ignored", ""))
    use_plan = f.source_plan.gt(0) & f.source_plan.notna()
    weight = pd.Series(np.where(use_plan, f.plan / f.source_plan, f.dolg / f.source_debt.replace(0, np.nan)), index=f.index)
    unsupported = weight.isna() & f._total_fact.ne(0)
    if unsupported.any():
        raise ValueError("Cannot allocate a nonzero forecast where both total plan and debt are zero")
    f["fact"] = f._total_fact * weight.fillna(0)
    # Put floating-point residual on one eligible cohort, never repeat the total.
    for _, indices in f.groupby("report_dt").groups.items():
        part = f.loc[indices]
        index = weight.loc[indices].fillna(0).idxmax()
        f.loc[index, "fact"] += part._total_fact.iloc[0] - part.fact.sum()
    f["is_forecast"] = 1
    if not str(forecast_model).strip():
        raise ValueError("Choose a nonempty allocated forecast model name")
    f["forecast_model"] = forecast_model
    return f[columns].sort_values(key).reset_index(drop=True)


def read_filtered_dashboard_csv(path):
    """Read our UTF-8/ISO CSV without destroying filter labels or cohort keys."""
    labels = ["subproduct", "dpd_bucket", "terbank", "pvr_bucket", "forecast_model"]
    columns = ["report_month", "report_dt", "dt_in", *labels[:4], "inflow", "dolg", "plan", "fact",
               "is_forecast", "forecast_date", "forecast_model"]
    f = pd.read_csv(path, encoding="utf-8-sig", dtype={c: "string" for c in labels})
    missing = set(columns) - set(f)
    if missing:
        raise ValueError(f"Filtered upload CSV missing columns: {sorted(missing)}")
    for col in ["report_month", "report_dt", "dt_in", "forecast_date"]:
        f[col] = pd.to_datetime(f[col], format="%Y-%m-%d", errors="raise")
        if f[col].isna().any():
            raise ValueError(f"Filtered CSV has missing {col}")
    for col in labels:
        if f[col].isna().any() or f[col].str.strip().eq("").any():
            raise ValueError(f"Filtered CSV has missing {col}")
    for col in ["inflow", "dolg", "plan", "fact", "is_forecast"]:
        f[col] = pd.to_numeric(f[col], errors="raise")
        if np.isinf(f[col]).any() or (col != "plan" and f[col].isna().any()):
            raise ValueError(f"Filtered CSV has invalid {col}")
    if f.empty or not f.inflow.isin([0, 1]).all() or not f.is_forecast.eq(1).all():
        raise ValueError("Filtered CSV needs rows with inflow=0/1 and is_forecast=1")
    if f.plan.lt(0).any() or f.dolg.lt(0).any() or f.dt_in.gt(f.report_dt).any():
        raise ValueError("Filtered CSV has negative plan/debt or a future cohort entry")
    if not f.report_month.eq(f.report_dt.dt.to_period("M").dt.start_time).all():
        raise ValueError("Filtered CSV month/date mismatch")
    key = [c for c in columns if c not in ["dolg", "plan", "fact", "is_forecast"]]
    if f.duplicated(key).any():
        raise ValueError("Filtered CSV contains duplicate forecast keys")
    if f[["report_month", "forecast_date", "forecast_model"]].drop_duplicates().shape[0] != 1:
        raise ValueError("Upload one month/model/release slice at a time")
    return f[columns]
