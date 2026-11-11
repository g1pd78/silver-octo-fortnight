"""Generated DB-loader adapter; rebuild with tools/build_standalone_notebook.py."""
import numpy as np
import pandas as pd


def days(month):
    p = pd.Period(month, freq="M")
    return pd.date_range(p.start_time, p.end_time.normalize(), freq="D")


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
