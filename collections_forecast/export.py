"""Database upload at one explicit grain, separate from analytical reports."""
import numpy as np
import pandas as pd
from .config import days, origin


def upload_frame(pred, config):
    """One daily monetary total per date; never attach totals to segment rows."""
    dims = list(config.segments)
    keys = ["date", *dims]
    required = [*keys, "forecast_daily", "plan"]
    missing = set(required) - set(pred)
    if missing:
        raise ValueError(f"Upload: missing columns {sorted(missing)}")
    if pred.empty or pred[keys].isna().any().any() or pred.duplicated(keys).any():
        raise ValueError(f"Upload: nonempty unique segment/date keys required: {keys}")
    f = pred[required].copy()
    f["date"] = pd.to_datetime(f.date).dt.normalize()
    if f.duplicated(keys).any():
        raise ValueError("Upload: duplicate segment/date keys after date normalization")
    expected = days(config.forecast_month)
    for _, group in f.groupby(dims):
        if not pd.DatetimeIndex(group.date.sort_values()).equals(expected):
            raise ValueError("Upload: every segment must cover the entire target month")
    f["forecast_daily"] = pd.to_numeric(f.forecast_daily, errors="raise")
    f["plan"] = pd.to_numeric(f.plan, errors="raise")
    if not np.isfinite(f.forecast_daily).all():
        raise ValueError("Upload: forecast_daily must be finite monetary amounts")
    if np.isinf(f.plan).any() or f.plan.lt(0).any():
        raise ValueError("Upload: plan must be nonnegative; missing plan stays NULL")
    total = f.groupby("date", as_index=False).agg(
        forecast_daily=("forecast_daily", "sum"),
        plan_daily=("plan", lambda s: s.sum(skipna=False)))
    total.insert(0, "forecast_month", pd.Period(config.forecast_month, "M").start_time)
    total["forecast_date"] = origin(config.forecast_month).normalize()
    total["forecast_model"] = "collections.v2.total"
    return total


def database_upload_frame(pred, config):
    """RR forecast table: one cumulative snapshot per date and model segment.

    Cohorts have already been aggregated by the model: do not join dt_in back
    from raw RR or repeat the segment forecast for every source cohort.
    """
    upload_frame(pred, config)  # validate complete unique source grain first
    dims = list(config.segments)
    rr_dims = ["subproduct", "dpd_bucket", "pvr_bucket", "terbank", "inflow"]
    if set(dims) - set(rr_dims):
        raise ValueError("RR upload requires segment names from the forecast table: " + ", ".join(rr_dims))
    f = pred.sort_values([*dims, "date"]).reset_index(drop=True).copy()
    result = pd.DataFrame({"report_month": pd.Period(config.forecast_month, "M").start_time,
                           "report_dt": f.date, "dt_in": ""})
    defaults = {"subproduct": "", "dpd_bucket": "", "pvr_bucket": "0. [undef]", "terbank": "МБ", "inflow": "0"}
    for col in rr_dims:
        result[col] = f[col].astype(str) if col in dims else defaults[col]
        result[col] = result[col].replace({"ВСЕ": defaults[col], "__UNDEFINED__": defaults[col]})
    result["inflow"] = pd.to_numeric(result.inflow, errors="raise")
    if not result.inflow.isin([0, 1]).all():
        raise ValueError("RR upload inflow must be 0 or 1")
    result["inflow"] = result.inflow.astype(int)
    # RR fact/plan columns are snapshots ON the date, not daily increments.
    result["fact"] = f.groupby(dims).forecast_daily.cumsum()
    result["plan"] = f.groupby(dims).plan.transform(lambda s: s.cumsum(skipna=False))
    result["dolg"] = f["debt"] if "debt" in f else np.nan
    if "model" in f:
        result.loc[f.model.str.startswith("fallback_"), "dolg"] = np.nan
    result["is_forecast"] = 1
    result["forecast_date"] = origin(config.forecast_month).normalize()
    result["forecast_model"] = "collections.v2"
    key = ["report_month", "report_dt", "dt_in", *rr_dims, "forecast_date", "forecast_model"]
    if result.duplicated(key).any():
        raise ValueError("RR upload: distinct model segments collapse to the same database key; resolve segment labels")
    return result[["report_month", "report_dt", "dt_in", *rr_dims, "dolg", "plan", "fact",
                   "is_forecast", "forecast_date", "forecast_model"]]
