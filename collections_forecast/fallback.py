"""Money-only forecast when the official future plan is unavailable."""
import numpy as np
import pandas as pd
from .config import days, origin
from .data import raw_asof, daily_actual


def money_forecast(history, plan, config):
    dims = list(config.segments)
    h = daily_actual(raw_asof(history, config, origin(config.forecast_month)), config)
    dates = days(config.forecast_month)
    # Include recent portfolio segments even when a whole segment's plan is absent.
    recent = h[h.date >= dates[0] - pd.DateOffset(months=config.reference_months)]
    if recent.empty:
        recent = h
    segments = pd.concat([recent[dims], plan[dims]]).drop_duplicates()
    if segments.empty:
        raise ValueError("A money forecast needs historical observations or at least one planned segment")
    output = []
    for values in segments.itertuples(index=False, name=None):
        past, known = h, plan
        for dim, value in zip(dims, values):
            past = past[past[dim].eq(value)]
            known = known[known[dim].eq(value)]
        curve_samples = []
        for month, group in past.groupby(past.date.dt.to_period("M")):
            group = group.sort_values("date")
            if pd.DatetimeIndex(group.date).equals(days(str(month))):
                curve_samples.append(np.interp(np.arange(1, len(dates) + 1) / len(dates),
                                               np.linspace(0, 1, len(group) + 1), np.r_[0., group.fact.cumsum()]))
        official = known.set_index("date").plan.reindex(dates)
        if curve_samples:
            cumulative = np.mean(curve_samples[-config.reference_months:], axis=0)
            method = "history_curve"
        elif not past.empty:
            # Incomplete history still supports a weekday estimate; missing dates
            # are not treated as zero payments.
            average = past.groupby(past.date.dt.dayofweek).fact.mean()
            daily = pd.Series(dates.dayofweek).map(average).fillna(past.fact.mean()).to_numpy()
            cumulative, method = np.cumsum(daily), "history_weekday"
        else:
            # A new segment with no observations: use its available plan as a
            # provisional money estimate, explicitly labelled as cold start.
            daily = official.fillna(official.mean()).fillna(0).to_numpy()
            cumulative, method = np.cumsum(daily), "cold_start_plan_or_zero"
        part = pd.DataFrame({"date": dates, **dict(zip(dims, values)), "plan": official.to_numpy(),
                             "forecast_cumulative": cumulative, "forecast_daily": np.diff(np.r_[0., cumulative])})
        part["plan_cumulative"] = part.plan.cumsum(skipna=False)
        part["plan_month"] = part.plan.sum(skipna=False)
        part["plan_missing"] = part.plan.isna()
        part["base_forecast_cumulative"] = cumulative
        part["reconciliation_cumulative"] = part["reconciliation_daily"] = 0.0
        part["debt"] = 0.0
        part["origin"] = origin(config.forecast_month)
        part["model"] = "fallback_" + method
        output.append(part)
    return pd.concat(output, ignore_index=True).sort_values([*dims, "date"]).reset_index(drop=True)
