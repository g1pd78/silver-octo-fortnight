"""Model-only estimates of an unknown plan tail; official plans are never filled."""
import numpy as np
import pandas as pd
from .config import days, origin
from .data import PlanUnavailableError, raw_asof


def complete_plan_tail(history, plan, config, month):
    dims = list(config.segments)
    result = plan.copy()
    result["plan_estimated"] = False
    if result.empty or not result.plan.isna().any():
        return result
    h = raw_asof(history, config, origin(month))
    historical = h.groupby(["date", *dims], as_index=False).agg(
        plan=("target", lambda s: s.sum(skipna=False)))
    dates = days(month)
    progress = np.arange(1, len(dates) + 1) / len(dates)
    for values, group in result.groupby(dims, sort=False):
        group = group.sort_values("date")
        if not group.plan.isna().any():
            continue
        if not pd.DatetimeIndex(group.date).equals(dates):
            raise PlanUnavailableError("Model plan needs a full date grid")
        known = group.plan.notna().to_numpy()
        count = int(known.sum())
        if not count or not known[:count].all() or known[count:].any():
            raise PlanUnavailableError("Only a missing plan tail after a known prefix can be estimated")
        anchor = float(group.plan.iloc[:count].sum())
        if anchor <= 0:
            raise PlanUnavailableError("A model plan tail needs a positive known cumulative anchor")
        values = values if isinstance(values, tuple) else (values,)
        past = historical
        for dim, value in zip(dims, values):
            past = past[past[dim].eq(value)]
        shapes = []
        for previous, rows in past.groupby(past.date.dt.to_period("M"), sort=True):
            rows = rows.sort_values("date")
            if not pd.DatetimeIndex(rows.date).equals(days(str(previous))) or rows.plan.isna().any():
                continue
            cumulative = rows.plan.cumsum().to_numpy()
            if cumulative[-1] <= 0 or (cumulative < 0).any():
                continue
            shapes.append(np.interp(progress, np.linspace(0, 1, len(rows) + 1),
                                    np.r_[0., cumulative / cumulative[-1]]))
        if not shapes:
            raise PlanUnavailableError("No complete historical plan curve for estimating the missing tail")
        shape = np.median(shapes[-config.reference_months:], axis=0)
        if shape[count - 1] <= 0:
            raise PlanUnavailableError("Historical plan curve is zero at the known anchor")
        cumulative = anchor * shape / shape[count - 1]
        tail = np.diff(np.r_[anchor, cumulative[count:]])
        if not np.isfinite(tail).all() or (config.plan_mode == "daily" and (tail < 0).any()):
            raise PlanUnavailableError("Historical plan tail cannot supply valid daily model amounts")
        result.loc[group.index[count:], "plan"] = tail
        result.loc[group.index[count:], "plan_estimated"] = True
    return result
