"""Candidate models fitted without observations from the forecast month."""
import warnings
import numpy as np
import pandas as pd
from .config import month_start, origin, days
from .data import raw_asof, daily_actual


def label_features(features, actual, config):
    dims = list(config.segments)
    f = features.merge(actual[["date", *dims, "fact"]], on=["date", *dims], how="outer", validate="one_to_one", indicator=True)
    if f.fact.isna().any() or f._merge.ne("both").any():
        return None
    f = f.drop(columns="_merge").sort_values([*dims, "date"]).reset_index(drop=True)
    f["fact_cumulative"] = f.groupby(dims).fact.cumsum()
    return f


def training_data(history, month, config, builder):
    available = raw_asof(history, config, origin(month))
    actual = daily_actual(available, config)
    actual["month"] = actual.date.dt.strftime("%Y-%m")
    frames, totals, skipped = [], [], []
    for previous in sorted(available.month.unique()):
        if previous >= str(month):
            continue
        try:
            x = builder.build(history, previous)
            a = actual[actual.month == previous]
            f = label_features(x, a, config)
            if f is None:
                raise ValueError("missing segment-day facts; zeros were not imputed")
            total = builder.total(x)
            actual_total = a.groupby("date").fact.sum().reindex(total.date)
            if actual_total.isna().any():
                raise ValueError("missing daily total")
            total["fact"] = actual_total.to_numpy()
            total["fact_cumulative"] = total.fact.cumsum()
            total["efficiency"] = total.fact_cumulative / total.plan_cumulative.replace(0, np.nan)
            f["training_month"] = previous
            total["training_month"] = previous
            frames.append(f)
            totals.append(total)
        except ValueError as exc:
            skipped.append({"month": previous, "reason": str(exc)})
    if not frames:
        reasons = "; ".join(f"{row['month']}: {row['reason']}" for row in skipped)
        raise ValueError(f"No complete training months before {month}. Reasons by month: {reasons or 'no earlier observations'}")
    return pd.concat(frames, ignore_index=True), pd.concat(totals, ignore_index=True), skipped


def profile_forecast(history, features, config, calendar=False, builder=None):
    dims = list(config.segments)
    h = raw_asof(history, config, origin(features.date.iloc[0].strftime("%Y-%m")))
    h = h[h.date >= features.date.min() - pd.DateOffset(months=config.reference_months)].copy()
    if h.empty:
        raise ValueError("Reference profile has no historical data")
    if not calendar:
        return normalized_reference(h, features, config)
    h["weekday"] = h.date.dt.day_of_week
    h["day"] = h.date.dt.day
    h["month_start_flag"] = h.day.le(5).astype(int)
    # Holiday information is supplied from the same production calendar used by ML.
    if calendar:
        cal = pd.concat([builder.calendar(m) for m in sorted(h.month.unique())])
        h = h.merge(cal[["date", "is_workday", "is_holiday"]], on="date", validate="many_to_one")
        keys = [*dims, "weekday", "month_start_flag", "is_workday", "is_holiday"]
    else:
        keys = [*dims, "day"]
    # A negative change in a cumulative PLAN is a revision, not a request for
    # negative collections on that day. Learn calendar-shaped daily collection
    # fractions of the final monthly plan instead of fact / signed plan change.
    monthly_plan = h.groupby(["month", *dims]).target.transform("sum")
    h["daily_fraction"] = h.fact / monthly_plan.where(monthly_plan.gt(0))
    grouped = h.groupby(keys, as_index=False).agg(ratio=("daily_fraction", "mean"))
    segment = h.groupby(dims, as_index=False).agg(fallback_ratio=("daily_fraction", "mean"))
    global_ratio = h.daily_fraction.mean()
    if not np.isfinite(global_ratio):
        global_ratio = 0.0
    f = features.merge(grouped[[*keys, "ratio"]], on=keys, how="left", validate="many_to_one")
    f = f.merge(segment[[*dims, "fallback_ratio"]], on=dims, how="left", validate="many_to_one")
    # Scale by the known monthly endpoint; signed plan adjustments remain only
    # in the official KPI denominator and other models' calendar features.
    ratio = f.ratio.fillna(f.fallback_ratio).fillna(global_ratio)
    predicted_daily = f.plan_month * ratio
    return predicted_daily.groupby([f[c] for c in dims]).cumsum().to_numpy()


def normalized_reference(history, features, config):
    """Interpolate cumulative execution on month progress, preserving the last value."""
    dims = list(config.segments)
    h = daily_actual(history, config)
    h["month"] = h.date.dt.strftime("%Y-%m")
    def curves(part, n):
        result = []
        for month, group in part.groupby("month", sort=True):
            group = group.groupby("date", as_index=False)[["fact", "raw_plan"]].sum().sort_values("date")
            if not pd.DatetimeIndex(group.date).equals(days(month)) or group.raw_plan.sum() <= 0:
                continue
            cumulative = np.r_[0.0, group.fact.cumsum().to_numpy() / group.raw_plan.sum()]
            result.append(np.interp(np.arange(1, n + 1) / n, np.linspace(0, 1, len(cumulative)), cumulative))
        return np.mean(result, axis=0) if result else None
    n = features.date.nunique()
    fallback = curves(h, n)
    if fallback is None:
        raise ValueError("No complete reference month; missing facts were not replaced with zeros")
    prediction = pd.Series(index=features.index, dtype=float)
    for key, group in features.groupby(dims, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        part = h
        for col, value in zip(dims, key):
            part = part[part[col].eq(value)]
        shape = curves(part, n)
        group = group.sort_values("date")
        prediction.loc[group.index] = (fallback if shape is None else shape) * group.plan_month.iloc[0]
    return prediction.to_numpy()


def calendar_design(frame):
    """A small regularized calendar regression, also valid for signed adjustments."""
    columns = [np.ones(len(frame))]
    for field, values in [("day", range(1, 32)), ("weekday", range(7)), ("month_number", range(1, 13))]:
        columns.extend(frame[field].eq(v).to_numpy(dtype=float) for v in values)
    for field in ["is_workday", "is_holiday", "month_start_flag", "month_end_flag",
                  "pre_holiday", "post_holiday", "plan_daily_share", "total_history_eff_3m"]:
        columns.append(frame[field].to_numpy(dtype=float))
    return np.column_stack(columns)


def calendar_regression(train, future):
    matrix, target = calendar_design(train), train.fact.to_numpy() / train.plan_month.clip(lower=1).to_numpy()
    ridge = np.eye(matrix.shape[1]) * 10.0
    ridge[0, 0] = 1e-6
    coefficients = np.linalg.solve(matrix.T @ matrix + ridge, matrix.T @ target)
    daily = calendar_design(future) @ coefficients * future.plan_month.to_numpy()
    return np.cumsum(daily)


def _matrix(frame, columns):
    return frame[columns].copy()


def candidates(history, month, config, builder):
    x = builder.build(history, month)
    total_x = builder.total(x)
    leaves, totals, status = {}, {}, []
    leaves["reference_profile"] = profile_forecast(history, x, config)
    leaves["calendar_profile"] = profile_forecast(history, x, config, calendar=True, builder=builder)
    if builder.inputs.reference is not None:
        ext = builder.inputs.vintage(builder.inputs.reference, month, ["forecast_daily"], ["date"])
        joined = x[["date", *config.segments]].merge(ext, on=["date", *config.segments], how="left", validate="one_to_one")
        if joined.forecast_daily.isna().any():
            status.append({"model": "external_reference", "status": "unavailable", "reason": "incomplete pre-origin reference"})
        else:
            leaves["external_reference"] = joined.groupby(list(config.segments)).forecast_daily.cumsum().to_numpy()
    train, train_total, skipped = training_data(history, month, config, builder)
    status.extend({"model": "training_month", "status": "excluded", **item} for item in skipped)
    if train.training_month.nunique() < config.min_train_months:
        raise ValueError(f"Only {train.training_month.nunique()} complete training months before {month}; need {config.min_train_months}")
    totals["total_calendar_regression"] = calendar_regression(train_total, total_x)
    recent = train_total[train_total.training_month.isin(sorted(train_total.training_month.unique())[-config.reference_months:])]
    by_day = recent.groupby("day").efficiency.median()
    fallback = recent.efficiency.median()
    if not np.isfinite(fallback):
        fallback = 1.0
    totals["total_reference"] = total_x.day.map(by_day).fillna(fallback).to_numpy() * total_x.plan_cumulative.to_numpy()
    if config.enable_catboost:
        try:
            from catboost import CatBoostRegressor
        except ImportError:
            status.append({"model": "catboost", "status": "unavailable", "reason": "dependency not installed"})
        else:
            cols = [c for c in x if c != "date"]
            total_cols = [c for c in total_x if c != "date"]
            for depth in config.catboost_depths:
                params = dict(iterations=config.catboost_iterations, depth=depth, learning_rate=0.05,
                              l2_leaf_reg=10, loss_function="RMSE", random_seed=config.seed,
                              thread_count=1, verbose=False, allow_writing_files=False)
                for mode, target in [("daily", "fact"), ("cumulative", "fact_cumulative")]:
                    name = f"catboost_{mode}_d{depth}"
                    model = CatBoostRegressor(**params)
                    # Equalize monthly monetary scales without deriving any target-month features.
                    scale = train.plan_month.clip(lower=1.0)
                    model.fit(_matrix(train, cols), train[target] / scale, cat_features=list(config.segments))
                    pred = model.predict(_matrix(x, cols)) * x.plan_month.clip(lower=1.0).to_numpy()
                    if mode == "daily":
                        pred = pd.Series(pred).groupby([x[c] for c in config.segments]).cumsum().to_numpy()
                    leaves[name] = pred
                valid = train_total.efficiency.notna()
                model = CatBoostRegressor(**params)
                model.fit(_matrix(train_total.loc[valid], total_cols), train_total.loc[valid, "efficiency"])
                totals[f"total_catboost_d{depth}"] = model.predict(_matrix(total_x, total_cols)) * total_x.plan_cumulative.to_numpy()
    if config.enable_sarimax:
        try:
            totals["total_sarimax"] = sarimax_forecast(history, total_x, month, config, builder)
        except (ImportError, ValueError, np.linalg.LinAlgError) as exc:
            status.append({"model": "total_sarimax", "status": "unavailable", "reason": str(exc)})
    for name, pred in [*leaves.items(), *totals.items()]:
        if not np.isfinite(pred).all():
            raise ValueError(f"Nonfinite predictions: {name}")
        status.append({"model": name, "status": "available", "month": str(month)})
    return x, leaves, totals, status


def sarimax_forecast(history, total_x, month, config, builder):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    h = raw_asof(history, config, origin(month))
    dates = pd.date_range(h.date.min(), month_start(month) - pd.Timedelta(days=1), freq="D")
    a = h.groupby("date")[["fact", "target"]].sum().reindex(dates)
    if a.isna().any().any():
        raise ValueError("SARIMAX needs complete daily history; missing days were not imputed")
    calendars = pd.concat([builder.calendar(m) for m in pd.period_range(dates.min(), dates.max(), freq="M")])
    c = calendars.set_index("date").reindex(dates)
    columns = ["month_start_flag", "month_end_flag", "is_workday", "is_holiday", "annual_sin", "annual_cos"]
    past = c[columns].astype(float).copy()
    future = total_x[columns].astype(float).copy()
    for wd in range(1, 7):
        past[f"weekday_{wd}"] = (dates.day_of_week == wd).astype(float)
        future[f"weekday_{wd}"] = total_x.weekday.eq(wd).astype(float)
    plan_scale = max(float(a.target.abs().mean()), 1.0)
    fact_scale = max(float(a.fact.abs().mean()), 1.0)
    past["plan"] = a.target.to_numpy() / plan_scale
    future["plan"] = total_x.plan.to_numpy() / plan_scale
    # Drop constant regressors using training data only (a SARIMAX intercept is included).
    usable = past.columns[past.nunique() > 1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = SARIMAX(a.fact.to_numpy() / fact_scale, exog=past[usable].to_numpy(),
                         order=(1, 0, 1), seasonal_order=(1, 0, 0, 7), trend="c",
                         enforce_stationarity=False, enforce_invertibility=False).fit(disp=False, maxiter=150)
    if not fitted.mle_retvals.get("converged", False):
        raise ValueError("SARIMAX optimizer did not converge")
    return np.cumsum(np.asarray(fitted.get_forecast(len(total_x), exog=future[usable].to_numpy()).predicted_mean) * fact_scale)


def specs(leaves, totals, weights):
    for leaf in sorted(leaves):
        yield {"leaf": leaf, "total": None, "weight": 0.0}
        for total in sorted(totals):
            for weight in weights:
                if weight > 0:
                    yield {"leaf": leaf, "total": total, "weight": float(weight)}


def spec_id(spec):
    return f"{spec['leaf']}|{spec['total'] or 'bottom_up'}|{spec['weight']:g}"


def reconcile(features, leaf_cumulative, total_cumulative, weight, config):
    dims = list(config.segments)
    f = features[["date", *dims, "plan", "plan_cumulative", "plan_month", "debt"]].copy()
    f["base_forecast_cumulative"] = np.asarray(leaf_cumulative, dtype=float)
    bottom = f.groupby("date").base_forecast_cumulative.sum()
    direct = bottom.to_numpy() if total_cumulative is None else np.asarray(total_cumulative)
    chosen = (1 - weight) * bottom.to_numpy() + weight * direct
    target = pd.Series(chosen, index=bottom.index)
    daily_den = f.groupby("date").plan_cumulative.transform("sum")
    monthly_den = f.groupby("date").plan_month.transform("sum")
    debt = f.debt.clip(lower=0)
    debt_den = debt.groupby(f.date).transform("sum")
    count = f.groupby("date").date.transform("size")
    alpha = np.where(daily_den > 0, f.plan_cumulative / daily_den.replace(0, np.nan),
                     np.where(monthly_den > 0, f.plan_month / monthly_den.replace(0, np.nan),
                              np.where(debt_den > 0, debt / debt_den.replace(0, np.nan), 1 / count)))
    delta = f.date.map(target - bottom).to_numpy()
    f["reconciliation_cumulative"] = alpha * delta
    f["forecast_cumulative"] = f.base_forecast_cumulative + f.reconciliation_cumulative
    f = f.sort_values([*dims, "date"]).reset_index(drop=True)
    for cumulative, daily in [("forecast_cumulative", "forecast_daily"),
                              ("reconciliation_cumulative", "reconciliation_daily")]:
        f[daily] = f.groupby(dims)[cumulative].diff().fillna(f[cumulative])
    summed = f.groupby("date").forecast_cumulative.sum().reindex(target.index)
    if not np.allclose(summed, target, rtol=1e-10, atol=1e-7):
        raise AssertionError("Reconciliation does not preserve the total")
    f["origin"] = origin(f.date.iloc[0].strftime("%Y-%m"))
    return f
