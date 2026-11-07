import numpy as np
import pandas as pd
from .config import month_start, origin, days
from .data import raw_asof, numeric


class FeatureBuilder:
    def __init__(self, config, inputs):
        self.config, self.inputs = config, inputs
        self._calendar_cache = {}
        self._feature_cache = {}

    def calendar(self, month):
        month = str(month)
        if month in self._calendar_cache:
            return self._calendar_cache[month].copy()
        dates = days(month)
        start, end = dates[0], dates[-1]
        extended = pd.date_range(start - pd.Timedelta(days=40), end + pd.Timedelta(days=40))
        try:
            import holidays
            holiday_set = holidays.country_holidays(self.config.country, years=sorted(set(extended.year)))
            holiday = pd.Series([int(d.date() in holiday_set) for d in extended], index=extended)
        except ImportError:
            raise ImportError("Install holidays from requirements.txt to construct calendar features")
        workday = pd.Series([int(holiday_set.is_working_day(d.date())) for d in extended], index=extended)
        overrides = self.inputs.calendar
        if overrides is not None:
            indexed = overrides.set_index("date")
            if self.config.require_calendar_overrides and not dates.isin(indexed.index).all():
                raise ValueError(f"Production calendar is incomplete for {month}")
            common = extended.intersection(indexed.index)
            workday.loc[common] = indexed.loc[common, "is_workday"].astype(int)
            if "is_holiday" in indexed:
                holiday.loc[common] = indexed.loc[common, "is_holiday"].astype(int)
        f = pd.DataFrame({"date": dates})
        f["day"] = dates.day
        f["weekday"] = dates.day_of_week
        f["month_number"] = start.month
        f["year_index"] = start.year - 2024
        f["days_in_month"] = len(dates)
        f["days_left"] = len(dates) - f.day
        f["week_of_month"] = (f.day - 1) // 7
        f["month_progress"] = f.day / len(dates)
        f["is_weekend"] = (f.weekday >= 5).astype(int)
        f["is_workday"] = workday.loc[dates].to_numpy()
        f["is_holiday"] = holiday.loc[dates].to_numpy()
        f["pre_holiday"] = holiday.loc[dates + pd.Timedelta(days=1)].to_numpy()
        f["post_holiday"] = holiday.loc[dates - pd.Timedelta(days=1)].to_numpy()
        before, after = {}, {}
        streak = 0
        for date, is_working in workday.items():
            before[date] = streak
            streak = 0 if is_working else streak + 1
        streak = 0
        for date, is_working in workday.iloc[::-1].items():
            after[date] = streak
            streak = 0 if is_working else streak + 1
        f["nonwork_before"] = [before[d] for d in dates]
        f["nonwork_after"] = [after[d] for d in dates]
        f["workdays_left"] = f.is_workday.iloc[::-1].cumsum().iloc[::-1].to_numpy()
        f["workday_number"] = f.is_workday.cumsum()
        f["first_workday"] = (f.is_workday.eq(1) & f.workday_number.eq(1)).astype(int)
        f["last_workday"] = (f.is_workday.eq(1) & f.workdays_left.eq(1)).astype(int)
        f["month_start_flag"] = (f.day <= 5).astype(int)
        f["month_end_flag"] = (f.days_left < 5).astype(int)
        f["year_start_flag"] = int(start.month == 1)
        f["year_end_flag"] = int(start.month == 12)
        f["annual_sin"] = np.sin(2 * np.pi * dates.day_of_year / 365.25)
        f["annual_cos"] = np.cos(2 * np.pi * dates.day_of_year / 365.25)
        self._calendar_cache[month] = f
        return f

    def portfolio(self, history, month):
        dims = list(self.config.segments)
        external = self.inputs.portfolio
        if external is not None:
            f = external[(external.month == str(month)) & (external.known_at <= origin(month))
                         & (external.snapshot_date <= origin(month))].copy()
            if f.empty:
                if history.empty:
                    return pd.DataFrame(columns=[*dims, "debt", "acc_count", "avg_dpd", "avg_pvr", "snapshot_age_days"])
                raise ValueError(f"No portfolio snapshot available before {month}")
            if f.duplicated([*dims, "known_at"]).any():
                raise ValueError("Duplicate portfolio vintage")
            f = f.sort_values(["snapshot_date", "known_at"]).drop_duplicates(dims, keep="last")
            for c in ["avg_dpd", "avg_pvr"]:
                if c not in f:
                    f[c] = np.nan
                f[c] = pd.to_numeric(f[c], errors="raise")
            f["snapshot_age_days"] = (month_start(month) - f.snapshot_date).dt.days
            return f[[*dims, "debt", "acc_count", "avg_dpd", "avg_pvr", "snapshot_age_days"]]
        h = raw_asof(history, self.config, origin(month))
        if h.empty or not {"w1", "acc_count", "avg_dpd", "avg_pvr"} <= set(h):
            # The first historical training month may have no earlier snapshot.
            # Mark missing initial state; never substitute that month's observed portfolio.
            return pd.DataFrame(columns=[*dims, "debt", "acc_count", "avg_dpd", "avg_pvr", "snapshot_age_days"])
        latest = h.date.max()
        h = h[h.date == latest].copy()
        h["dpd_weighted"] = h.avg_dpd * h.w1
        h["pvr_weighted"] = h.avg_pvr * h.w1
        h["dpd_debt"] = h.w1.where(h.avg_dpd.notna(), 0)
        h["pvr_debt"] = h.w1.where(h.avg_pvr.notna(), 0)
        f = h.groupby(dims, as_index=False).agg(
            debt=("w1", "sum"), acc_count=("acc_count", "sum"),
            dpd_weighted=("dpd_weighted", "sum"), pvr_weighted=("pvr_weighted", "sum"),
            dpd_debt=("dpd_debt", "sum"), pvr_debt=("pvr_debt", "sum"))
        f["avg_dpd"] = f.dpd_weighted / f.dpd_debt.replace(0, np.nan)
        f["avg_pvr"] = f.pvr_weighted / f.pvr_debt.replace(0, np.nan)
        f["snapshot_age_days"] = (month_start(month) - latest).days
        return f[[*dims, "debt", "acc_count", "avg_dpd", "avg_pvr", "snapshot_age_days"]]

    def build(self, history, month):
        """All observed features are frozen before month, including in training."""
        cache_key = (id(history), str(month))
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key].copy()
        dims = list(self.config.segments)
        h = raw_asof(history, self.config, origin(month))
        f = self.inputs.plan(month)
        f = f.merge(self.calendar(month), on="date", validate="many_to_one")
        f = f.sort_values([*dims, "date"]).reset_index(drop=True)
        f["plan_cumulative"] = f.groupby(dims).plan.cumsum()
        f["plan_month"] = f.groupby(dims).plan.transform("sum")
        f["plan_daily_share"] = f.plan / f.plan_month.replace(0, np.nan)
        f["plan_cumulative_share"] = f.plan_cumulative / f.plan_month.replace(0, np.nan)
        portfolio = self.portfolio(h, month)
        absent = portfolio[dims].merge(f[dims].drop_duplicates(), how="left", indicator=True, on=dims)
        if absent._merge.eq("left_only").any():
            raise ValueError(f"{month}: portfolio segments missing from plan; supply explicit zero-plan rows")
        f = f.merge(portfolio, how="left", on=dims, validate="many_to_one")
        f["new_segment"] = f.debt.isna().astype(int)
        f["debt"] = f.debt.fillna(0)
        f["acc_count"] = f.acc_count.fillna(0)
        f["average_debt"] = f.debt / f.acc_count.replace(0, np.nan)
        f["portfolio_total_debt"] = portfolio.debt.sum()
        f["portfolio_debt_share"] = f.debt / max(float(portfolio.debt.sum()), 1.0)
        for n in (1, 3, 6):
            cutoff = month_start(month) - pd.DateOffset(months=n)
            part = h[h.date >= cutoff]
            stats = part.groupby(dims, as_index=False).agg(
                past_fact=("fact", "sum"), past_plan=("target", "sum"))
            stats[f"history_eff_{n}m"] = stats.past_fact / stats.past_plan.replace(0, np.nan)
            stats = stats.rename(columns={"past_fact": f"history_fact_{n}m", "past_plan": f"history_plan_{n}m"})
            f = f.merge(stats, on=dims, how="left", validate="many_to_one")
            denominator = part.target.sum()
            f[f"total_history_eff_{n}m"] = part.fact.sum() / denominator if denominator else np.nan
        for n in (7, 14, 28):
            part = h[h.date >= month_start(month) - pd.Timedelta(days=n)]
            stats = part.groupby(dims, as_index=False).agg(recent_fact=("fact", "sum"),
                                                         observed_days=("date", "nunique"))
            stats["recent_fact"] = stats.recent_fact.where(stats.observed_days.eq(n))
            stats = stats.rename(columns={"recent_fact": f"recent_fact_{n}d", "observed_days": f"recent_days_{n}d"})
            f = f.merge(stats, on=dims, how="left", validate="many_to_one")
        inflow = self.inputs.vintage(self.inputs.inflow, month, ["incoming_debt"], ["date"])
        f["incoming_debt"] = 0.0
        if inflow is not None:
            unknown = inflow[dims].drop_duplicates().merge(f[dims].drop_duplicates(), on=dims, how="left", indicator=True)
            if unknown._merge.eq("left_only").any():
                raise ValueError("Planned inflow has segments missing from plan")
            f = f.drop(columns="incoming_debt").merge(inflow, on=[*dims, "date"], how="left", validate="one_to_one")
            # Inflow input is a sparse schedule; absent entries mean no scheduled inflow.
            f["incoming_debt"] = f.incoming_debt.fillna(0)
        f["incoming_debt_cumulative"] = f.groupby(dims).incoming_debt.cumsum()
        if self.inputs.operations is not None:
            ops = self.inputs.operations
            ops = ops[(ops.date < month_start(month)) & (ops.known_at <= origin(month))]
            key = [*dims, "date"]
            if ops.duplicated([*key, "known_at"]).any():
                raise ValueError("Duplicate operations vintage")
            ops = ops.sort_values("known_at").drop_duplicates(key, keep="last")
            for n in (7, 30):
                part = ops[ops.date >= month_start(month) - pd.Timedelta(days=n)]
                cols = [c for c in self.inputs.operations if c.startswith("op_")]
                summed = part.groupby(dims, as_index=False)[cols].sum()
                summed = summed.rename(columns={c: f"{c}_{n}d" for c in cols})
                f = f.merge(summed, on=dims, how="left", validate="many_to_one")
        for col in f:
            if col not in [*dims, "date"]:
                f[col] = pd.to_numeric(f[col], errors="raise").replace([np.inf, -np.inf], np.nan).fillna(-1.0)
        self._feature_cache[cache_key] = f
        return f.copy()

    def total(self, features):
        dims = list(self.config.segments)
        calendar = self.calendar(features.date.iloc[0].strftime("%Y-%m"))
        sums = ["plan", "plan_cumulative", "plan_month", "debt", "acc_count", "incoming_debt", "incoming_debt_cumulative"]
        t = features.groupby("date", as_index=False)[sums].sum().merge(calendar, on="date", validate="one_to_one")
        t["plan_daily_share"] = t.plan / t.plan_month.replace(0, np.nan)
        t["plan_cumulative_share"] = t.plan_cumulative / t.plan_month.replace(0, np.nan)
        for col in features:
            if col.startswith("total_history_"):
                t[col] = features.groupby("date")[col].first().reindex(t.date).to_numpy()
        for col in ["avg_dpd", "avg_pvr"]:
            valid = features[col].ge(0)
            num = (features[col] * features.debt.where(valid, 0)).groupby(features.date).sum()
            den = features.debt.where(valid, 0).groupby(features.date).sum()
            t[col] = (num / den.replace(0, np.nan)).reindex(t.date).to_numpy()
        for col in features:
            if col.startswith("op_"):
                t[col] = features.groupby("date")[col].sum().reindex(t.date).to_numpy()
        return t.replace([np.inf, -np.inf], np.nan).fillna(-1.0)
