import hashlib
import importlib.metadata
import json
from dataclasses import replace
from pathlib import Path
import numpy as np
import pandas as pd
from .config import Config, month_start, origin, days
from .data import Inputs, load_raw, prepare_actual, raw_asof, daily_actual, numeric, split_dated_frame
from .features import FeatureBuilder
from .models import candidates, reconcile, specs, spec_id
from .export import upload_frame, database_upload_frame, aggregate_database_upload


def evaluate(forecast, actual, config):
    """Evaluation is the only API consuming target-month actuals."""
    keys = ["date", *config.segments]
    if forecast.empty or forecast.duplicated(keys).any():
        raise ValueError("Forecast must have a unique, nonempty segment-day grid")
    forecast = forecast.copy()
    numeric(forecast, ["forecast_daily"])
    forecast["plan_missing"] = forecast.plan.isna()
    if forecast.date.dt.to_period("M").nunique() != 1:
        raise ValueError("Evaluate one forecast month at a time")
    expected = days(forecast.date.iloc[0].strftime("%Y-%m"))
    for _, group in forecast.groupby(list(config.segments)):
        if not pd.DatetimeIndex(group.date.sort_values()).equals(expected):
            raise ValueError("Forecast grid must cover every date of the month for every segment")
    if not actual.date.isin(expected).all():
        raise ValueError("Actual dates fall outside the forecast month")
    if actual.duplicated(keys).any():
        raise ValueError("Actuals must be aggregated to the forecast grain")
    f = forecast.merge(actual[[*keys, "fact"]], on=keys, how="outer", indicator=True, validate="one_to_one")
    f["missing_actual"] = f._merge.eq("left_only") | f.fact.isna()
    f["unplanned_actual"] = f._merge.eq("right_only")
    # Extra actual segments must contribute to the total, never disappear in an inner join.
    for c in ["plan", "forecast_daily", "reconciliation_daily"]:
        f.loc[f.unplanned_actual, c] = 0.0
    f = f.sort_values([*config.segments, "date"]).reset_index(drop=True)
    f["forecast_cumulative"] = f.groupby(list(config.segments)).forecast_daily.cumsum()
    f["plan_cumulative"] = f.groupby(list(config.segments)).plan.transform(lambda s: s.cumsum(skipna=False))
    f["fact_cumulative"] = f.groupby(list(config.segments)).fact.transform(lambda s: s.cumsum(skipna=False))
    f["signed_error_daily"] = f.forecast_daily - f.fact
    f["signed_error_cumulative"] = f.forecast_cumulative.fillna(0) - f.fact_cumulative
    total = f.groupby("date", as_index=False).agg(
        plan_daily=("plan", lambda s: s.sum(skipna=False)), forecast_daily=("forecast_daily", "sum"),
        missing_plan_cells=("plan_missing", "sum"),
        observed_fact_daily=("fact", lambda s: s.sum(min_count=1)),
        missing_actual_cells=("missing_actual", "sum"), unplanned_actual_cells=("unplanned_actual", "sum"))
    total["fact_daily"] = total.observed_fact_daily.where(total.missing_actual_cells.eq(0))
    for c in ["plan", "forecast", "fact"]:
        total[f"{c}_cumulative"] = total[f"{c}_daily"].cumsum(skipna=False)
    denominator = total.plan_cumulative.where(total.plan_cumulative.gt(0))
    total["actual_efficiency_pct"] = 100 * total.fact_cumulative / denominator
    total["forecast_efficiency_pct"] = 100 * total.forecast_cumulative / denominator
    total["signed_error_pp"] = total.forecast_efficiency_pct - total.actual_efficiency_pct
    total["absolute_error_pp"] = total.signed_error_pp.abs()
    total["within_tolerance"] = total.absolute_error_pp.le(config.tolerance_pp) & total.absolute_error_pp.notna()
    total["signed_error_money"] = total.forecast_cumulative - total.fact_cumulative
    total["month"] = total.date.dt.strftime("%Y-%m")
    return f.drop(columns="_merge"), total


def metrics(total, config):
    eligible = total.plan_cumulative.gt(0)
    valid = eligible & total.absolute_error_pp.notna()
    full_grid = bool(len(total) and pd.DatetimeIndex(total.date).equals(days(total.date.iloc[0].strftime("%Y-%m"))))
    missing_plan = int(total["missing_plan_cells"].sum()) if "missing_plan_cells" in total else 0
    complete = bool(full_grid and not missing_plan and total.missing_actual_cells.sum() == 0 and total.unplanned_actual_cells.sum() == 0
                    and valid.sum() == eligible.sum() and eligible.any())
    errors = total.loc[valid, "absolute_error_pp"]
    return {
        "max_error_pp": float(errors.max()) if len(errors) else None,
        "mae_pp": float(errors.mean()) if len(errors) else None,
        "p90_error_pp": float(errors.quantile(0.9)) if len(errors) else None,
        "bias_pp": float(total.loc[valid, "signed_error_pp"].mean()) if len(errors) else None,
        "end_error_pp": float(total.absolute_error_pp.iloc[-1]) if len(total) and pd.notna(total.absolute_error_pp.iloc[-1]) else None,
        "hit_rate": float(total.loc[eligible, "within_tolerance"].mean()) if eligible.any() else None,
        "eligible_days": int(eligible.sum()), "scored_days": int(valid.sum()),
        "missing_actual_cells": int(total.missing_actual_cells.sum()),
        "unplanned_actual_cells": int(total.unplanned_actual_cells.sum()),
        "complete": complete, "full_month_grid": full_grid, "missing_plan_cells": missing_plan,
        "goal_achieved": bool(complete and errors.le(config.tolerance_pp).all()),
    }


def json_write(path, data):
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str, allow_nan=False), encoding="utf-8")


class ForecastRunner:
    def __init__(self, config, *, history_df=None, plan_df=None):
        config = config.resolved()
        config.validate()
        if (history_df is None) != (plan_df is None):
            raise ValueError("Supply both history_df and plan_df when using DataFrame inputs")
        self.config = config
        self._from_frames = history_df is not None
        self.inputs = Inputs(config, plan_df=plan_df)
        self.builder = FeatureBuilder(config, self.inputs)
        if self._from_frames:
            self.history = prepare_actual(history_df, config, before=config.forecast_month)
        else:
            self.history = load_raw(config, before=config.forecast_month)
        self.selection = None
        self.status = []
        self.fold_log = []
        self.development_curves = pd.DataFrame()
        self.control_curves = pd.DataFrame()
        self.report_verified = False

    def audit(self):
        c = self.config
        h = raw_asof(self.history, c, origin(c.forecast_month))
        daily = daily_actual(h, c)
        report = {
            "history_start": str(h.date.min().date()), "history_end": str(h.date.max().date()),
            "raw_rows": len(h), "negative_daily_fact_rows": int(h.fact.lt(0).sum()),
            "raw_plan_negative_rows": int(h.target.lt(0).sum()),
            "raw_vintage_mode": "timestamped" if c.raw_available_column else "retrospective_reconstructed",
            "input_mode": "dataframes" if self._from_frames else "sqlite",
            "plan_source": "df_plan" if self._from_frames else f"{c.plan_table or c.raw_table}.{c.plan_value_column}",
            "plan_is_forecast": None if self._from_frames else c.plan_is_forecast,
            "plan_vintage_mode": "timestamped" if c.plan_available_column else "snapshot_no_plan_vintages",
            "calendar_mode": "approved_overrides" if self.inputs.calendar is not None else "holidays_library_with_workday_transfers",
            "fact_semantics": c.fact_mode,
            "forecast_grain": c.forecast_grain,
            "plan_semantics": c.plan_mode, "semantics_verified_against_report": self.report_verified,
            "available_target_plan_rows": len(self.inputs.plan(c.forecast_month)),
            "external_reference": bool(c.reference_path),
            "reference_note": "external current-method forecast" if c.reference_path else "normalized historical cumulative shape with endpoint preservation",
        }
        rows = []
        for m in sorted(h.month.unique()):
            part = daily[daily.date.dt.strftime("%Y-%m") == m]
            expected = days(m)
            rows.append({"month": m, "observed_days": part.date.nunique(), "expected_days": len(expected),
                         "fact_sum": float(part.fact.sum()), "raw_plan_sum": float(part.raw_plan.sum())})
        # Build target features now so missing frozen portfolios/plan keys fail before a long backtest.
        if not self.needs_money_fallback():
            self.builder.build(self.history, c.forecast_month)
        report["missing_future_plan_fallback"] = self.needs_money_fallback()
        return report, pd.DataFrame(rows), daily

    def needs_money_fallback(self):
        return self.target_plan_status()["mode"] == "money_fallback"

    def target_plan_status(self):
        plan = self.inputs.plan(self.config.forecast_month)
        reasons = []
        if plan.empty:
            reasons.append("Нет строк плана целевого месяца из выбранного источника.")
        missing_cells = int(plan.plan.isna().sum())
        if missing_cells:
            reasons.append(f"Неизвестен дневной план в {missing_cells} ячейках дата × сегмент.")
        if not plan.empty and not missing_cells and plan.plan.sum() <= 0:
            reasons.append("Суммарный план равен нулю; эффективность относительно плана не определена.")
        h = raw_asof(self.history, self.config, origin(self.config.forecast_month))
        recent = h[h.date >= month_start(self.config.forecast_month) - pd.DateOffset(months=self.config.reference_months)]
        dims = list(self.config.segments)
        missing = recent[dims].drop_duplicates().merge(plan[dims].drop_duplicates(), on=dims, how="left", indicator=True)
        absent = missing.loc[missing._merge.eq("left_only"), dims]
        if len(absent):
            reasons.append(f"В плане нет {len(absent)} сегментов, присутствовавших в недавней истории.")
        return {"mode": "money_fallback" if reasons else "selected_model", "reasons": reasons,
                "forecast_grain": self.config.forecast_grain,
                "missing_plan_cells": missing_cells, "missing_segments": len(absent),
                "missing_plan_dates": plan.loc[plan.plan.isna(), "date"].drop_duplicates().dt.strftime("%Y-%m-%d").tolist(),
                "missing_segment_examples": absent.head(10).to_dict("records"),
                "plan_is_forecast": self.config.plan_is_forecast,
                "validation": "not_run" if reasons else "model_selection_and_control"}

    def _predict_candidates(self, month):
        result = candidates(self.history, month, self.config, self.builder)
        self.status.extend({**entry, "origin_month": str(month)} for entry in result[3])
        return result[:3]

    def verify_report(self, checks_path, tolerance_money=0.01):
        """Reconcile daily sums against exported, independently calculated report totals."""
        checks = pd.read_csv(checks_path)
        if not {"date", "fact_daily", "plan_daily"} <= set(checks):
            raise ValueError("Report checks need date,fact_daily,plan_daily")
        if checks.empty:
            raise ValueError("Reference report checks are empty")
        numeric(checks, ["fact_daily", "plan_daily"])
        checks["date"] = pd.to_datetime(checks.date).dt.normalize()
        if checks.date.duplicated().any() or (checks.date >= month_start(self.config.forecast_month)).any():
            raise ValueError("Report check dates must be unique and before the target month")
        h = raw_asof(self.history, self.config, origin(self.config.forecast_month))
        computed = h.groupby("date", as_index=False).agg(computed_fact=("fact", "sum"), computed_plan=("target", "sum"))
        merged = checks.merge(computed, on="date", how="left", validate="one_to_one")
        merged["fact_difference"] = merged.computed_fact - merged.fact_daily
        merged["plan_difference"] = merged.computed_plan - merged.plan_daily
        if merged[["fact_difference", "plan_difference"]].isna().any().any() or not (
                merged[["fact_difference", "plan_difference"]].abs() <= tolerance_money).all().all():
            raise ValueError("Daily amounts disagree with the reference report; inspect fact/plan semantics before fitting")
        self.report_verified = True
        return merged

    def _truth(self, month):
        # Evaluation truth may be finalized/revised after a prediction origin.
        # It is never passed back into that origin's features or fitted model.
        h = raw_asof(self.history, self.config, pd.Timestamp.max - pd.Timedelta(days=1))
        return daily_actual(h[h.month == str(month)], self.config)

    def _apply(self, x, leaves, totals, spec):
        if spec["leaf"] not in leaves or (spec["total"] is not None and spec["total"] not in totals):
            raise ValueError(f"Selected candidate is unavailable: {spec_id(spec)}")
        f = reconcile(x, leaves[spec["leaf"]], totals.get(spec["total"]), spec["weight"], self.config)
        f["model"] = spec_id(spec)
        return f

    def select(self):
        c = self.config
        rows, curves, specifications = [], [], {}
        expected_months = [str(m) for m in pd.period_range(c.development_start, c.development_end, freq="M")]
        for month in expected_months:
            try:
                x, leaves, totals = self._predict_candidates(month)
                truth = self._truth(month)
                month_complete = True
                for spec in specs(leaves, totals, c.blend_weights):
                    key = spec_id(spec)
                    pred = self._apply(x, leaves, totals, spec)
                    _, total = evaluate(pred, truth, c)
                    score = metrics(total, c)
                    if not score["complete"]:
                        month_complete = False
                        break
                    specifications[key] = spec
                    rows.append({"candidate": key, "month": month, **score})
                    total["candidate"] = key
                    curves.append(total)
                if not month_complete:
                    rows = [r for r in rows if r["month"] != month]
                    curves = [r for r in curves if r.month.iloc[0] != month]
                    raise ValueError("incomplete or unplanned facts on evaluation grid")
                self.fold_log.append({"month": month, "phase": "development", "status": "ok"})
                print(f"Development {month}: {len(leaves)} segment models, {len(totals)} total models")
            except ValueError as exc:
                self.fold_log.append({"month": month, "phase": "development", "status": "failed", "reason": str(exc)})
                if c.require_all_folds:
                    raise ValueError(f"Development fold {month} failed: {exc}") from exc
                print(f"Development {month}: excluded: {exc}")
        scores = pd.DataFrame(rows)
        if scores.empty:
            raise ValueError("No complete development folds; no model has been selected")
        complete_months = scores.month.nunique()
        self.development_curves = pd.concat(curves, ignore_index=True)
        ranking = self.development_curves.groupby("candidate", as_index=False).agg(
            max_error_pp=("absolute_error_pp", "max"), mae_pp=("absolute_error_pp", "mean"),
            months=("month", "nunique"), hit_rate=("within_tolerance", "mean"))
        # Candidates must compete on exactly the same available months.
        ranking = ranking[ranking.months.eq(complete_months)].sort_values(
            ["max_error_pp", "mae_pp", "candidate"], kind="stable").reset_index(drop=True)
        if ranking.empty:
            raise ValueError("No candidate was available on every development fold")
        chosen = ranking.candidate.iloc[0]
        self.selection = specifications[chosen]
        self.development_scores, self.ranking = scores, ranking
        return ranking, dict(self.selection)

    def control(self):
        if self.selection is None:
            raise ValueError("Run select() first, or load a previously frozen selection")
        c = self.config
        rows, curves = [], []
        for m in pd.period_range(c.control_start, c.control_end, freq="M"):
            month = str(m)
            x, leaves, totals = self._predict_candidates(month)
            pred = self._apply(x, leaves, totals, self.selection)
            _, total = evaluate(pred, self._truth(month), c)
            score = metrics(total, c)
            if not score["complete"] and c.require_all_folds:
                raise ValueError(f"Incomplete control fold {month}")
            rows.append({"month": month, "candidate": spec_id(self.selection), **score})
            curves.append(total)
            self.fold_log.append({"month": month, "phase": "control", "status": "ok" if score["complete"] else "incomplete"})
            print(f"Control {month}: max error {score['max_error_pp']} pp")
        self.control_curves = pd.concat(curves, ignore_index=True)
        self.control_scores = pd.DataFrame(rows)
        return self.control_scores

    def forecast(self, selection=None, save=True):
        """Returns a frozen forecast without reading a single target-month actual."""
        if selection is not None:
            self.selection = dict(selection)
        if self.needs_money_fallback():
            from .fallback import money_forecast
            pred = money_forecast(self.history, self.inputs.plan(self.config.forecast_month), self.config)
            self.status.append({"model": "money_fallback", "status": "available",
                                "reason": "Official future plan incomplete; forecast based on historical collections"})
            pred["forecast_efficiency_pct"] = 100 * pred.forecast_cumulative / pred.plan_cumulative.where(pred.plan_cumulative.gt(0))
            self.forecast_frame = pred
            if save:
                self.save_forecast(pred)
            return pred
        if self.selection is None:
            raise ValueError("Run select() or supply a frozen selection")
        x, leaves, totals = self._predict_candidates(self.config.forecast_month)
        pred = self._apply(x, leaves, totals, self.selection)
        pred["forecast_efficiency_pct"] = 100 * pred.forecast_cumulative / pred.plan_cumulative.where(pred.plan_cumulative.gt(0))
        self.forecast_frame = pred
        if save:
            self.save_forecast(pred)
        return pred

    def save_forecast(self, pred):
        c = self.config
        daily_upload = upload_frame(pred, c)
        upload = database_upload_frame(pred, c)
        out = Path(c.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        payload = pred.to_csv(index=False, lineterminator="\n").encode("utf-8-sig")
        forecast_path = out / "forecast.csv"
        if forecast_path.exists() and forecast_path.read_bytes() != payload:
            raise FileExistsError(f"A different frozen forecast already exists in {out}; choose a new output_dir")
        forecast_path.write_bytes(payload)
        upload.to_csv(out / "forecast_upload.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
        aggregate_database_upload(upload).to_csv(out / "forecast_upload_total.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
        daily_upload.to_csv(out / "forecast_daily_totals.csv", index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
        json_write(out / "forecast_run_status.json", self.target_plan_status())
        json_write(out / "forecast_upload_schema.json", {
            "file": "forecast_upload.csv", "table": "forecast", "grain": "date x model segment; cohorts already aggregated",
            "key": ["report_month", "report_dt", "dt_in", "subproduct", "dpd_bucket", "pvr_bucket", "terbank", "inflow", "forecast_date", "forecast_model"],
            "replace_slice": ["report_month", "forecast_model", "forecast_date"],
            "rows": len(upload), "forecast_month_total": float(daily_upload.forecast_daily.sum()),
            "amounts": "fact is predicted cumulative money ON report_dt; plan is cumulative ON report_dt. Never sum snapshots across dates.",
            "cohorts": "dt_in is empty: model forecasts are aggregated across cohorts. Do not attach source dt_in rows.",
            "missing_plan": "empty CSV cell means NULL, not zero",
            "load": "upsert by key or replace the entire model/month/origin slice; never append repeated runs",
            "join": "do not join forecasts to raw cohort rows; aggregate actuals to the same date/segment grain first",
        })
        total = pred.groupby("date", as_index=False).agg(
            plan_daily=("plan", lambda s: s.sum(skipna=False)), forecast_daily=("forecast_daily", "sum"))
        total["plan_cumulative"] = total.plan_daily.cumsum(skipna=False)
        total["forecast_cumulative"] = total.forecast_daily.cumsum()
        total["forecast_efficiency_pct"] = 100 * total.forecast_cumulative / total.plan_cumulative.where(total.plan_cumulative.gt(0))
        total["origin"] = origin(c.forecast_month)
        total.to_csv(out / "forecast_total.csv", index=False, encoding="utf-8-sig")
        from .reporting import save_forecast_chart
        save_forecast_chart(total, c, out / "forecast.html")
        for attr, name in [("ranking", "development_ranking"), ("development_scores", "development_scores"),
                           ("control_scores", "control_scores"), ("control_curves", "control_curves"),
                           ("development_curves", "development_curves")]:
            frame = getattr(self, attr, None)
            if frame is not None and not frame.empty:
                frame.to_csv(out / f"{name}.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(self.status).to_csv(out / "model_status.csv", index=False, encoding="utf-8-sig")
        pd.DataFrame(self.fold_log).to_csv(out / "fold_log.csv", index=False, encoding="utf-8-sig")
        history = raw_asof(self.history, c, origin(c.forecast_month))
        digest = hashlib.sha256(pd.util.hash_pandas_object(history, index=False).to_numpy().tobytes()).hexdigest()
        versions = {}
        for package in ["pandas", "numpy", "catboost", "statsmodels", "holidays", "plotly"]:
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = None
        inputs = {}
        for m in sorted(set(history.month.unique()) | {c.forecast_month}):
            try:
                frame = self.inputs.plan(m)
                inputs[m] = hashlib.sha256(pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()).hexdigest()
            except ValueError:
                continue
        audit, monthly, daily = self.audit()
        json_write(out / "audit.json", audit)
        monthly.to_csv(out / "history_months.csv", index=False, encoding="utf-8-sig")
        daily.to_csv(out / "history_daily_audit.csv", index=False, encoding="utf-8-sig")
        # Hash prediction inputs only, never the database file containing target actuals.
        json_write(out / "manifest.json", {"config": c.as_dict(), "selection": self.selection,
                   "history_sha256": digest, "plan_sha256": inputs, "versions": versions,
                   "forecast_sha256": hashlib.sha256((out / "forecast.csv").read_bytes()).hexdigest(),
                   "input_features_sha256": None if self.needs_money_fallback() else hashlib.sha256(pd.util.hash_pandas_object(
                       self.builder.build(self.history, c.forecast_month), index=False).to_numpy().tobytes()).hexdigest(),
                   "source_mode": audit["raw_vintage_mode"],
                   "input_mode": audit["input_mode"],
                   "asof_guarantee": "user-provided availability metadata; historical revisions may be unobservable without raw vintages"})

    def compare(self, forecast=None, save=True, *, actual_df=None):
        pred = self.forecast_frame if forecast is None else forecast
        if actual_df is not None:
            actual_raw = prepare_actual(actual_df, self.config, month=self.config.forecast_month)
        elif self._from_frames:
            raise ValueError("Supply actual_df to compare a DataFrame forecast with actuals")
        else:
            actual_raw = load_raw(self.config, month=self.config.forecast_month)
        # Evaluation uses the latest available actual revision, not a fictitious pre-month snapshot.
        actual_raw = raw_asof(actual_raw, self.config, pd.Timestamp.max - pd.Timedelta(days=1))
        segment, total = evaluate(pred, daily_actual(actual_raw, self.config), self.config)
        score = metrics(total, self.config)
        if save:
            from .reporting import save_report
            save_report(segment, total, score, self.config)
        return segment, total, score


def _run(df, plan_df=None, actual_df=None, *, config=None, forecast_month=None,
        fact_mode=None, plan_mode=None, save=True, report_checks_path=None):
    """Forecast from a date/fact/plan frame or from separate RR frames."""
    config = Config() if config is None else config
    overrides = {key: value for key, value in {
        "forecast_month": forecast_month, "fact_mode": fact_mode, "plan_mode": plan_mode
    }.items() if value is not None}
    config = replace(config, **overrides).resolved()
    config.validate()
    if plan_df is None:
        config = replace(config, plan_value_column="target")
        history_df, plan_df, combined_actual = split_dated_frame(df, config)
        if actual_df is None:
            actual_df = combined_actual
    else:
        history_df = df
    runner = ForecastRunner(config, history_df=history_df, plan_df=plan_df)
    audit, history_months, _ = runner.audit()
    report_checks = runner.verify_report(report_checks_path) if report_checks_path else None
    if report_checks is not None:
        audit["semantics_verified_against_report"] = True
    if runner.needs_money_fallback():
        ranking, selection, control = pd.DataFrame(), None, pd.DataFrame()
    else:
        ranking, selection = runner.select()
        control = runner.control()
    forecast = runner.forecast(save=save)

    segment_comparison = comparison = score = None
    if actual_df is not None and not actual_df.empty:
        segment_comparison, comparison, score = runner.compare(actual_df=actual_df, save=save)
    return {
        "forecast": forecast, "upload": database_upload_frame(forecast, config),
        "upload_total": aggregate_database_upload(database_upload_frame(forecast, config)),
        "daily_totals": upload_frame(forecast, config), "plan_status": runner.target_plan_status(),
        "comparison": comparison, "segment_comparison": segment_comparison,
        "metrics": score, "ranking": ranking, "control": control, "selection": selection,
        "audit": audit, "history_months": history_months, "report_checks": report_checks,
    }


def run(df, plan_df=None, actual_df=None, *, config=None, forecast_month=None,
        fact_mode=None, plan_mode=None, save=True, report_checks_path=None):
    """Run forecasting; retain the complete input error locally for diagnosis."""
    try:
        return _run(df, plan_df, actual_df, config=config, forecast_month=forecast_month,
                    fact_mode=fact_mode, plan_mode=plan_mode, save=save, report_checks_path=report_checks_path)
    except ValueError as exc:
        if save:
            out = Path(config.output_dir if config is not None else Config().output_dir)
            out.mkdir(parents=True, exist_ok=True)
            messages, cause = [], exc
            while cause is not None:
                messages.append(str(cause))
                cause = cause.__cause__
            (out / "input_error.txt").write_text("\n\n".join(messages), encoding="utf-8")
        raise
