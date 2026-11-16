import copy
import json
from pathlib import Path
import tempfile
import unittest
import nbformat
import pandas as pd
from collections_forecast import Config
from collections_forecast.reporting import diagnostic_panels, load_saved_run_diagnostics, save_run_diagnostics


class DiagnosticPanelTests(unittest.TestCase):
    def example(self):
        selected = "calendar_profile|total_catboost_d6|1"
        rows = [{"candidate": candidate, "months": 6, "max_error_pp": worst, "mae_pp": worst / 2}
                for candidate, worst in [(selected, 2.), ("plan_baseline|total_catboost_d6|1", 2.),
                    ("reference_profile|bottom_up|0", 3.), ("plan_baseline|bottom_up|0", 5.),
                    ("plan_execution_curve|bottom_up|0", 6.)]]
        return {
            "selection": {"leaf": "calendar_profile", "total": "total_catboost_d6", "weight": 1.},
            "forecast": pd.DataFrame({"model": [selected]}),
            "ranking": pd.DataFrame(rows),
            "development_scores": pd.DataFrame([
                {"candidate": selected, "month": "2026-01", "max_error_pp": 2., "mae_pp": 1.,
                 "bias_pp": -.5, "hit_rate": .5, "complete": True},
                {"candidate": "plan_baseline|bottom_up|0", "month": "2026-01", "max_error_pp": 5.}]),
            "control": pd.DataFrame([{"month": "2026-09", "max_error_pp": 4., "mae_pp": 2.,
                                      "bias_pp": 1.5, "hit_rate": .2, "complete": True}]),
            "plan_status": {"mode": "selected_model", "validation_coverage": {
                "control": {"excluded_months": ["2026-08"]}}, "reasons": [], "warnings": []},
            "metrics": {"mae_pp": 6., "bias_pp": 6., "max_error_pp": 10., "hit_rate": 0.,
                        "scored_days": 2, "unscored_days": 29, "complete": False,
                        "last_scored_date": "2026-10-10", "last_scored_error_pp": 2.},
            "comparison": pd.DataFrame({"date": ["2026-10-01", "2026-10-10"],
                "plan_cumulative": [987654321., 987654321.], "actual_efficiency_pct": [101., 102.],
                "forecast_efficiency_pct": [111., 104.], "signed_error_pp": [10., 2.],
                "absolute_error_pp": [10., 2.]}),
            "model_status": pd.DataFrame([{"model": "total_sarimax", "status": "unavailable",
                                           "reason": "missing historical plan"}]),
        }

    def test_compact_panels_explain_effective_weight_and_preserve_result(self):
        result = self.example()
        original = copy.deepcopy(result)
        panels = diagnostic_panels(result, Config(forecast_month="2026-10"))
        self.assertIn("100% total_catboost_d6", panels[0])
        self.assertIn("выше факта на всех", panels[0])
        self.assertIn("1–5", panels[0])
        self.assertIn("6–конец", panels[0])
        self.assertNotIn("987654321", "\n".join(panels))
        self.assertIn("plan_execution_curve|bottom_up|0", panels[1])
        self.assertEqual(panels[1].count("total_catboost_d6|1"), 1)
        self.assertIn("2026-09", panels[2])
        self.assertIn("точность не подтверждена", panels[2])
        self.assertIn("2026-08", panels[2])
        self.assertIn("total_sarimax", panels[3])
        for key in ["ranking", "development_scores", "comparison", "control"]:
            pd.testing.assert_frame_equal(result[key], original[key])
        self.assertEqual(result["selection"], original["selection"])

    def write_saved(self, directory):
        result = self.example()
        config = Config(forecast_month="2026-10", output_dir=str(directory)).resolved()
        out = Path(directory)
        for key, filename in [("forecast", "forecast.csv"), ("ranking", "development_ranking.csv"),
                              ("development_scores", "development_scores.csv"), ("control", "control_scores.csv"),
                              ("comparison", "comparison_total.csv"), ("model_status", "model_status.csv")]:
            result[key].to_csv(out / filename, index=False)
        for key, filename in [("plan_status", "forecast_run_status.json"), ("metrics", "metrics.json")]:
            (out / filename).write_text(json.dumps(result[key]), encoding="utf-8")
        (out / "manifest.json").write_text(json.dumps({"config": config.as_dict(), "selection": result["selection"]}), encoding="utf-8")
        return result, config

    def test_saved_results_can_be_diagnosed_without_retraining_or_rewriting(self):
        with tempfile.TemporaryDirectory() as directory:
            _, config = self.write_saved(directory)
            files = {p.name: p.read_bytes() for p in Path(directory).iterdir()}
            result, loaded_config = load_saved_run_diagnostics(directory)
            save_run_diagnostics(result, loaded_config)
            for filename, content in files.items():
                self.assertEqual((Path(directory) / filename).read_bytes(), content)
            summary = (Path(directory) / "diagnostics_summary.txt").read_text(encoding="utf-8")
            self.assertIn("2026-09", summary)
            self.assertNotIn("987654321", summary)
            self.assertEqual(loaded_config.forecast_month, config.forecast_month)

    def test_fallback_and_missing_optional_files_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            result, config = self.write_saved(directory)
            (Path(directory) / "control_scores.csv").unlink()
            loaded, _ = load_saved_run_diagnostics(directory)
            self.assertIn("Нет control_scores.csv.", diagnostic_panels(loaded, config)[-1])
            panels = diagnostic_panels({"selection": None, "plan_status": {"mode": "money_fallback"}}, config)
            self.assertIn("резерв", panels[0])
            self.assertIn("Подбор не выполнен", panels[1])

    def test_helper_notebook_runs_from_saved_outputs_without_project_import(self):
        path = Path(__file__).resolve().parents[1] / "forecast_diagnostics.ipynb"
        notebook = nbformat.read(path, as_version=4)
        nbformat.validate(notebook)
        with tempfile.TemporaryDirectory() as directory:
            self.write_saved(directory)
            namespace = {"__name__": "__main__"}
            for cell in notebook.cells:
                if cell.cell_type == "code":
                    self.assertNotIn("from collections_forecast", cell.source)
                    source = cell.source.replace('OUTPUT_DIR = "outputs/forecast_rr_fixed_10_10"',
                        "OUTPUT_DIR = " + repr(directory))
                    exec(compile(source, str(path), "exec"), namespace)
            self.assertTrue((Path(directory) / "diagnostics_summary.txt").exists())
            self.assertEqual(namespace["saved_result"]["selection"]["weight"], 1.)


if __name__ == "__main__":
    unittest.main()
