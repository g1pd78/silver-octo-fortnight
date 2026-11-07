from dataclasses import asdict, dataclass, replace
import pandas as pd


@dataclass(frozen=True)
class Config:
    db_path: str = "input/m1p_database_v2.db"
    plan_table: str | None = None  # None means raw_table
    plan_value_column: str = "target"
    plan_is_forecast: int = 0
    plan_available_column: str | None = None
    portfolio_path: str | None = None
    inflow_path: str | None = None
    operations_path: str | None = None
    calendar_path: str | None = None
    reference_path: str | None = None
    output_dir: str = "outputs/forecast"
    forecast_month: str | None = None
    development_start: str | None = None
    development_end: str | None = None
    control_start: str | None = None
    control_end: str | None = None
    fact_mode: str | None = None  # daily or cumulative, explicitly chosen by the caller
    plan_mode: str | None = None
    min_train_months: int = 12
    raw_table: str = "rr1plus_data_raw"
    raw_available_column: str | None = None
    raw_date_encoding: str = "iso"  # iso, yyyymmdd, s, ms, us, ns
    segments: tuple[str, ...] = (
        "subproduct", "dpd_bucket", "pvr_bucket", "terbank", "inflow"
    )
    tolerance_pp: float = 1.0
    seed: int = 42
    reference_months: int = 3
    enable_catboost: bool = True
    enable_sarimax: bool = True
    catboost_iterations: int = 600
    catboost_depths: tuple[int, ...] = (4, 6)
    blend_weights: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0)
    country: str = "RU"
    require_all_folds: bool = True
    require_calendar_overrides: bool = False

    def resolved(self):
        if self.forecast_month is None:
            raise ValueError("Specify forecast_month, e.g. Config(forecast_month='2026-11', fact_mode='daily', plan_mode='daily')")
        target = pd.Period(self.forecast_month, "M")
        values = dict(forecast_month=str(target))
        # Six development months followed by three untouched control months.
        for name, offset in [("development_start", 9), ("development_end", 4),
                             ("control_start", 3), ("control_end", 1)]:
            value = getattr(self, name)
            values[name] = str(pd.Period(value, "M")) if value is not None else str(target - offset)
        return replace(self, **values)

    def validate(self):
        if self.fact_mode not in {"daily", "cumulative"} or self.plan_mode not in {"daily", "cumulative"}:
            raise ValueError("Explicitly set fact_mode and plan_mode to 'daily' or 'cumulative' after checking the source report")
        if self.plan_is_forecast not in (0, 1):
            raise ValueError("plan_is_forecast must be 0 or 1; plan rows must not be summed twice")
        if not self.segments or len(set(self.segments)) != len(self.segments):
            raise ValueError("segments must be nonempty and unique")
        if self.min_train_months < 1 or self.tolerance_pp <= 0:
            raise ValueError("Invalid training length or tolerance")
        if self.reference_months < 1 or self.catboost_iterations < 1 or not self.catboost_depths:
            raise ValueError("Reference window and model sizes must be positive")
        if any(not 1 <= d <= 10 for d in self.catboost_depths):
            raise ValueError("catboost_depths must be between 1 and 10")
        if any(not 0 <= weight <= 1 for weight in self.blend_weights):
            raise ValueError("Blend weights must be between zero and one")
        if self.raw_date_encoding not in {"iso", "yyyymmdd", "s", "ms", "us", "ns"}:
            raise ValueError("Unsupported raw_date_encoding")
        if not (pd.Period(self.development_start, 'M') <= pd.Period(self.development_end, 'M')
                < pd.Period(self.control_start, 'M') <= pd.Period(self.control_end, 'M')
                < pd.Period(self.forecast_month, 'M')):
            raise ValueError("Development, control and target periods must be ordered and disjoint")

    def as_dict(self):
        return asdict(self)


def month_start(month):
    return pd.Period(month, freq="M").start_time


def origin(month):
    return month_start(month) - pd.Timedelta(nanoseconds=1)


def days(month):
    p = pd.Period(month, freq="M")
    return pd.date_range(p.start_time, p.end_time.normalize(), freq="D")
