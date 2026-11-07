"""Compatibility entry point for the known-plan collections forecast.

Python: from collections_forecast import Config, ForecastRunner, run
CLI:    python collections_forecast.py --help
The previous unknown-plan implementation is archived in legacy/collections_forecast.py.
"""
from collections_forecast import Config, ForecastRunner, run

__all__ = ["Config", "ForecastRunner", "run"]

if __name__ == "__main__":
    from collections_forecast.cli import main
    raise SystemExit(main())
