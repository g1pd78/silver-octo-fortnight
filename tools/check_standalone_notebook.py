"""Execute all notebook cells in an isolated process without the project package."""
import ast
from pathlib import Path
import subprocess
import sys
import tempfile
import nbformat

ROOT = Path(__file__).resolve().parents[1]


def main():
    notebook = nbformat.read(ROOT / "october_forecast_standalone.ipynb", as_version=4)
    nbformat.validate(notebook)
    sources = []
    for cell in notebook.cells:
        if cell.cell_type != "code":
            continue
        source = cell.source.replace("DEMO = False", "DEMO = True")
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.ImportFrom):
                assert not node.level and not (node.module or "").startswith("collections_forecast")
        sources.append(source)
    with tempfile.TemporaryDirectory() as directory:
        script = Path(directory) / "standalone_check.py"
        script.write_text("""import importlib.util
assert importlib.util.find_spec('collections_forecast') is None
def display(value):
    pass
import plotly.io as pio
pio.show = lambda *args, **kwargs: None
""" + "\n\n".join(sources) + """
assert len(result['comparison']) == 31
assert result['metrics']['complete']
assert len(result['daily_totals']) == 31
assert not result['daily_totals'].date.duplicated().any()
assert len(result['upload']) == len(result['forecast'])
assert (Path(config.output_dir) / 'forecast_upload.csv').exists()
legacy_csv = pd.read_csv(f'forecast_{config.forecast_month}.csv')
assert len(legacy_csv) == 31
assert not legacy_csv.report_dt.duplicated().any()
assert pd.to_datetime(legacy_csv.dt_in).eq(month_start(config.forecast_month)).all()
assert pd.to_datetime(legacy_csv.forecast_date).eq(month_start(config.forecast_month)).all()
print('Standalone notebook execution and old-uploader CSV passed')
""", encoding="utf-8")
        subprocess.run([sys.executable, "-I", str(script)], cwd=directory, check=True)


if __name__ == "__main__":
    main()
