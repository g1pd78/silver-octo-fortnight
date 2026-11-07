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
""" + "\n\n".join(sources) + "\nassert len(result['comparison']) == 31\nassert result['metrics']['complete']\nprint('Standalone notebook execution passed')\n", encoding="utf-8")
        subprocess.run([sys.executable, "-I", str(script)], cwd=directory, check=True)


if __name__ == "__main__":
    main()
