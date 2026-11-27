"""1-hour session addendum: the hashed forecast (scripts/forecast.py) re-run for 6 attempts per arm.

forecast.py is imported unchanged; only M_NEW (attempts per arm) is set to 6 and the output folder is
addendum_1h/results/. Same inputs, same seed (20261127), same method. Run from the repository root:

    uv run --no-project --with numpy --with scipy python \
        training_source/project/experiments/final_session_forecast/scripts/forecast_6.py
"""

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("forecast", HERE / "forecast.py")
f = importlib.util.module_from_spec(spec)
spec.loader.exec_module(f)

f.M_NEW = 6
f.F = HERE.parent / "addendum_1h"
(f.F / "results").mkdir(parents=True, exist_ok=True)
f.main()

tables = f.F / "results" / "forecast_tables.md"
tables.write_text(tables.read_text().replace("(of 12)", "(of 6)"))
