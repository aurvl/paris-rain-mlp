"""Train, evaluate, and export the Paris rain MLP."""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from paris_rain_mlp.pipeline import run_experiment  # noqa: E402

if __name__ == "__main__":
    run_experiment(PROJECT_ROOT)
