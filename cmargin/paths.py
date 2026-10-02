"""Data and result locations. Override with the environment variables CMARGIN_DATA and CMARGIN_RESULTS."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("CMARGIN_DATA", ROOT / "data"))
RESULTS = Path(os.environ.get("CMARGIN_RESULTS", ROOT / "results"))
RESULTS.mkdir(parents=True, exist_ok=True)
