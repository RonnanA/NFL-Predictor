import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = Path(os.getenv("NFL_PREDICTOR_DATA_DIR", PROJECT_ROOT / "data"))
PROCESSED_DIR = PROJECT_ROOT / DATA_DIR / "processed"
RAW_DIR = PROJECT_ROOT / DATA_DIR / "raw"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)