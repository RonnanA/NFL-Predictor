import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DATA_DIR = Path(os.getenv("NFL_PREDICTOR_DATA_DIR", PROJECT_ROOT / "data"))
MODELS_DIR = PROJECT_ROOT / "models"
TEST_DIR = MODELS_DIR / "test"
PRODUCTION_DIR = MODELS_DIR / "production"
PROCESSED_DIR = PROJECT_ROOT / DATA_DIR / "processed"
RAW_DIR = PROJECT_ROOT / DATA_DIR / "raw"

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR.mkdir(parents=True, exist_ok=True)

YEAR = 2025