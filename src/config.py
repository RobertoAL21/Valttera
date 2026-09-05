"""Central project configuration shared by data, modelling, and API modules."""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_TRAIN_PATH = RAW_DATA_DIR / "train.csv"
PROCESSED_TRAIN_PATH = PROCESSED_DATA_DIR / "train_clean.csv"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "house_price_pipeline.joblib"

TARGET_COLUMN = "SalePrice"
IDENTIFIER_COLUMN = "Id"
RANDOM_STATE = 42
TEST_SIZE = 0.20
