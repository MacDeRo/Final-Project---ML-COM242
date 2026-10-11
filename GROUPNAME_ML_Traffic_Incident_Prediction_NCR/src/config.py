"""Shared paths and constants for the NCR road-crash prediction project."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORIGINAL_DIR = ROOT / "data" / "original"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
DOC_DIR = ROOT / "documentation"
FIG_DIR = DOC_DIR / "figures"

ORIGINAL_WORKBOOK = ORIGINAL_DIR / "Prediction_crashes_calculation_using_mmda_records final1.xlsx"
ORIGINAL_CSV = ORIGINAL_DIR / "source_data_wide.csv"
PROCESSED_CSV = PROCESSED_DIR / "ncr_road_crashes_clean.csv"

RANDOM_STATE = 42
TARGET = "crashes"
FEATURES = ["aadt", "prev_crashes", "lane_km", "year", "pandemic"]
FEATURE_LABELS = {
    "aadt": "AADT (vehicles/day)",
    "prev_crashes": "Previous-year crashes",
    "lane_km": "Lane-kilometers",
    "year": "Year",
    "pandemic": "Pandemic year (2020-2021)",
}

PANDEMIC_YEARS = (2020, 2021)
TEST_ROAD = "EDSA"
# 2015 has no previous-year record, so usable years are 2016-2024 (9 years).
# Chronological 80/20: train 2016-2022 (7 years), test 2023-2024 (2 years).
LAST_TRAIN_YEAR = 2022
MODEL_PATH = MODELS_DIR / "best_model.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata.json"
