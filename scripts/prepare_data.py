"""Validate raw Ames data and create the separate processed dataset."""

from __future__ import annotations

from src.config import PROCESSED_TRAIN_PATH, RAW_TRAIN_PATH
from src.data.preprocessing import clean_housing_data, load_raw_data, save_processed_data, validate_housing_data


def main() -> None:
    raw_data = load_raw_data(RAW_TRAIN_PATH)
    report = validate_housing_data(raw_data)
    cleaned_data = clean_housing_data(raw_data)
    destination = save_processed_data(cleaned_data, PROCESSED_TRAIN_PATH)

    print(f"Validated {report.rows:,} rows and {report.columns} columns.")
    print(f"Saved cleaned data to {destination}.")


if __name__ == "__main__":
    main()
