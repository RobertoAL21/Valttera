"""Unit tests for deterministic data cleaning behavior."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.preprocessing import clean_housing_data, save_processed_data, validate_housing_data


def test_clean_housing_data_removes_id_and_preserves_raw_input() -> None:
    raw_data = pd.DataFrame(
        {
            "Id": [1, 1],
            "MSSubClass": [20, 20],
            "SalePrice": [100_000, 100_000],
        }
    )

    cleaned = clean_housing_data(raw_data)

    assert len(cleaned) == 1
    assert "Id" not in cleaned.columns
    assert str(cleaned["MSSubClass"].dtype) == "string"
    assert "Id" in raw_data.columns
    assert len(raw_data) == 2


def test_validate_housing_data_reports_quality_issues() -> None:
    data = pd.DataFrame(
        {
            "SalePrice": [100_000, -1],
            "MoSold": [1, 13],
            "OverallQual": [5, 11],
            "OverallCond": [5, 0],
            "YearBuilt": [2000, 2000],
            "YearRemodAdd": [2000, 1999],
        }
    )

    report = validate_housing_data(data)

    assert report.impossible_values == {
        "SalePrice": 1,
        "MoSold": 1,
        "OverallQual": 1,
        "OverallCond": 1,
        "YearRemodAdd_before_YearBuilt": 1,
    }


def test_save_processed_data_rejects_raw_destination(tmp_path: object) -> None:
    with pytest.raises(ValueError, match="must not be written"):
        save_processed_data(pd.DataFrame({"SalePrice": [1]}), tmp_path / "raw" / "data.csv")
