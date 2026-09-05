"""Unit tests for interpretable engineered features."""

from __future__ import annotations

import pandas as pd

from src.features.engineering import add_engineered_features


def test_add_engineered_features_calculates_expected_values() -> None:
    source = pd.DataFrame(
        {
            "TotalBsmtSF": [800], "1stFlrSF": [1000], "2ndFlrSF": [500],
            "YrSold": [2010], "YearBuilt": [2000], "YearRemodAdd": [2005],
            "FullBath": [2], "HalfBath": [1], "BsmtFullBath": [1], "BsmtHalfBath": [0],
            "WoodDeckSF": [100], "OpenPorchSF": [50], "EnclosedPorch": [0],
            "3SsnPorch": [20], "ScreenPorch": [30], "OverallQual": [7],
            "GrLivArea": [1500], "GarageArea": [400],
        }
    )

    engineered = add_engineered_features(source)

    assert engineered.loc[0, "TotalSF"] == 2300
    assert engineered.loc[0, "HouseAge"] == 10
    assert engineered.loc[0, "YearsSinceRemodel"] == 5
    assert engineered.loc[0, "TotalBathrooms"] == 3.5
    assert engineered.loc[0, "TotalOutdoorSpace"] == 200
    assert engineered.loc[0, "OverallQualGrLivArea"] == 10_500
    assert engineered.loc[0, "HasRemodel"] == 1
    assert engineered.loc[0, "HasGarage"] == 1
    assert engineered.loc[0, "HasBasement"] == 1
    assert "TotalSF" not in source.columns
