"""Deterministic, domain-based feature engineering for Ames housing data."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


FEATURE_DESCRIPTIONS = {
    "TotalSF": "Combined basement and first-/second-floor finished area in square feet.",
    "HouseAge": "Age of the house at the time of sale.",
    "YearsSinceRemodel": "Years since the most recent remodel at the time of sale.",
    "TotalBathrooms": "Full-equivalent bathroom count, where half baths count as 0.5.",
    "TotalOutdoorSpace": "Combined deck and porch area in square feet.",
    "OverallQualGrLivArea": "Interaction between overall quality and above-ground living area.",
    "HasRemodel": "Whether the property was remodeled after construction.",
    "HasGarage": "Whether the property has any garage area.",
    "HasBasement": "Whether the property has any basement area.",
}


def add_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    """Add interpretable features without fitting or learning from the dataset.

    The input frame is not mutated. All calculations use attributes available
    for a property at prediction time, so the same function can be used for
    training and inference within one scikit-learn pipeline.
    """
    _require_columns(
        data,
        {
            "TotalBsmtSF",
            "1stFlrSF",
            "2ndFlrSF",
            "YrSold",
            "YearBuilt",
            "YearRemodAdd",
            "FullBath",
            "HalfBath",
            "BsmtFullBath",
            "BsmtHalfBath",
            "WoodDeckSF",
            "OpenPorchSF",
            "EnclosedPorch",
            "3SsnPorch",
            "ScreenPorch",
            "OverallQual",
            "GrLivArea",
            "GarageArea",
        },
    )

    engineered = data.copy()
    engineered["TotalSF"] = (
        engineered["TotalBsmtSF"] + engineered["1stFlrSF"] + engineered["2ndFlrSF"]
    )
    engineered["HouseAge"] = (engineered["YrSold"] - engineered["YearBuilt"]).clip(lower=0)
    engineered["YearsSinceRemodel"] = (
        engineered["YrSold"] - engineered["YearRemodAdd"]
    ).clip(lower=0)
    engineered["TotalBathrooms"] = (
        engineered["FullBath"]
        + 0.5 * engineered["HalfBath"]
        + engineered["BsmtFullBath"]
        + 0.5 * engineered["BsmtHalfBath"]
    )
    engineered["TotalOutdoorSpace"] = (
        engineered["WoodDeckSF"]
        + engineered["OpenPorchSF"]
        + engineered["EnclosedPorch"]
        + engineered["3SsnPorch"]
        + engineered["ScreenPorch"]
    )
    engineered["OverallQualGrLivArea"] = engineered["OverallQual"] * engineered["GrLivArea"]
    engineered["HasRemodel"] = (engineered["YearRemodAdd"] > engineered["YearBuilt"]).astype("int8")
    engineered["HasGarage"] = (engineered["GarageArea"] > 0).astype("int8")
    engineered["HasBasement"] = (engineered["TotalBsmtSF"] > 0).astype("int8")

    return engineered


def _require_columns(data: pd.DataFrame, required_columns: Iterable[str]) -> None:
    missing = sorted(set(required_columns).difference(data.columns))
    if missing:
        missing_display = ", ".join(missing)
        raise ValueError(f"Cannot engineer features; missing required columns: {missing_display}.")
