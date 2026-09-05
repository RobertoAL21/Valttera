"""Pydantic schemas for the house-price prediction API."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, ConfigDict, Field


class PropertyFeatures(BaseModel):
    """Raw Ames property attributes required by the trained pipeline."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    MSSubClass: int = Field(..., ge=20, le=190)
    MSZoning: str
    LotFrontage: Optional[float] = Field(None, ge=0)
    LotArea: float = Field(..., gt=0)
    Street: str
    Alley: Optional[str]
    LotShape: str
    LandContour: str
    Utilities: str
    LotConfig: str
    LandSlope: str
    Neighborhood: str
    Condition1: str
    Condition2: str
    BldgType: str
    HouseStyle: str
    OverallQual: int = Field(..., ge=1, le=10)
    OverallCond: int = Field(..., ge=1, le=10)
    YearBuilt: int = Field(..., ge=1800, le=2100)
    YearRemodAdd: int = Field(..., ge=1800, le=2100)
    RoofStyle: str
    RoofMatl: str
    Exterior1st: str
    Exterior2nd: str
    MasVnrType: Optional[str]
    MasVnrArea: Optional[float] = Field(None, ge=0)
    ExterQual: str
    ExterCond: str
    Foundation: str
    BsmtQual: Optional[str]
    BsmtCond: Optional[str]
    BsmtExposure: Optional[str]
    BsmtFinType1: Optional[str]
    BsmtFinSF1: float = Field(..., ge=0)
    BsmtFinType2: Optional[str]
    BsmtFinSF2: float = Field(..., ge=0)
    BsmtUnfSF: float = Field(..., ge=0)
    TotalBsmtSF: float = Field(..., ge=0)
    Heating: str
    HeatingQC: str
    CentralAir: str
    Electrical: Optional[str]
    first_floor_sf: float = Field(..., alias="1stFlrSF", ge=0)
    second_floor_sf: float = Field(..., alias="2ndFlrSF", ge=0)
    LowQualFinSF: float = Field(..., ge=0)
    GrLivArea: float = Field(..., gt=0)
    BsmtFullBath: float = Field(..., ge=0)
    BsmtHalfBath: float = Field(..., ge=0)
    FullBath: float = Field(..., ge=0)
    HalfBath: float = Field(..., ge=0)
    BedroomAbvGr: int = Field(..., ge=0)
    KitchenAbvGr: int = Field(..., ge=0)
    KitchenQual: str
    TotRmsAbvGrd: int = Field(..., ge=0)
    Functional: str
    Fireplaces: int = Field(..., ge=0)
    FireplaceQu: Optional[str]
    GarageType: Optional[str]
    GarageYrBlt: Optional[float] = Field(None, ge=1800, le=2100)
    GarageFinish: Optional[str]
    GarageCars: Optional[float] = Field(None, ge=0)
    GarageArea: float = Field(..., ge=0)
    GarageQual: Optional[str]
    GarageCond: Optional[str]
    PavedDrive: str
    WoodDeckSF: float = Field(..., ge=0)
    OpenPorchSF: float = Field(..., ge=0)
    EnclosedPorch: float = Field(..., ge=0)
    three_season_porch: float = Field(..., alias="3SsnPorch", ge=0)
    ScreenPorch: float = Field(..., ge=0)
    PoolArea: float = Field(..., ge=0)
    PoolQC: Optional[str]
    Fence: Optional[str]
    MiscFeature: Optional[str]
    MiscVal: float = Field(..., ge=0)
    MoSold: int = Field(..., ge=1, le=12)
    YrSold: int = Field(..., ge=2006, le=2010)
    SaleType: str
    SaleCondition: str


class PredictionResponse(BaseModel):
    predicted_price: float = Field(..., gt=0, description="Predicted residential sale price in USD.")


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool


class ApplicationInfo(BaseModel):
    name: str
    version: str
    docs_url: str
