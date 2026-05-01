"""
Data transfer objects and Pydantic schemas for the Smart Fleet Intelligence API.
This module defines the validation and structure for all incoming requests.
"""

from pydantic import BaseModel, Field, ConfigDict
from typing import Literal, List, ClassVar, Any, Optional
import pandas as pd

AllowedWeatherCodes = Literal[
    0.0, 1.0, 2.0, 3.0, 51.0, 53.0, 55.0, 61.0, 63.0, 65.0, 71.0, 73.0, 75.0
]

class BaseMLInput(BaseModel):
    """
    Base class for all Machine Learning input models.
    Provides utility methods to seamlessly convert lists of Pydantic models to Pandas DataFrames
    while applying categorical type casting where necessary.
    """
    cat_cols: ClassVar[list[str]] = []

    def to_df(self) -> pd.DataFrame:
        """Converts a single instance to a Pandas DataFrame."""
        df = pd.DataFrame([self.model_dump()])
        if self.cat_cols:
            df[self.cat_cols] = df[self.cat_cols].astype('category')
        return df

    @classmethod
    def list_to_df(cls, items: List[Any]) -> pd.DataFrame:
        """Converts a list of instances to a Pandas DataFrame."""
        if not items:
            return pd.DataFrame()
        df = pd.DataFrame([item.model_dump() for item in items])
        if cls.cat_cols:
            for col in cls.cat_cols:
                if col in df.columns:
                    df[col] = df[col].astype('category')
        return df


class DemandInput(BaseMLInput):
    """Input schema for Demand (6 Hours) prediction."""
    cat_cols: ClassVar[list[str]] = ['PULocationID', 'pickup_hour', 'day_of_week', 'is_weekend', 'is_rain', 'weather_code', 'is_holiday']
    PULocationID: int = Field(..., ge=1, le=265)
    pickup_hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    is_weekend: int = Field(..., ge=0, le=1)
    temp_c: float
    rain_mm: float = Field(..., ge=0.0)
    is_rain: int = Field(..., ge=0, le=1) 
    weather_code: float
    is_holiday: int = Field(..., ge=0, le=1)
    lag_1_6h: float
    lag_2_6h: float
    lag_4_6h: float
    rolling_mean_24h: float

class Demand6hRequest(BaseModel):
    """Request wrapper for batch processing of 6h Demand predictions."""
    rows: List[DemandInput]
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "rows": [
                    {
                        "PULocationID": 237,
                        "pickup_hour": 18,
                        "day_of_week": 3,
                        "is_weekend": 0,
                        "temp_c": 18.5,
                        "rain_mm": 0.0,
                        "is_rain": 0,
                        "weather_code": 0.0,
                        "is_holiday": 0,
                        "lag_1_6h": 120.5,
                        "lag_2_6h": 115.0,
                        "lag_4_6h": 110.0,
                        "rolling_mean_24h": 95.5
                    }
                ]
            }
        }
    )

class RevenueInput(DemandInput):
    """Input schema for Revenue prediction."""
    cat_cols: ClassVar[list[str]] = ['PULocationID', 'pickup_hour', 'day_of_week', 'is_weekend']
    rev_lag_1_6h: float
    rev_lag_1_week: float
    rev_rolling_mean_7d: float
    rev_rolling_mean_30d: float
    avg_fare: float
    tip_rate: float

class RevenueRequest(BaseModel):
    """Request wrapper for batch processing of Revenue predictions."""
    rows: List[RevenueInput]
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "rows": [
                    {
                        "PULocationID": 161,
                        "pickup_hour": 8,
                        "day_of_week": 1,
                        "is_weekend": 0,
                        "temp_c": 18.5,
                        "rain_mm": 0.0,
                        "is_rain": 0,
                        "weather_code": 0.0,
                        "is_holiday": 0,
                        "lag_1_6h": 120.5,
                        "lag_2_6h": 115.0,
                        "lag_4_6h": 110.0,
                        "rolling_mean_24h": 95.5,
                        "rev_lag_1_6h": 4500.50,
                        "rev_lag_1_week": 4200.75,
                        "rev_rolling_mean_7d": 4100.0,
                        "rev_rolling_mean_30d": 3950.0,
                        "avg_fare": 18.50,
                        "tip_rate": 0.15
                    }
                ]
            }
        }
    )

class StockOutInput(BaseMLInput):
    """Input schema for Stockout Risk prediction."""
    cat_cols: ClassVar[list[str]] = ['is_rain', 'weather_code', 'is_holiday']
    zone_id: int = Field(..., ge=1, le=265)
    hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    is_weekend: int = Field(..., ge=0, le=1)
    pickup_count: float
    dropoff_count: float
    net_flow: float
    activity_ratio: float
    lag_1_pickup: float
    lag_1_dropoff: float
    lag_1_net_flow: float
    temp_c: float
    rain_mm: float
    is_rain: int = Field(..., ge=0, le=1)
    weather_code: float
    is_holiday: int = Field(..., ge=0, le=1)
    lag_1_6h: float
    lag_2_6h: float
    lag_4_6h: float
    rolling_mean_24h: float

class StockOutRequest(BaseModel):
    """Request wrapper for batch processing of Stockout predictions."""
    rows: List[StockOutInput]
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "rows": [
                    {
                        "zone_id": 132,
                        "hour": 22,
                        "day_of_week": 4,
                        "is_weekend": 0,
                        "pickup_count": 85.0,
                        "dropoff_count": 120.0,
                        "net_flow": -35.0,
                        "activity_ratio": 1.2,
                        "lag_1_pickup": 90.0,
                        "lag_1_dropoff": 110.0,
                        "lag_1_net_flow": -20.0,
                        "temp_c": 15.0,
                        "rain_mm": 0.0,
                        "is_rain": 0,
                        "weather_code": 0.0,
                        "is_holiday": 0,
                        "lag_1_6h": 80.5,
                        "lag_2_6h": 75.0,
                        "lag_4_6h": 70.0,
                        "rolling_mean_24h": 65.5
                    }
                ]
            }
        }
    )

class ETAInput(BaseModel):
    """Input schema for Travel Time (ETA) prediction."""
    pickup_datetime: str = Field(..., description="Format: YYYY-MM-DD HH:MM:SS")
    PULocationID: int = Field(..., ge=1, le=265)
    DOLocationID: int = Field(..., ge=1, le=265)
    trip_distance: float = Field(..., ge=0.0)
    temp_c: float = 20.0
    rain_mm: float = Field(0.0, ge=0.0)
    weather_code: float = 0.0

class ETARequest(BaseModel):
    """Request wrapper for batch processing of ETA predictions."""
    rows: List[ETAInput]
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "rows": [
                    {
                        "pickup_datetime": "2024-05-15 14:30:00",
                        "PULocationID": 237,
                        "DOLocationID": 236,
                        "trip_distance": 1.5,
                        "temp_c": 22.0,
                        "rain_mm": 0.0,
                        "weather_code": 0.0
                    }
                ]
            }
        }
    )

class Demand15MinRowInput(BaseModel):
    """Input schema for ultra-short-term (15m) demand prediction."""
    PULocationID: int
    pickup_cnt: float
    lag_1: float
    lag_4: float
    lag_96: float
    roll_mean_1h: float
    roll_mean_3h: float
    hour: int
    minute: int
    day_of_week: int
    is_weekend: int
    month: int
    temp_c: float
    rain_mm: float
    is_rain: int
    weather_code: float

class Demand15MinRequest(BaseModel):
    """Request wrapper for batch processing of 15m Demand predictions."""
    rows: List[Demand15MinRowInput]
    round_to_int: Optional[bool] = True 
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "rows": [
                    {
                        "PULocationID": 237,
                        "pickup_cnt": 15.0,
                        "lag_1": 12.0,
                        "lag_4": 10.0,
                        "lag_96": 14.0,
                        "roll_mean_1h": 11.5,
                        "roll_mean_3h": 10.2,
                        "hour": 14,
                        "minute": 30,
                        "day_of_week": 2,
                        "is_weekend": 0,
                        "month": 5,
                        "temp_c": 22.0,
                        "rain_mm": 0.0,
                        "is_rain": 0,
                        "weather_code": 0.0
                    }
                ],
                "round_to_int": True
            }
        }
    )

class DecisionConstraints(BaseModel):
    """Rules and constraints guiding the decision engine optimizations."""
    max_reposition_eta_min: float = Field(25.0, gt=0)
    max_empty_km: float = Field(12.0, gt=0)
    max_moves_total: Optional[int] = None
    min_net_gain_per_driver: float = 10.0
    calibrated_stockout_target: float = 0.55
    calibrated_stockout_source_max: float = Field(0.65, ge=0.0, le=1.0)
    min_target_gap: int = Field(1, ge=0)
    min_source_coverage_ratio: float = Field(0.75, ge=0.0, le=1.0)

class BusinessParams(BaseModel):
    """Financial and operational parameters used to calculate profit margins."""
    profit_mode: str = "detailed_costs"
    driver_cost_per_hour: float = 25.0
    fuel_cost_per_km: float = 0.3
    idle_cost_per_min: float = 0.60
    reposition_cost_per_km: float = 1.20
    commission_rate: float = 0.20
    driver_acceptance_prob: float = Field(0.85, ge=0.0, le=1.0)
    traffic_surge_multiplier: float = 1.0
    weather_risk_multiplier: float = 1.0
    sla_penalty_per_underserved_trip: float = 5.0
    event_zone_priority_boost: float = 1.2
    airport_zone_protection: bool = True
    strategic_reserve_ratio: float = 0.1

class ZonePairOverride(BaseModel):
    """Explicit overrides for specific zone-to-zone routing heuristics."""
    from_zone: int
    to_zone: int
    distance_km: float
    eta_min: float

class ZoneDecisionInput(BaseModel):
    """State input representing a single taxi zone for the decision engine."""
    zone_id: int = Field(..., ge=1, le=265)
    current_drivers: int = Field(..., ge=0)
    allow_as_source: bool = True
    allow_as_target: bool = True
    is_event_zone: bool = False
    is_airport_zone: bool = False
    
    # Shared Temporal/Weather Features
    hour: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    is_weekend: int = Field(..., ge=0, le=1)
    temp_c: float
    rain_mm: float = Field(..., ge=0.0)
    is_rain: int = Field(..., ge=0, le=1)
    weather_code: float
    is_holiday: int = Field(..., ge=0, le=1)
    
    # Demand features
    lag_1_6h: float
    lag_2_6h: float
    lag_4_6h: float
    rolling_mean_24h: float
    
    # Revenue features
    rev_lag_1_6h: float
    rev_lag_1_week: float
    rev_rolling_mean_7d: float
    rev_rolling_mean_30d: float
    avg_fare: float
    tip_rate: float
    
    # Stockout features
    pickup_count: float
    dropoff_count: float
    net_flow: float
    activity_ratio: float
    lag_1_pickup: float
    lag_1_dropoff: float
    lag_1_net_flow: float

class ProfitPlan6hRequest(BaseModel):
    """Request payload to generate a 6h fleet repositioning strategic plan."""
    question: str = "I want a plan to maximize profit in the next 6 hours"
    target_datetime: str
    current_zone: int = Field(..., ge=1, le=265)
    include_geojson: bool = False
    constraints: DecisionConstraints
    business_params: BusinessParams
    pair_overrides: List[ZonePairOverride] = []
    zones: List[ZoneDecisionInput]
