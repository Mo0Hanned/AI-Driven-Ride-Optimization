"""
Data transfer objects and Pydantic schemas for the Smart Fleet Intelligence API.
This module defines the validation and structure for all incoming requests.
"""

from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator
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


class RevenueInput(DemandInput):
    """Input schema for Revenue prediction."""
    cat_cols: ClassVar[list[str]] = ['PULocationID', 'pickup_hour', 'day_of_week', 'is_weekend']
    rev_lag_1_6h: float
    rev_lag_1_week: float
    rev_rolling_mean_7d: float
    rev_rolling_mean_30d: float
    avg_fare: float
    tip_rate: float


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


class DecisionConstraints(BaseModel):
    """Rules and constraints guiding the decision engine optimizations."""
    max_reposition_eta_min: float = Field(25.0, gt=0)
    max_empty_km: float = Field(12.0, gt=0)
    max_moves_total: Optional[int] = Field(
        None,
        ge=0,
        description="If None, calculated as 0 when no drivers exist, otherwise max(1, int(total_drivers * 0.1))"
    )
    min_net_gain_per_driver: float = Field(10.0, ge=0)
    
    calibrated_stockout_target: float = Field(0.55, ge=0.0, le=1.0)
    calibrated_stockout_source_max: float = Field(0.65, ge=0.0, le=1.0)
    min_target_gap: int = Field(1, ge=0)
    min_source_coverage_ratio: float = Field(0.75, ge=0.0, le=1.0)

class BusinessParams(BaseModel):
    """Financial and operational parameters used to calculate profit margins."""
    profit_mode: str = "detailed_costs"
    driver_cost_per_hour: float = 25.0
    fuel_cost_per_km: float = 0.3
    idle_cost_per_min: float = 0.6
    reposition_cost_per_km: float = 1.2
    commission_rate: float = 0.2
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
    allow_as_source: Optional[bool] = None
    allow_as_target: Optional[bool] = None
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
    constraints: DecisionConstraints = Field(default_factory=DecisionConstraints)
    business_params: BusinessParams = Field(default_factory=BusinessParams)
    pair_overrides: List[ZonePairOverride] = []
    zones: List[ZoneDecisionInput]
    
    @model_validator(mode='after')
    def compute_dynamic_constraints(self):
        """Auto-compute max_moves_total based on total movable surplus drivers.
        If total_movable == 0, max_moves_total becomes 0. Otherwise, 10% of total movable surplus (min 1).
        Note: This is a placeholder; actual calculation happens in evaluate_profit_plan after movable surplus is computed.
        """
        # Placeholder: actual computation in service
        return self
