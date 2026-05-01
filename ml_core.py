"""
Machine Learning Core Module for the Smart Fleet Intelligence API.
Handles the loading, caching, and distribution of predictive models,
along with utilities for handling spatial data (GeoJSON).
"""

import os
import joblib
import pickle
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List
from pandas.api.types import CategoricalDtype
import lightgbm as lgb
import geopandas as gpd
import __main__

def sha256_file(path: str) -> str:
    """Calculates the SHA-256 hash of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

# ==========================================
# Dataclasses required for unpickling ETA Model
# ==========================================
@dataclass
class EtaBaselineTables:
    global_median_duration: float
    global_slowdown_index: float
    od_hour_median: Any
    od_15min_median: Any
    distance_bucket_median: Any

@dataclass
class EtaModelArtifact:
    model_p50: Any
    model_p90: Any
    features: List[str]
    categorical_features: List[str]
    baselines: EtaBaselineTables
    congestion_stats: Any
    categorical_levels: Dict[str, List[str]]
    fillna_policy: Dict[str, Any]
    feature_gen_config: Dict[str, Any] = None

# Inject classes into the __main__ module space so joblib can deserialize them safely.
setattr(__main__, "EtaModelArtifact", EtaModelArtifact)
setattr(__main__, "EtaBaselineTables", EtaBaselineTables)

class MLModelManager:
    """
    Singleton-style manager to load and store Machine Learning models
    in memory, preventing repetitive disk I/O operations.
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(MLModelManager, cls).__new__(cls)
            cls._instance.initialize_models()
        return cls._instance

    def initialize_models(self):
        """Loads all required models into memory upon initialization."""
        self.demand_model = None
        self.rev_model_p50 = None
        self.rev_model_p90 = None
        self.stockout_model = None
        self.eta_artifact = None
        
        self.demand_15m_bundle = None
        self.demand_15m_booster = None
        self.demand_15m_features = []
        self.demand_15m_zone_dtype = None
        self.demand_15m_weather_dtype = None
        self.demand_15m_artifact_sha256 = ""
        
        # In-Memory Cache for Model Pipelining
        self.demand_cache: Dict[str, float] = {}

        self.gdf = None
        self._load_shapefile()

        try:
            print("⏳ Loading Machine Learning models...")
            # Load standard sklearn/LGBM models
            self.demand_model = joblib.load('model/lgbm_demand_model_ml1(6h).pkl')
            self.rev_model_p50 = joblib.load('model/lgbm_revenue_p50_ml5.pkl')
            self.rev_model_p90 = joblib.load('model/lgbm_revenue_p90_ml5.pkl')
            self.stockout_model = joblib.load('model/lgbm_stockout_model_ml4.pkl')
            self.eta_artifact = joblib.load('model/eta_model_artifact.joblib')
            
            # Load specialized Demand 15m bundle
            bundle_path = 'model/lgbm_bundle_tplus15m.pkl'
            with open(bundle_path, "rb") as f:
                self.demand_15m_bundle = pickle.load(f)
            
            booster_path = self.demand_15m_bundle["model_path"]
            if not os.path.exists(booster_path):
                booster_path = os.path.join('model', os.path.basename(booster_path.replace('\\', '/')))
                
            self.demand_15m_booster = lgb.Booster(model_file=booster_path)
            self.demand_15m_artifact_sha256 = sha256_file(booster_path)
            self.demand_15m_features = self.demand_15m_bundle["feature_cols"]
            self.demand_15m_zone_dtype = CategoricalDtype(categories=self.demand_15m_bundle["zone_categories"])
            self.demand_15m_weather_dtype = CategoricalDtype(categories=self.demand_15m_bundle["weather_categories"])

            print("✅ All models loaded successfully!")
        except Exception as e:
            print(f"⚠️ Warning: Could not load some models. Error: {e}")

    def _load_shapefile(self):
        """Loads the NYC Taxi Zones shapefile for GeoJSON generation."""
        try:
            print("⏳ Loading Shapefile...")
            gdf = gpd.read_file("taxi_zones/taxi_zones.shp")
            if gdf.crs != "EPSG:4326":
                gdf = gdf.to_crs("EPSG:4326")
            self.gdf = gdf
            print("✅ Shapefile loaded successfully!")
        except Exception as e:
            print(f"⚠️ Warning: Could not load shapefile. Error: {e}")

    def get_zone_geojson(self, zone_id: int) -> dict | None:
        """Retrieves the GeoJSON geometry for a given taxi zone ID."""
        if self.gdf is None:
            return None
        zone = self.gdf[self.gdf['LocationID'] == zone_id]
        if zone.empty:
            return None
        return json.loads(zone.to_json())

# Export a single global instance
model_manager = MLModelManager()
