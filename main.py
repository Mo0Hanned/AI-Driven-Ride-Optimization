"""
Smart Fleet Intelligence API
Entry point and FastAPI routes.

This file follows Clean Architecture principles by acting purely as the Controller layer.
It delegates all complex ML and business logic to the `api_services` module and 
validates input/output via the `api_schemas` module.
"""

from fastapi import FastAPI, HTTPException
from api_schemas import (
    Demand6hRequest, RevenueRequest, StockOutRequest, ETARequest,
    Demand15MinRequest, ProfitPlan6hRequest
)
from api_services import PredictionService, DecisionEngineService

# ==========================================
# FastAPI Application Initialization
# ==========================================
app = FastAPI(
    title="Smart Fleet Intelligence API",
    description="API for NYC Taxi Demand (6h & 15m), Revenue, Stock-out, ETA Predictions, and Decision Engine",
    version="2.0.0" # Bumped version for Clean Architecture Refactor
)

# ==========================================
# API Endpoints
# ==========================================

@app.post("/predict/demand_6h", tags=["Predictions"])
def predict_demand_6h(req: Demand6hRequest):
    """
    Predicts the expected taxi demand (number of trips) for a given zone over a 6-hour period.
    """
    try:
        return PredictionService.predict_demand_6h(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/revenue", tags=["Predictions"])
def predict_revenue(req: RevenueRequest):
    """
    Predicts the expected 6-hour revenue percentiles (P50 and P90).
    Automatically integrates with the Demand Prediction model behind the scenes.
    """
    try:
        return PredictionService.predict_revenue(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/stockout", tags=["Predictions"])
def predict_stockout(req: StockOutRequest):
    """
    Predicts the probability of a vehicle stockout (high demand, low supply).
    Automatically integrates with the Demand Prediction model behind the scenes.
    """
    try:
        return PredictionService.predict_stockout(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/eta", tags=["Predictions"])
def predict_eta(req: ETARequest):
    """
    Predicts the Estimated Time of Arrival (ETA) between a pickup and dropoff zone.
    Returns median and upper bound predictions in both seconds and minutes.
    """
    try:
        return PredictionService.predict_eta(req)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"ETA Prediction failed: {str(e)}")

@app.post("/predict/demand_15min", tags=["Predictions"])
def predict_demand_15min(req: Demand15MinRequest):
    """
    Predicts the ultra-short-term (15 minute) taxi demand for a given zone.
    Useful for immediate dispatching decisions.
    """
    try:
        return PredictionService.predict_demand_15min(req)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# ==========================================
# Decision Engine Endpoint
# ==========================================

@app.post("/decision/profit_plan_6h", tags=["Decision Engine"])
def decision_profit_plan_6h(req: ProfitPlan6hRequest):
    """
    Complex Decision Engine that evaluates multiple zones, runs inference pipelines (Demand, Revenue, Stockout),
    and creates a repositioning plan (moving empty vehicles to maximize profit).
    """
    try:
        return DecisionEngineService.evaluate_profit_plan(req)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))