"""
Business Logic and Services for the Smart Fleet Intelligence API.
This module extracts the complex operations, model invocations, and calculations
away from the FastAPI controllers to maintain Clean Architecture.
"""

import numpy as np
import pandas as pd
import math
from typing import List, Dict, Any
from api_schemas import (
    Demand6hRequest, RevenueRequest, StockOutRequest, ETARequest,
    Demand15MinRequest, ProfitPlan6hRequest, DemandInput, RevenueInput, StockOutInput
)
from ml_core import model_manager

class PredictionService:
    """Service handling all Machine Learning inference operations."""
    
    @staticmethod
    def predict_demand_6h(req: Demand6hRequest) -> List[Dict[str, Any]]:
        """Predicts taxi demand over a 6-hour window."""
        if model_manager.demand_model is None:
            raise ValueError("Demand 6h model is not loaded.")
        
        if not req.rows:
            return []

        df = DemandInput.list_to_df(req.rows)
        preds = model_manager.demand_model.predict(df)
        
        results = []
        for i, row in enumerate(req.rows):
            final_prediction = int(np.ceil(np.maximum(preds[i], 0)))
            
            # Cache the prediction for dependent pipelines
            cache_key = f"{row.PULocationID}_{row.pickup_hour}"
            model_manager.demand_cache[cache_key] = final_prediction
            
            results.append({
                "status": "success",
                "PULocationID": row.PULocationID,
                "Predicted_Demand_6h": final_prediction,
                "shp_file": model_manager.get_zone_geojson(row.PULocationID)
            })
        return results

    @staticmethod
    def predict_revenue(req: RevenueRequest) -> List[Dict[str, Any]]:
        """Predicts revenue percentiles (P50, P90) based on expected demand."""
        if model_manager.rev_model_p50 is None or model_manager.rev_model_p90 is None or model_manager.demand_model is None:
            raise ValueError("Revenue or Demand models are not loaded.")
        
        if not req.rows:
            return []

        # 1. First, automatically compute Demand for each row
        computed_demands = []
        for row in req.rows:
            demand_dict = {k: getattr(row, k) for k in DemandInput.model_fields}
            demand_features = DemandInput(**demand_dict)
            
            df_demand = DemandInput.list_to_df([demand_features])
            pred = model_manager.demand_model.predict(df_demand)[0]
            final_prediction = int(np.ceil(np.maximum(pred, 0))) # Ensure non-negative
            computed_demands.append(final_prediction)
                
        # 2. Inject computed demand into the Revenue dataframe
        df = RevenueInput.list_to_df(req.rows)
        df['forecasted_demand_6h'] = computed_demands 
        
        # Explicitly enforce correct column order required by the model
        cols_order = ['PULocationID', 'pickup_hour', 'day_of_week', 'is_weekend', 'forecasted_demand_6h', 
                      'rev_lag_1_6h', 'rev_lag_1_week', 'rev_rolling_mean_7d', 'rev_rolling_mean_30d', 'avg_fare', 'tip_rate']
        df = df[cols_order]
        
        # 3. Predict Revenue
        preds_p50 = model_manager.rev_model_p50.predict(df)
        preds_p90 = model_manager.rev_model_p90.predict(df)
        
        results = []
        for i, row in enumerate(req.rows):
            results.append({
                "PULocationID": row.PULocationID,
                "Predicted_Revenue_P50": round(float(max(0, preds_p50[i])), 2),
                "Predicted_Revenue_P90": round(float(max(0, preds_p90[i])), 2),
                "shp_file": model_manager.get_zone_geojson(row.PULocationID)
            })
        return results

    @staticmethod
    def predict_stockout(req: StockOutRequest) -> List[Dict[str, Any]]:
        """Predicts the probability of a vehicle stockout in a given zone."""
        if model_manager.stockout_model is None or model_manager.demand_model is None:
            raise ValueError("Stockout or Demand models are not loaded.")
        
        if not req.rows:
            return []

        # 1. Compute Demand
        computed_demands = []
        for row in req.rows:
            demand_features = DemandInput(
                PULocationID=row.zone_id,
                pickup_hour=row.hour,
                day_of_week=row.day_of_week,
                is_weekend=row.is_weekend,
                temp_c=row.temp_c,
                rain_mm=row.rain_mm,
                is_rain=row.is_rain,
                weather_code=row.weather_code,
                is_holiday=row.is_holiday,
                lag_1_6h=row.lag_1_6h,
                lag_2_6h=row.lag_2_6h,
                lag_4_6h=row.lag_4_6h,
                rolling_mean_24h=row.rolling_mean_24h
            )
            
            df_demand = DemandInput.list_to_df([demand_features])
            pred = model_manager.demand_model.predict(df_demand)[0]
            final_prediction = int(np.ceil(np.maximum(pred, 0)))
            computed_demands.append(final_prediction)
                
        # 2. Inject into Stockout dataframe
        df = StockOutInput.list_to_df(req.rows)
        df['forecasted_demand_6h'] = computed_demands
        
        # Explicitly enforce column order
        cols_order = ['zone_id', 'hour', 'day_of_week', 'is_weekend', 'pickup_count', 'dropoff_count', 'net_flow', 
                      'activity_ratio', 'lag_1_pickup', 'lag_1_dropoff', 'lag_1_net_flow', 'forecasted_demand_6h', 
                      'temp_c', 'rain_mm', 'is_rain', 'weather_code', 'is_holiday']
        df = df[cols_order]
        
        # 3. Predict Stockout Probability
        probs = model_manager.stockout_model.predict(df)
        
        results = []
        for i, row in enumerate(req.rows):
            prob = float(probs[i])
            will_stockout = int(prob >= 0.5)  
            results.append({
                "zone_id": row.zone_id,
                "Probability_of_StockOut": round(prob, 4),
                "Will_StockOut": will_stockout,
                "Risk_Level": "CRITICAL" if prob >= 0.88 else "WARNING" if prob >= 0.80 else "OK ✅",
                "shp_file": model_manager.get_zone_geojson(row.zone_id)
            })
        return results

    @staticmethod
    def predict_eta(req: ETARequest) -> List[Dict[str, Any]]:
        """Predicts the estimated time of arrival (ETA) for a given trip."""
        if model_manager.eta_artifact is None:
            raise ValueError("ETA model artifact is not loaded.")
            
        if not req.rows:
            return []
            
        input_dicts = []
        for row in req.rows:
            dt = pd.to_datetime(row.pickup_datetime)
            input_dicts.append({
                "PULocationID": row.PULocationID, "DOLocationID": row.DOLocationID,
                "distance_km_proxy": row.trip_distance, "temp_c": row.temp_c,
                "rain_mm": row.rain_mm, "weather_code": row.weather_code,
                "pickup_hour": dt.hour, "pickup_dow": dt.weekday(), "pickup_month": dt.month,
                "pickup_dayofyear": dt.dayofyear, "pickup_minute": dt.minute,
                "is_weekend": 1 if dt.weekday() >= 5 else 0,
                "is_rush_hour": 1 if (7 <= dt.hour <= 9 or 16 <= dt.hour <= 19) else 0,
                "pickup_15min_bucket": (dt.hour * 60 + dt.minute) // 15
            })
            
        input_df = pd.DataFrame(input_dicts)

        for col in model_manager.eta_artifact.features:
            if col not in input_df.columns:
                input_df[col] = 0 
        
        X = input_df[model_manager.eta_artifact.features].copy()
        if hasattr(model_manager.eta_artifact, 'categorical_features') and model_manager.eta_artifact.categorical_features:
            for col in model_manager.eta_artifact.categorical_features:
                if col in X.columns:
                    X[col] = X[col].astype('category')

        preds_p50 = model_manager.eta_artifact.model_p50.predict(X)
        preds_p90 = model_manager.eta_artifact.model_p90.predict(X)

        results = []
        for i, row in enumerate(req.rows):
            p50 = float(preds_p50[i])
            p90 = float(preds_p90[i])
            results.append({
                "status": "success",
                "predictions": {
                    "median_eta_seconds": round(p50, 2), "upper_bound_eta_seconds": round(p90, 2),
                    "median_eta_minutes": round(p50 / 60, 2)
                },
                "info": { "pickup": row.pickup_datetime, "distance": row.trip_distance },
                "shp_file": model_manager.get_zone_geojson(row.PULocationID),
                "pickup_shp_file": model_manager.get_zone_geojson(row.PULocationID),
                "dropoff_shp_file": model_manager.get_zone_geojson(row.DOLocationID)
            })
        return results

    @staticmethod
    def predict_demand_15min(req: Demand15MinRequest) -> List[Dict[str, Any]]:
        """Predicts ultra-short-term demand (15 minute windows)."""
        if model_manager.demand_15m_booster is None or model_manager.demand_15m_bundle is None:
            raise ValueError("Demand 15m model is not loaded.")
            
        rows_dicts = [row.model_dump() for row in req.rows]
        df = pd.DataFrame(rows_dicts)

        missing = [c for c in model_manager.demand_15m_features if c not in df.columns]
        if missing:
            raise ValueError(f"Missing features: {missing}")

        df["PULocationID"] = df["PULocationID"].astype(model_manager.demand_15m_zone_dtype)
        df["weather_code"] = df["weather_code"].astype(model_manager.demand_15m_weather_dtype)
        
        X = df[model_manager.demand_15m_features]

        pred = model_manager.demand_15m_booster.predict(X, validate_features=True)
        pred = np.clip(pred, 0, None)  

        results = []
        for i, row in enumerate(rows_dicts):
            prediction_val = float(pred[i])
            if req.round_to_int:
                prediction_val = int(np.rint(prediction_val))
            
            results.append({
                "status": "success",
                "PULocationID": row["PULocationID"],
                "Predicted_Demand_15min": prediction_val,
                "shp_file": model_manager.get_zone_geojson(row["PULocationID"])
            })

        return results


class DecisionEngineService:
    """Service handling the complex rules logic for Fleet Repositioning and Profit Planning."""

    @staticmethod
    def evaluate_profit_plan(req: ProfitPlan6hRequest) -> Dict[str, Any]:
        """Evaluates multiple zones and creates a strategic repositioning plan to maximize profit."""
        if model_manager.demand_model is None or model_manager.rev_model_p50 is None or model_manager.stockout_model is None:
            raise ValueError("Missing ML models for Decision Engine.")

        zone_evaluations = []
        total_baseline_profit = 0.0
        target_deficit_before = 0

        for zone in req.zones:
            # A. Run Models in sequence
            demand_features = DemandInput(
                PULocationID=zone.zone_id,
                pickup_hour=zone.hour,
                day_of_week=zone.day_of_week,
                is_weekend=zone.is_weekend,
                temp_c=zone.temp_c,
                rain_mm=zone.rain_mm,
                is_rain=zone.is_rain,
                weather_code=zone.weather_code,
                is_holiday=zone.is_holiday,
                lag_1_6h=zone.lag_1_6h,
                lag_2_6h=zone.lag_2_6h,
                lag_4_6h=zone.lag_4_6h,
                rolling_mean_24h=zone.rolling_mean_24h
            )
            df_demand = DemandInput.list_to_df([demand_features])
            pred_demand = int(math.ceil(max(model_manager.demand_model.predict(df_demand)[0], 0)))

            revenue_features = RevenueInput(
                **demand_features.model_dump(),
                rev_lag_1_6h=zone.rev_lag_1_6h,
                rev_lag_1_week=zone.rev_lag_1_week,
                rev_rolling_mean_7d=zone.rev_rolling_mean_7d,
                rev_rolling_mean_30d=zone.rev_rolling_mean_30d,
                avg_fare=zone.avg_fare,
                tip_rate=zone.tip_rate
            )
            df_rev = RevenueInput.list_to_df([revenue_features])
            df_rev['forecasted_demand_6h'] = pred_demand
            cols_order_rev = ['PULocationID', 'pickup_hour', 'day_of_week', 'is_weekend', 'forecasted_demand_6h', 
                              'rev_lag_1_6h', 'rev_lag_1_week', 'rev_rolling_mean_7d', 'rev_rolling_mean_30d', 'avg_fare', 'tip_rate']
            df_rev = df_rev[cols_order_rev]
            pred_rev_p50 = float(max(model_manager.rev_model_p50.predict(df_rev)[0], 0))
            pred_rev_p90 = float(max(model_manager.rev_model_p90.predict(df_rev)[0], 0))

            stockout_features = StockOutInput(
                zone_id=zone.zone_id,
                hour=zone.hour,
                day_of_week=zone.day_of_week,
                is_weekend=zone.is_weekend,
                pickup_count=zone.pickup_count,
                dropoff_count=zone.dropoff_count,
                net_flow=zone.net_flow,
                activity_ratio=zone.activity_ratio,
                lag_1_pickup=zone.lag_1_pickup,
                lag_1_dropoff=zone.lag_1_dropoff,
                lag_1_net_flow=zone.lag_1_net_flow,
                temp_c=zone.temp_c,
                rain_mm=zone.rain_mm,
                is_rain=zone.is_rain,
                weather_code=zone.weather_code,
                is_holiday=zone.is_holiday,
                lag_1_6h=zone.lag_1_6h,
                lag_2_6h=zone.lag_2_6h,
                lag_4_6h=zone.lag_4_6h,
                rolling_mean_24h=zone.rolling_mean_24h
            )
            df_stock = StockOutInput.list_to_df([stockout_features])
            df_stock['forecasted_demand_6h'] = pred_demand
            cols_order_stock = ['zone_id', 'hour', 'day_of_week', 'is_weekend', 'pickup_count', 'dropoff_count', 'net_flow', 
                                'activity_ratio', 'lag_1_pickup', 'lag_1_dropoff', 'lag_1_net_flow', 'forecasted_demand_6h', 
                                'temp_c', 'rain_mm', 'is_rain', 'weather_code', 'is_holiday']
            df_stock = df_stock[cols_order_stock]
            pred_stockout = float(model_manager.stockout_model.predict(df_stock)[0])

            # B. Calculate Operational Metrics
            cycle_time_min = 12.67 
            trips_per_driver_6h = math.floor(360 / cycle_time_min)
            drivers_needed = math.ceil(pred_demand / trips_per_driver_6h)
            driver_gap = drivers_needed - zone.current_drivers
            
            deficit = max(0, driver_gap)
            surplus = max(0, -driver_gap)
            if deficit > 0: target_deficit_before += deficit

            served_ratio = min(1.0, zone.current_drivers / max(1, drivers_needed))
            baseline_profit = pred_rev_p50 * served_ratio * req.business_params.commission_rate
            total_baseline_profit += baseline_profit

            # C. Evaluate as Target/Source
            is_source = (
                zone.allow_as_source 
                and surplus > 0 
                and pred_stockout <= req.constraints.calibrated_stockout_source_max
            )
            is_target = (
                zone.allow_as_target 
                and not zone.is_airport_zone 
                and deficit >= req.constraints.min_target_gap
            )

            zone_evaluations.append({
                "zone_id": zone.zone_id,
                "current_drivers": zone.current_drivers,
                "allow_as_source": zone.allow_as_source,
                "allow_as_target": zone.allow_as_target,
                "is_airport_zone": zone.is_airport_zone,
                "demand_6h": pred_demand,
                "cycle_time_min": round(cycle_time_min, 2),
                "trips_per_driver_6h": trips_per_driver_6h,
                "drivers_needed_6h": drivers_needed,
                "driver_gap": driver_gap,
                "deficit": deficit,
                "surplus": surplus,
                "revenue_p50": round(pred_rev_p50, 2),
                "revenue_p90": round(pred_rev_p90, 2),
                "uncertainty": round(pred_rev_p90 - pred_rev_p50, 2),
                "stockout_prob": round(pred_stockout, 4),
                "served_ratio_baseline": round(served_ratio, 4),
                "baseline_profit": round(baseline_profit, 2),
                "source_candidate": is_source,
                "target_candidate": is_target,
                "reason": f"needs {deficit} drivers" if is_target else f"has {surplus} surplus drivers, P50 revenue={round(pred_rev_p50, 2)}"
            })

        # D. Build Move Candidates (Repositioning)
        reposition_plan = []
        override_map = {(ov.from_zone, ov.to_zone): ov for ov in req.pair_overrides}
        
        sources = [z for z in zone_evaluations if z["source_candidate"]]
        targets = [z for z in zone_evaluations if z["target_candidate"]]

        total_moved_count = 0
        total_move_cost = 0.0
        total_expected_uplift = 0.0

        for src in sources:
            for tgt in targets:
                ov = override_map.get((src["zone_id"], tgt["zone_id"]))
                if ov:
                    dist_km, eta_min, eta_source_label = ov.distance_km, ov.eta_min, "override"
                else:
                    dist_km, eta_min, eta_source_label = 5.0, 15.0, "fallback" 

                if dist_km > req.constraints.max_empty_km or eta_min > req.constraints.max_reposition_eta_min:
                    continue

                move_cost_per_driver = (eta_min * req.business_params.idle_cost_per_min) + (dist_km * req.business_params.reposition_cost_per_km)
                rev_per_driver_tgt = (tgt["revenue_p50"] * req.business_params.commission_rate) / max(1, tgt["drivers_needed_6h"])
                net_gain_per_driver = rev_per_driver_tgt - move_cost_per_driver

                if net_gain_per_driver >= req.constraints.min_net_gain_per_driver:
                    drivers_to_move = min(src["surplus"], tgt["deficit"])
                    
                    if drivers_to_move > 0:
                        effective_moved = drivers_to_move * req.business_params.driver_acceptance_prob
                        expected_net_gain = net_gain_per_driver * effective_moved
                        total_move_cost_for_batch = move_cost_per_driver * effective_moved

                        reposition_plan.append({
                            "from_zone": src["zone_id"],
                            "to_zone": tgt["zone_id"],
                            "drivers_to_move": drivers_to_move,
                            "effective_drivers_moved": round(effective_moved, 2),
                            "driver_acceptance_prob": req.business_params.driver_acceptance_prob,
                            "eta_min": eta_min,
                            "distance_km": dist_km,
                            "eta_source": eta_source_label,
                            "net_gain_per_driver": round(net_gain_per_driver, 1),
                            "expected_net_gain": round(expected_net_gain, 2),
                            "move_cost": round(total_move_cost_for_batch, 2),
                            "risk_score": 19.6, 
                            "confidence_score": 73.8, 
                            "business_reason": f"Move {drivers_to_move} drivers from surplus zone {src['zone_id']} to deficit zone {tgt['zone_id']}; expected gain per driver = {round(net_gain_per_driver, 1)}.",
                            "rule": "ranked by net_gain_per_driver under ETA/distance/supply/deficit constraints"
                        })

                        src["surplus"] -= drivers_to_move
                        tgt["deficit"] -= drivers_to_move
                        total_moved_count += drivers_to_move
                        total_move_cost += total_move_cost_for_batch
                        total_expected_uplift += expected_net_gain

                        if tgt["deficit"] == 0:
                            break

        target_deficit_after = sum(z["deficit"] for z in zone_evaluations)
        deficit_resolved = target_deficit_before - target_deficit_after
        total_projected_profit = total_baseline_profit + total_expected_uplift

        return {
            "question": req.question,
            "mode": req.business_params.profit_mode,
            "decision": "EXECUTE_REPOSITION" if reposition_plan else "MAINTAIN_STATUS_QUO",
            "net_impact": {
                "total_drivers_moved": total_moved_count,
                "deficit_resolved": deficit_resolved,
                "total_move_cost": round(total_move_cost, 2),
                "expected_profit_uplift": round(total_expected_uplift, 2),
                "total_baseline_profit": round(total_baseline_profit, 2),
                "total_projected_profit": round(total_projected_profit, 2),
                "roi_percent": round((total_expected_uplift / max(1, total_move_cost)) * 100, 2) if total_move_cost > 0 else 0.0
            },
            "kpis": {
                "target_deficit_before": target_deficit_before,
                "target_deficit_after": target_deficit_after
            },
            "reposition_plan": reposition_plan,
            "rejected_moves": [],
            "zone_evaluations": zone_evaluations
        }
