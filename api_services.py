"""
Decision Engine for Fleet Repositioning and Profit Planning.
Handles strategic repositioning optimization for taxi fleets.
"""

import math
from typing import Dict, Any
from api_schemas import (
    ProfitPlan6hRequest, DemandInput, RevenueInput, StockOutInput
)
from ml_core import model_manager


class DecisionEngineService:
    """Service handling the complex rules logic for Fleet Repositioning and Profit Planning."""

    @staticmethod
    def evaluate_profit_plan(req: ProfitPlan6hRequest) -> Dict[str, Any]:
        """Evaluates multiple zones and creates a strategic repositioning plan to maximize profit."""
        if model_manager.demand_model is None or model_manager.rev_model_p50 is None or model_manager.rev_model_p90 is None or model_manager.stockout_model is None:
            raise ValueError("Missing ML models for Decision Engine: demand_model, rev_model_p50, rev_model_p90, and stockout_model are required.")

        zone_evaluations = []
        total_baseline_profit = 0.0
        target_deficit_before = 0

        # Resolve allow_as_source and allow_as_target with defaults and business rules
        def resolve_allow_as_source(zone, business_params):
            """Resolve allow_as_source: explicit value > business rule > default True"""
            if zone.allow_as_source is not None:
                return zone.allow_as_source
            # Airport protection: airports cannot be sources (pull-out is protected)
            if zone.is_airport_zone and business_params.airport_zone_protection:
                return False
            return True
        
        def resolve_allow_as_target(zone, business_params):
            """Resolve allow_as_target: explicit value > default True"""
            if zone.allow_as_target is not None:
                return zone.allow_as_target
            return True

        for zone in req.zones:
            # Resolve source/target flags early
            allow_as_source_resolved = resolve_allow_as_source(zone, req.business_params)
            allow_as_target_resolved = resolve_allow_as_target(zone, req.business_params)

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

            # Evaluate as Target/Source
            # NOTE: Airport is protected from being a SOURCE (pulling drivers), not from being a TARGET
            movable_drivers = max(0, zone.current_drivers - math.ceil(drivers_needed * req.constraints.min_source_coverage_ratio))
            is_source = (
                allow_as_source_resolved
                and surplus > 0 
                and pred_stockout <= req.constraints.calibrated_stockout_source_max
            )
            is_target = (
                allow_as_target_resolved 
                and deficit >= req.constraints.min_target_gap
                and pred_stockout >= req.constraints.calibrated_stockout_target
            )
            
            # Determine source and target reason codes
            if surplus > 0 and allow_as_source_resolved:
                if pred_stockout > req.constraints.calibrated_stockout_source_max:
                    source_reason_code = "STOCKOUT_TOO_HIGH"
                else:
                    source_reason_code = "HAS_SURPLUS"
            elif not allow_as_source_resolved:
                if zone.is_airport_zone and req.business_params.airport_zone_protection:
                    source_reason_code = "AIRPORT_PROTECTED"
                else:
                    source_reason_code = "DISALLOWED"
            else:
                source_reason_code = "NO_SURPLUS"
            
            if deficit >= req.constraints.min_target_gap and allow_as_target_resolved:
                if pred_stockout < req.constraints.calibrated_stockout_target:
                    target_reason_code = "SURPLUS_TOO_HIGH"
                else:
                    target_reason_code = "HAS_DEFICIT"
            elif not allow_as_target_resolved:
                target_reason_code = "DISALLOWED"
            else:
                target_reason_code = "DEFICIT_TOO_SMALL"

            zone_evaluations.append({
                "zone_id": zone.zone_id,
                "current_drivers": zone.current_drivers,
                "allow_as_source": allow_as_source_resolved,
                "allow_as_target": allow_as_target_resolved,
                "is_airport_zone": zone.is_airport_zone,
                "demand_6h": pred_demand,
                "cycle_time_min": round(cycle_time_min, 2),
                "trips_per_driver_6h": trips_per_driver_6h,
                "drivers_needed_6h": drivers_needed,
                "driver_gap": driver_gap,
                "deficit": deficit,
                "surplus": surplus,
                "movable_drivers_under_coverage_policy": movable_drivers,
                "revenue_p50": round(pred_rev_p50, 2),
                "revenue_p90": round(pred_rev_p90, 2),
                "uncertainty": round(pred_rev_p90 - pred_rev_p50, 2),
                "stockout_prob": round(pred_stockout, 4),
                "served_ratio_baseline": round(served_ratio, 4),
                "baseline_profit": round(baseline_profit, 2),
                "source_candidate": is_source,
                "target_candidate": is_target,
                "source_reason_code": source_reason_code,
                "source_reason_text": f"surplus={surplus}, movable={movable_drivers}, stockout={round(pred_stockout, 4)}" if surplus > 0 else f"deficit={deficit}, allow_as_source={allow_as_source_resolved}",
                "target_reason_code": target_reason_code,
                "target_reason_text": f"deficit={deficit}, stockout={round(pred_stockout, 4)}" if deficit >= req.constraints.min_target_gap else f"deficit={deficit} < min_gap({req.constraints.min_target_gap})",
                **({"geojson": model_manager.get_zone_geojson(zone.zone_id)} if req.include_geojson else {})
            })

        # D. Build Move Candidates (Repositioning)
        reposition_plan = []
        rejected_moves = []
        override_map = {(ov.from_zone, ov.to_zone): ov for ov in req.pair_overrides}
        
        sources = [z for z in zone_evaluations if z["source_candidate"]]
        targets = [z for z in zone_evaluations if z["target_candidate"]]

        # Debug summary with counters
        debug_summary = {
            "candidate_pairs_count": len(sources) * len(targets),
            "source_candidate_count": len(sources),
            "target_candidate_count": len(targets),
            "rejected_pairs_count": 0,  # Will be updated after pair evaluation
            "executed_moves_count": 0,  # Will be updated after move processing
            "notes": []
        }
        if not sources:
            debug_summary["notes"].append("no source candidates available")
        if not targets:
            debug_summary["notes"].append("no target candidates available")

        total_moved_count = 0
        total_move_cost = 0.0
        total_expected_uplift = 0.0
        
        # Track movable quantities separately to avoid mutating zone_evaluations
        zone_movable_surplus = {src["zone_id"]: max(0, src["current_drivers"] - math.ceil(src["drivers_needed_6h"] * req.constraints.min_source_coverage_ratio)) for src in sources}
        zone_remaining_deficit = {tgt["zone_id"]: tgt["deficit"] for tgt in targets}

        # Compute max_moves_total if not provided
        effective_max_moves_total = req.constraints.max_moves_total
        if effective_max_moves_total is None:
            total_movable_surplus = sum(zone_movable_surplus.values())
            effective_max_moves_total = max(1, int(total_movable_surplus * 0.1)) if total_movable_surplus > 0 else 0

        # Pre-compute all valid candidate pairs with their metrics for ranking
        candidate_pairs = []
        for src in sources:
            src_zone_id = src["zone_id"]
            for tgt in targets:
                tgt_zone_id = tgt["zone_id"]
                
                ov = override_map.get((src_zone_id, tgt_zone_id))
                if not ov:
                    rejected_moves.append({
                        "from_zone": src_zone_id,
                        "to_zone": tgt_zone_id,
                        "reason": "no override provided; requires pair_override with distance_km and eta_min"
                    })
                    debug_summary["rejected_pairs_count"] += 1
                    continue
                
                dist_km, eta_min = ov.distance_km, ov.eta_min
                if dist_km > req.constraints.max_empty_km or eta_min > req.constraints.max_reposition_eta_min:
                    rejected_moves.append({
                        "from_zone": src_zone_id,
                        "to_zone": tgt_zone_id,
                        "reason": f"distance_km ({dist_km}) exceeds max_empty_km ({req.constraints.max_empty_km})" if dist_km > req.constraints.max_empty_km else f"eta_min ({eta_min}) exceeds max_reposition_eta_min ({req.constraints.max_reposition_eta_min})"
                    })
                    debug_summary["rejected_pairs_count"] += 1
                    continue
                
                # Calculate move cost: idle + reposition + driver wages + fuel
                # Apply traffic surge multiplier to reposition cost
                idle_cost = eta_min * req.business_params.idle_cost_per_min
                reposition_cost = dist_km * req.business_params.reposition_cost_per_km * req.business_params.traffic_surge_multiplier
                driver_wage_cost = (eta_min / 60.0) * req.business_params.driver_cost_per_hour  # convert minutes to hours
                fuel_cost = dist_km * req.business_params.fuel_cost_per_km
                
                move_cost_per_driver = idle_cost + reposition_cost + driver_wage_cost + fuel_cost
                
                # Calculate revenue per driver at target with weather risk adjustment
                # Apply weather risk multiplier as a discount on revenue (increases conservatism)
                rev_per_driver_tgt = (tgt["revenue_p50"] * req.business_params.commission_rate) / max(1, tgt["drivers_needed_6h"])
                rev_per_driver_tgt = rev_per_driver_tgt / req.business_params.weather_risk_multiplier  # Higher multiplier = lower expected revenue
                
                # Apply event zone priority boost to target zone revenue if it's an event zone
                if tgt.get("is_event_zone", False):
                    rev_per_driver_tgt = rev_per_driver_tgt * req.business_params.event_zone_priority_boost
                
                net_gain_per_driver = rev_per_driver_tgt - move_cost_per_driver

                if net_gain_per_driver >= req.constraints.min_net_gain_per_driver:
                    candidate_pairs.append({
                        "src": src,
                        "tgt": tgt,
                        "src_zone_id": src_zone_id,
                        "tgt_zone_id": tgt_zone_id,
                        "dist_km": dist_km,
                        "eta_min": eta_min,
                        "move_cost_per_driver": move_cost_per_driver,
                        "rev_per_driver_tgt": rev_per_driver_tgt,
                        "net_gain_per_driver": net_gain_per_driver
                    })
                else:
                    rejected_moves.append({
                        "from_zone": src_zone_id,
                        "to_zone": tgt_zone_id,
                        "reason": f"net_gain_per_driver ({round(net_gain_per_driver, 2)}) below min_net_gain_per_driver ({req.constraints.min_net_gain_per_driver})"
                    })
                    debug_summary["rejected_pairs_count"] += 1
        
        # RANK candidates by net_gain_per_driver (descending), with priority for current_zone
        def ranking_key(pair):
            # Primary: net_gain_per_driver (higher is better)
            # Tiebreaker: prioritize targets in current_zone, deprioritize sources from current_zone
            src_is_current = (pair["src_zone_id"] == req.current_zone)
            tgt_is_current = (pair["tgt_zone_id"] == req.current_zone)
            
            # Current zone bonus: stronger multiplier to increase impact
            # Reduce penalty for pulling from current zone: -15% (was -10%)
            # Increase reward for adding to current zone: +20% (was +5%)
            current_zone_bonus = (
                pair["net_gain_per_driver"] * (-0.15 if src_is_current else 0) 
                + pair["net_gain_per_driver"] * (0.20 if tgt_is_current else 0)
            )
            return (pair["net_gain_per_driver"] + current_zone_bonus, -pair["dist_km"])
        
        candidate_pairs.sort(key=ranking_key, reverse=True)
        
        # Process ranked pairs
        for pair in candidate_pairs:
            if effective_max_moves_total is not None and total_moved_count >= effective_max_moves_total:
                break
            
            src_zone_id = pair["src_zone_id"]
            tgt_zone_id = pair["tgt_zone_id"]
            
            # Calculate remaining moves allowed
            remaining_moves = (
                effective_max_moves_total - total_moved_count
                if effective_max_moves_total is not None
                else float("inf")
            )
            
            if remaining_moves <= 0:
                break
            
            # Use local tracking (do NOT mutate zone_evaluations)
            current_movable = zone_movable_surplus.get(src_zone_id, 0)
            current_deficit = zone_remaining_deficit.get(tgt_zone_id, 0)
            
            if current_movable <= 0 or current_deficit <= 0:
                continue
            
            drivers_to_move = min(int(current_movable), int(current_deficit), int(remaining_moves))
            
            if drivers_to_move > 0:
                effective_moved = drivers_to_move * req.business_params.driver_acceptance_prob
                expected_net_gain = pair["net_gain_per_driver"] * effective_moved
                total_move_cost_for_batch = pair["move_cost_per_driver"] * effective_moved

                reposition_plan.append({
                    "from_zone": src_zone_id,
                    "to_zone": tgt_zone_id,
                    "drivers_to_move": drivers_to_move,
                    "effective_drivers_moved": round(effective_moved, 2),
                    "driver_acceptance_prob": req.business_params.driver_acceptance_prob,
                    "eta_min": pair["eta_min"],
                    "distance_km": pair["dist_km"],
                    "eta_source": "override",
                    "net_gain_per_driver": round(pair["net_gain_per_driver"], 1),
                    "expected_net_gain": round(expected_net_gain, 2),
                    "move_cost": round(total_move_cost_for_batch, 2),
                    "business_reason": f"Move {drivers_to_move} drivers from surplus zone {src_zone_id} to deficit zone {tgt_zone_id}; expected gain per driver = {round(pair['net_gain_per_driver'], 1)}.",
                    "rule": "ranked by net_gain_per_driver under ETA/distance/supply/deficit constraints",
                    **({"from_zone_geojson": model_manager.get_zone_geojson(src_zone_id), "to_zone_geojson": model_manager.get_zone_geojson(tgt_zone_id)} if req.include_geojson else {})
                })
                debug_summary["executed_moves_count"] += 1

                # Update local tracking (do NOT mutate zone_evaluations)
                zone_movable_surplus[src_zone_id] -= drivers_to_move
                zone_remaining_deficit[tgt_zone_id] -= drivers_to_move
                total_moved_count += drivers_to_move
                total_move_cost += total_move_cost_for_batch
                total_expected_uplift += expected_net_gain

        target_deficit_after = sum(zone_remaining_deficit.values())
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
            "rejected_moves": rejected_moves,
            "zone_evaluations": zone_evaluations,
            "debug_summary": debug_summary
        }
