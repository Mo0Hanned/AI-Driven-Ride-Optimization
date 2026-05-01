# API Reference

This document outlines the available endpoints, their purposes, and expected responses in the Smart Fleet Intelligence platform.

All endpoints accept `POST` requests and support **batch processing** (accepting a list of `rows` instead of a single object). 

All predictions return an attached `shp_file` containing the GeoJSON geometry for the requested zone, allowing instant frontend rendering.

---

### `POST /predict/demand_6h`
**Description:** Predicts the total volume of taxi trips originating from a specific zone over the next 6 hours.
- **Input:** `Demand6hRequest` (Features include weather, holidays, and lag historical data).
- **Output:** `Predicted_Demand_6h` (Integer >= 0)

---

### `POST /predict/demand_15min`
**Description:** Predicts the immediate, ultra-short-term demand for a specific zone over the next 15 minutes.
- **Input:** `Demand15MinRequest`
- **Output:** `Predicted_Demand_15min` (Integer >= 0)

---

### `POST /predict/revenue`
**Description:** Predicts the exact revenue generated in a specific zone.
- **Input:** `RevenueRequest` (Does NOT require `forecasted_demand_6h`, but requires `demand_features` to compute it automatically).
- **Output:**
  - `Predicted_Revenue_P50`: The expected median revenue (50% confidence).
  - `Predicted_Revenue_P90`: The upper-bound revenue (90% confidence, useful for surge planning).

---

### `POST /predict/stockout`
**Description:** Evaluates the risk of a zone completely running out of available taxis within the next hour.
- **Input:** `StockOutRequest` (Requires `demand_features`, traffic ratios, and pickup/dropoff historical counts).
- **Output:**
  - `Probability_of_StockOut`: Float between 0 and 1.
  - `Will_StockOut`: Boolean (0 or 1).
  - `Risk_Level`: Textual indicator (`CRITICAL`, `WARNING`, or `OK ✅`).

---

### `POST /predict/eta`
**Description:** Calculates the highly accurate Estimated Time of Arrival (ETA) between two zones.
- **Input:** `ETARequest` (Requires PU/DO Location IDs, distance, weather).
- **Output:**
  - `median_eta_seconds`
  - `upper_bound_eta_seconds`
  - `median_eta_minutes`

---

### `POST /decision/profit_plan_6h`
**Description:** The core decision engine. It takes an array of zones, evaluates their deficits/surpluses using the predictive models, and outputs a strict repositioning plan.
- **Input:** `ProfitPlan6hRequest` (Requires zones, their feature sets, and business constraints like fuel cost and SLA penalties).
- **Output:**
  - `decision`: `EXECUTE_REPOSITION` or `MAINTAIN_STATUS_QUO`
  - `net_impact`: Dictionary showing projected ROI, total move cost, and expected profit uplift.
  - `reposition_plan`: List of exact moves (`Move X drivers from Zone Y to Zone Z`).
