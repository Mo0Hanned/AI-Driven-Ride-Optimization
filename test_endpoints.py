from fastapi.testclient import TestClient
from main1 import app
from api_schemas import Demand6hRequest, RevenueRequest, StockOutRequest, ETARequest, Demand15MinRequest, ProfitPlan6hRequest
import json

client = TestClient(app)

print("Starting tests...")

endpoints = {
    "Demand 6h": ("/predict/demand_6h", Demand6hRequest.model_config["json_schema_extra"]["example"]),
    "Revenue": ("/predict/revenue", RevenueRequest.model_config["json_schema_extra"]["example"]),
    "StockOut": ("/predict/stockout", StockOutRequest.model_config["json_schema_extra"]["example"]),
    "ETA": ("/predict/eta", ETARequest.model_config["json_schema_extra"]["example"]),
    "Demand 15m": ("/predict/demand_15min", Demand15MinRequest.model_config["json_schema_extra"]["example"])
}

for name, (url, payload) in endpoints.items():
    print(f"\nTesting {name} Endpoint ({url})...")
    try:
        response = client.post(url, json=payload)
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            resp_json = response.json()
            if isinstance(resp_json, list) and len(resp_json) > 0:
                print(f"Success. Output length: {len(resp_json)}")
            elif isinstance(resp_json, dict):
                print(f"Success. Output keys: {resp_json.keys()}")
            else:
                print(f"Success. Unknown output format: {resp_json}")
        else:
            print(f"Error! Response: {response.text}")
    except Exception as e:
        print(f"Exception during testing {name}: {e}")

print("\nTesting Profit Plan 6h Endpoint (/decision/profit_plan_6h)...")
try:
    payload = ProfitPlan6hRequest(
        target_datetime="2024-05-15 18:00:00",
        current_zone=161,
        constraints={
            "max_reposition_eta_min": 25.0,
            "max_empty_km": 12.0,
            "calibrated_stockout_target": 0.55,
            "calibrated_stockout_source_max": 0.65,
            "min_target_gap": 1,
            "min_source_coverage_ratio": 0.75,
            "min_net_gain_per_driver": 10.0
        },
        business_params={},
        zones=[
            {
                "zone_id": 161,
                "current_drivers": 10,
                "demand_features": Demand6hRequest.model_config["json_schema_extra"]["example"]["rows"][0],
                "revenue_features": RevenueRequest.model_config["json_schema_extra"]["example"]["rows"][0],
                "stockout_features": StockOutRequest.model_config["json_schema_extra"]["example"]["rows"][0]
            }
        ]
    ).model_dump()
    response = client.post("/decision/profit_plan_6h", json=payload)
    print(f"Status Code: {response.status_code}")
    if response.status_code == 200:
        print("Success! Keys:", response.json().keys())
    else:
        print("Error!", response.text)
except Exception as e:
    print("Exception:", e)

