# Smart Fleet Intelligence API 🚖🚕

![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=for-the-badge&logo=fastapi)
![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![LightGBM](https://img.shields.io/badge/LightGBM-4A4A4A?style=for-the-badge&logo=scikit-learn)
![Pandas](https://img.shields.io/badge/pandas-%23150458.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Dask](https://img.shields.io/badge/Dask-1E1C1A?style=for-the-badge&logo=dask&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)

The **Smart Fleet Intelligence API** is a machine-learning-powered backend designed for advanced fleet management, dynamic taxi dispatching, and urban mobility optimization in New York City.

By leveraging highly accurate gradient boosting models (LightGBM) trained on the NYC Taxi Dataset, this platform provides real-time predictive insights to maximize driver revenue, minimize idle time, and eliminate geographical vehicle stock-outs.

## 🚀 Features

- **Demand Prediction (6-hour & 15-minute intervals):** Anticipate precisely how many rides will be requested in a specific NYC Taxi Zone.
- **Revenue Forecasting:** Dynamically predict the 50th (P50) and 90th (P90) percentiles of revenue generated per zone.
- **Stockout Risk Analysis:** Predict the probability of a zone running out of available vehicles due to sudden demand spikes.
- **ETA (Estimated Time of Arrival):** Calculate highly accurate travel times between any two NYC zones considering rush hours and weather conditions.
- **Decision Engine (Profit Planner):** A complex strategic engine that analyzes all zones simultaneously to generate a multi-fleet repositioning plan, telling drivers exactly where to move to maximize overall network profit.
- **Geospatial Integration:** All endpoints natively return GeoJSON `shp_file` coordinates for seamless frontend map rendering.

## 🏗️ Architecture

The codebase strictly adheres to **Clean Architecture** principles to ensure maintainability, scalability, and robust error handling.

- `main1.py`: The Controller Layer (FastAPI routing).
- `api_schemas.py`: The Domain Layer (Pydantic data validation and payload definition).
- `ml_core.py`: The Infrastructure Layer (ML model loading, Singletons, and Shapefile processing).
- `api_services.py`: The Business Logic Layer (Inference pipelines and Decision Engine math).

For a deeper dive into the structure, see [ARCHITECTURE.md](docs/ARCHITECTURE.md).

## 🛠️ Installation & Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/Mo0Hanned/AI-Driven-Ride-Optimization.git
   cd smart-fleet-intelligence
   ```

2. **Create a virtual environment and install dependencies**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Ensure Model Files are Present**
   The application requires pre-trained `.pkl` and `.joblib` files inside the `model/` directory, and shapefiles inside `taxi_zones/`.

4. **Run the API Server**
   ```bash
   uvicorn main1:app --host 0.0.0.0 --port 8000 --reload
   ```

## 📚 API Documentation

Once the server is running, the interactive OpenAPI documentation is available at:
- **Swagger UI**: `http://localhost:8000/docs`
- **ReDoc**: `http://localhost:8000/redoc`

For detailed request/response examples, see the [API Reference](docs/API_REFERENCE.md).

## 🧪 Testing

A complete test suite is provided to validate the inference pipelines and API responses:
```bash
python test_endpoints.py
```

## ⚙️ Model Pipelining

The system uses advanced **Model Pipelining**. Certain models (like Revenue and Stockout) intrinsically depend on the Demand model's output. The `api_services.py` layer automatically handles this by running the Demand model in the background, injecting the generated features into the dependent models, and returning the final output seamlessly. This prevents negative demand edge-cases and reduces frontend complexity.
