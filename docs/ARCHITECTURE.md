# Clean Architecture in Smart Fleet Intelligence

This project strictly adheres to the principles of **Clean Architecture** to ensure that business logic is completely isolated from the web framework (FastAPI) and external data formats (GeoJSON, Pandas, etc.).

## System Layers

### 1. The Controller Layer (`main1.py`)
This is the entry point for the FastAPI application.
- **Responsibility:** Define HTTP routes, receive incoming requests, and return responses.
- **Rules:** Must not contain any business logic, ML inference code, or data processing. It merely delegates the parsed request from the schemas to the Service Layer.

### 2. The Domain / Schema Layer (`api_schemas.py`)
This layer defines the shape of the data using Pydantic `BaseModel` classes.
- **Responsibility:** Request validation, type enforcement, and default values.
- **Rules:** The schemas are the "vocabulary" of the application. The `BaseMLInput` class includes helper methods like `to_df()` to centralize the conversion of JSON data into Pandas DataFrames, preventing repetitive data wrangling across the application.

### 3. The Infrastructure Layer (`ml_core.py`)
This layer handles the physical assets of the application (Models and Spatial Data).
- **Responsibility:** Loading `.pkl` and `.joblib` files, processing the `taxi_zones.shp` shapefile, and maintaining memory efficiency.
- **Rules:** Designed as a Singleton (`MLModelManager`). This guarantees that heavy model files are only loaded into memory once during the application lifecycle. It also provides a global `get_zone_geojson()` method for spatial queries.

### 4. The Service / Business Logic Layer (`api_services.py`)
This is the core of the application where all the predictive math and heuristics happen.
- **Responsibility:** Execute Machine Learning predictions and calculate the Fleet Repositioning logic.
- **Rules:** Contains `PredictionService` and `DecisionEngineService`. These classes take strictly validated data from the schemas, process them using models provided by the `ml_core`, and return clean dictionaries to the controllers. 

## Model Dependency Pipeline
A unique aspect of this architecture is how it handles dependent ML models.
- The **Revenue** and **Stockout** models inherently require the output of the **Demand** model to function correctly.
- Instead of forcing the frontend to make two API calls, `api_services.py` intercepts the request, runs the Demand model in the background, guarantees the value is non-negative, and injects it seamlessly into the Revenue/Stockout dataframe right before prediction.
- **Result:** Faster API response times, fewer frontend network requests, and absolute data integrity.
