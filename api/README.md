# Chronic Disease Prediction API

This is a FastAPI application that provides a RESTful API for chronic disease prediction.

## Features

- Predict chronic disease risk based on patient data
- Health check endpoint
- Interactive API documentation

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Run the API:
   ```
   python main.py
   ```

   Or with uvicorn:
   ```
   uvicorn main:app --reload
   ```

## API Endpoints

- `GET /` - Root endpoint with API information
- `GET /health` - Health check endpoint
- `POST /predict` - Predict chronic disease risk for a patient

## Usage

### Predict Endpoint

Send a POST request to `/predict` with patient data in JSON format:

```json
{
  "age": 55,
  "bmi": 28.5,
  "glucose": 110,
  "blood_pressure_systolic": 140,
  "blood_pressure_diastolic": 90,
  "cholesterol": 220,
  "smoking_status": "Yes",
  "exercise_hours_per_week": 2
}
```

Response:
```json
{
  "prediction": 1,
  "probability": 0.785,
  "risk_level": "High"
}
```

## Docker

To run the API in a Docker container:

1. Build the image:
   ```
   docker build -t chronic-disease-api .
   ```

2. Run the container:
   ```
   docker run -p 8000:8000 chronic-disease-api
   ```

## Documentation

Once the API is running, visit:
- [http://localhost:8000/docs](http://localhost:8000/docs) for Swagger UI documentation
- [http://localhost:8000/redoc](http://localhost:8000/redoc) for ReDoc documentation