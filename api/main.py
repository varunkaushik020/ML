from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import joblib
from typing import List, Optional
import os

app = FastAPI(
    title="Chronic Disease Prediction API",
    description="API for predicting chronic disease risk based on patient data",
    version="1.0.0"
)

class DiabetesData(BaseModel):
    pregnancies: int
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree_function: float
    age: int


class KidneyData(BaseModel):
    age: float
    bp: float  
    sg: float  
    al: float  
    su: float  
    bgr: float 
    bu: float  
    sc: float 
    sod: float  
    pot: float  
    hemo: float 


class HeartData(BaseModel):
    age: float
    sex: int  
    cp: int  
    trestbps: float  
    chol: float 
    fbs: int  
    restecg: int  
    thalach: float
    exang: int 
    oldpeak: float 
    slope: int 
    ca: int 
    thal: int 


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    disease_type: str
    risk_level: str


def simulate_prediction(disease_type: str) -> tuple:
    """Simulate model prediction for demo purposes"""
    import random
    prediction = random.choice([0, 1])
    probability = random.uniform(0.1, 0.9)
    return prediction, probability


@app.post("/predict/diabetes", response_model=PredictionResponse)
async def predict_diabetes(patient: DiabetesData):

    try:
  
        prediction, probability = simulate_prediction("diabetes")

        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"

        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            disease_type="diabetes",
            risk_level=risk_level
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/kidney", response_model=PredictionResponse)
async def predict_kidney(patient: KidneyData):

    try:

        prediction, probability = simulate_prediction("kidney")

        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"

        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            disease_type="kidney disease",
            risk_level=risk_level
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/heart", response_model=PredictionResponse)
async def predict_heart(patient: HeartData):

    try:
      
        prediction, probability = simulate_prediction("heart")

     
        if probability < 0.3:
            risk_level = "Low"
        elif probability < 0.7:
            risk_level = "Medium"
        else:
            risk_level = "High"

        return PredictionResponse(
            prediction=int(prediction),
            probability=float(probability),
            disease_type="heart disease",
            risk_level=risk_level
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy"}


@app.get("/")
async def root():
    """
    Root endpoint with API information
    """
    return {
        "message": "Chronic Disease Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "diabetes_prediction": "/predict/diabetes",
            "kidney_prediction": "/predict/kidney",
            "heart_prediction": "/predict/heart",
            "health_check": "/health",
            "docs": "/docs"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
