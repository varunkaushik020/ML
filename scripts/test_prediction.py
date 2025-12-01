import pandas as pd
import numpy as np
import joblib


def load_model():
    model = joblib.load('models/chronic_disease_model_randomforest.pkl')
    scaler = joblib.load('models/scaler.pkl')
    return model, scaler


def make_prediction(model, scaler, features):
    features_array = np.array(features).reshape(1, -1)
    prediction = model.predict(features_array)
    probability = model.predict_proba(features_array)
    return prediction[0], probability[0]


def main():
    model, scaler = load_model()

    print("Model loaded successfully!")

    sample_data = [6, 148, 72, 35, 0, 33.6, 0.627, 50]

    prediction, probabilities = make_prediction(model, scaler, sample_data)

    print(f"Sample data: {sample_data}")
    print(f"Prediction: {prediction}")
    print(f"Probability of no diabetes: {probabilities[0]:.4f}")
    print(f"Probability of diabetes: {probabilities[1]:.4f}")


if __name__ == "__main__":
    main()
