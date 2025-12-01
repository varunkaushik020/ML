import requests
import json
import time


def test_api_endpoints():

    base_url = "http://localhost:8000"

    print("Testing root endpoint...")
    try:
        response = requests.get(base_url)
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

    print("\nTesting health endpoint...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {response.json()}")
    except Exception as e:
        print(f"Error: {e}")

    print("\nTesting prediction endpoint...")
    patient_data = {
        "age": 55,
        "bmi": 28.5,
        "glucose": 110,
        "blood_pressure_systolic": 140,
        "blood_pressure_diastolic": 90,
        "cholesterol": 220,
        "smoking_status": "Yes",
        "exercise_hours_per_week": 2
    }

    try:
        response = requests.post(
            f"{base_url}/predict",
            json=patient_data,
            headers={"Content-Type": "application/json"}
        )
        print(f"Status Code: {response.status_code}")
        if response.status_code == 200:
            print(f"Response: {response.json()}")
        else:
            print(f"Error Response: {response.text}")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    print("Testing Chronic Disease Prediction API")
    print("=" * 40)
    test_api_endpoints()
