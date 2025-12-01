import requests
import json


def test_api():
  
    base_url = "http://localhost:8001"

    print("Testing Diabetes Prediction API...")
    diabetes_data = {
        "pregnancies": 2,
        "glucose": 120,
        "blood_pressure": 70,
        "skin_thickness": 25,
        "insulin": 100,
        "bmi": 28.5,
        "diabetes_pedigree_function": 0.5,
        "age": 45
    }

    try:
        response = requests.post(
            f"{base_url}/predict/diabetes",
            json=diabetes_data,
            headers={"Content-Type": "application/json"}
        )
        print(f"Diabetes Prediction Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(
                f"Diabetes Prediction Result: {json.dumps(result, indent=2)}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error testing diabetes API: {e}")

    print("\nTesting Kidney Disease Prediction API...")
    kidney_data = {
        "age": 55,
        "bp": 80,
        "sg": 1.02,
        "al": 0,
        "su": 0,
        "bgr": 120,
        "bu": 40,
        "sc": 1.2,
        "sod": 135,
        "pot": 4.5,
        "hemo": 15.0
    }

    try:
        response = requests.post(
            f"{base_url}/predict/kidney",
            json=kidney_data,
            headers={"Content-Type": "application/json"}
        )
        print(f"Kidney Disease Prediction Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(
                f"Kidney Disease Prediction Result: {json.dumps(result, indent=2)}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error testing kidney disease API: {e}")

    print("\nTesting Heart Disease Prediction API...")
    heart_data = {
        "age": 63,
        "sex": 1,
        "cp": 3,
        "trestbps": 145,
        "chol": 233,
        "fbs": 1,
        "restecg": 0,
        "thalach": 150,
        "exang": 0,
        "oldpeak": 2.3,
        "slope": 0,
        "ca": 0,
        "thal": 1
    }

    try:
        response = requests.post(
            f"{base_url}/predict/heart",
            json=heart_data,
            headers={"Content-Type": "application/json"}
        )
        print(f"Heart Disease Prediction Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(
                f"Heart Disease Prediction Result: {json.dumps(result, indent=2)}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error testing heart disease API: {e}")

    print("\nTesting Health Check API...")
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Health Check Status Code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            print(f"Health Check Result: {json.dumps(result, indent=2)}")
        else:
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"Error testing health check API: {e}")


if __name__ == "__main__":
    print("Testing Chronic Disease Prediction API")
    print("=" * 40)
    test_api()
    print("\nAPI testing completed!")
