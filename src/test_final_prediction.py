
import pandas as pd
import joblib

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"

model_path = MODEL_DIR / "autovalue_final_model.joblib"

print("=" * 60)
print("       AUTOVALUE AI - PREDICTION ENGINE TEST")
print("=" * 60)

# Load saved model
model = joblib.load(model_path)

# Example vehicle input
car = pd.DataFrame([{
    "year": 2018,
    "km_driven": 45000,
    "fuel": "Petrol",
    "seller_type": "Individual",
    "transmission": "Manual",
    "owner": "First Owner",
    "mileage_value": 18.9,
    "engine_cc": 1197,
    "max_power_bhp": 81.8,
    "torque_nm": 113.0,
    "seats": 5,
    "brand": "Maruti"
}])

print("\nInput vehicle details:")
print(car.to_string(index=False))

# Predict price
predicted_price = model.predict(car)[0]

print("\n" + "-" * 60)
print("PREDICTION RESULT")
print("-" * 60)

print(f"Estimated resale price: Rs. {predicted_price:,.2f}")

if predicted_price > 0:
    print("\nPrediction test: PASSED")
else:
    print("\nPrediction test: FAILED")

print("\nPREDICTION ENGINE TEST COMPLETED")
