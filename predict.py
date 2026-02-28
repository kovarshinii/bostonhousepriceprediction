# predict.py
import argparse
import joblib
import numpy as np

# Load trained model and scaler
model, scaler = joblib.load('boston_model.joblib')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict Boston house price.')

    parser.add_argument('--CRIM', type=float, required=True)
    parser.add_argument('--ZN', type=float, required=True)
    parser.add_argument('--INDUS', type=float, required=True)
    parser.add_argument('--CHAS', type=float, required=True)
    parser.add_argument('--NOX', type=float, required=True)
    parser.add_argument('--RM', type=float, required=True)
    parser.add_argument('--AGE', type=float, required=True)
    parser.add_argument('--DIS', type=float, required=True)
    parser.add_argument('--RAD', type=float, required=True)
    parser.add_argument('--TAX', type=float, required=True)
    parser.add_argument('--PTRATIO', type=float, required=True)
    parser.add_argument('--B', type=float, required=True)
    parser.add_argument('--LSTAT', type=float, required=True)

    args = parser.parse_args()

    features = np.array([[
        args.CRIM, args.ZN, args.INDUS, args.CHAS,
        args.NOX, args.RM, args.AGE, args.DIS,
        args.RAD, args.TAX, args.PTRATIO, args.B,
        args.LSTAT
    ]])

    # Scale features
    features_scaled = scaler.transform(features)

    # Predict
    predicted_price = model.predict(features_scaled)[0]

    print(f"Predicted House Price: ${predicted_price * 1000:.2f}")
    