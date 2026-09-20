import joblib
import numpy as np

# Load model and scaler
model = joblib.load("model.pkl")
scaler = joblib.load("scaler.pkl")

# Sample input: [income, credit_score, loan_amount, employment_years]
sample_applicant = np.array([[75000, 720, 15000, 5]])

# Preprocess & Predict
sample_scaled = scaler.transform(sample_applicant)
prediction = model.predict(sample_scaled)

status = "APPROVED" if prediction[0] == 1 else "REJECTED"
print(f"Loan Status Prediction: {status}")