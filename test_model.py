import pandas as pd
import joblib

# Load the trained model
model = joblib.load("best_gym_churn_model.pkl")

# Create a sample likely to continue membership
sample = pd.DataFrame([{
    'gender': 1,  # Female = 1, Male = 0
    'Near_Location': 1,
    'Partner': 1,
    'Promo_friends': 1,
    'Phone': 1,
    'Contract_period': 12,
    'Group_visits': 1,
    'Age': 30,
    'Avg_additional_charges_total': 50,
    'Month_to_end_contract': 6,
    'Lifetime': 36,  # months since first joined
    'Avg_class_frequency_total': 5,  # high activity
    'Avg_class_frequency_current_month': 5  # high activity
}])

# Make prediction
prediction = model.predict(sample)[0]

# Confidence score
confidence = None
if hasattr(model, "predict_proba"):
    confidence = round(model.predict_proba(sample)[0][int(prediction)] * 100, 2)

# Correct Yes/No logic: 0 = continue (Yes), 1 = churn (No)
print(f"✅ Prediction: {'Yes' if prediction == 0 else 'No'}")
if confidence is not None:
    print(f"🎯 Confidence: {confidence}%")
