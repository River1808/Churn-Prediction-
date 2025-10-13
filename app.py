from flask import Flask, render_template, request
import pandas as pd
import joblib
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import json
import os

app = Flask(__name__)

# -------------------------------
# Load your trained model
# -------------------------------
model = joblib.load("best_gym_churn_model.pkl")

# -------------------------------
# Google Sheets connection setup
# -------------------------------

# Define the scope (permissions)
scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]

# Load credentials from environment variable (Render)
creds_json = json.loads(os.getenv('GOOGLE_CREDENTIALS'))
creds = ServiceAccountCredentials.from_json_keyfile_dict(creds_json, scope)

# Authorize access
client = gspread.authorize(creds)

# Connect to your Google Sheet
SHEET_ID = "1UvZSukYBxQJCPb_CCMIXHPhaLv3EuGtcl2CVPOgWgdo"
sheet = client.open_by_key(SHEET_ID).sheet1


# -------------------------------
# Flask routes
# -------------------------------
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect user input
        total_classes_attended = float(request.form['total_classes_attended'])
        current_month_classes = float(request.form['current_month_classes'])
        lifetime_months = float(request.form['Lifetime'])

        # Compute averages
        avg_class_frequency_total = total_classes_attended / (lifetime_months * 4)
        avg_class_frequency_current_month = current_month_classes / 4

        data = {
            'gender': request.form['gender'],
            'Near_Location': float(request.form['Near_Location']),
            'Partner': float(request.form['Partner']),
            'Promo_friends': float(request.form['Promo_friends']),
            'Phone': float(request.form['Phone']),
            'Contract_period': float(request.form['Contract_period']),
            'Group_visits': float(request.form['Group_visits']),
            'Age': float(request.form['Age']),
            'Avg_additional_charges_total': float(request.form['Avg_additional_charges_total']),
            'Month_to_end_contract': float(request.form['Month_to_end_contract']),
            'Lifetime': lifetime_months,
            'Avg_class_frequency_total': avg_class_frequency_total,
            'Avg_class_frequency_current_month': avg_class_frequency_current_month
        }

        # Prepare for prediction
        sample = pd.DataFrame([{
            **data,
            'gender': 1 if data['gender'] == 'Female' else 0
        }])

        # Model prediction
        prediction = model.predict(sample)[0]

        confidence = None
        if hasattr(model, "predict_proba"):
            confidence = round(model.predict_proba(sample)[0][int(prediction)] * 100, 2)

        result_text = f"Membership continuation: {'Yes' if prediction == 0 else 'No'}"
        if confidence is not None:
            result_text += f" — Confidence: {confidence}%"

        # Save data to Google Sheet
        sheet.append_row([
            data['gender'],
            data['Near_Location'],
            data['Partner'],
            data['Promo_friends'],
            data['Phone'],
            data['Contract_period'],
            data['Group_visits'],
            data['Age'],
            data['Avg_additional_charges_total'],
            data['Month_to_end_contract'],
            data['Lifetime'],
            data['Avg_class_frequency_total'],
            data['Avg_class_frequency_current_month'],
            'Yes' if prediction == 0 else 'No',
            confidence if confidence is not None else ''
        ])

        return render_template('index.html', prediction_text=result_text, form_data=data)

    except Exception as e:
        return render_template('index.html', prediction_text=f"⚠️ Error: {e}", form_data=request.form)


if __name__ == '__main__':
    app.run(debug=True)
