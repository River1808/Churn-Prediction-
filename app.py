from flask import Flask, render_template, request
import pandas as pd
import joblib

app = Flask(__name__)

# Load your trained model
model = joblib.load("best_gym_churn_model.pkl")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Collect user input from the form
        total_classes_attended = float(request.form['total_classes_attended'])
        current_month_classes = float(request.form['current_month_classes'])
        lifetime_months = float(request.form['Lifetime'])  

        # Automatically compute averages
        avg_class_frequency_total = total_classes_attended / (lifetime_months*4)
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

        # Encode gender
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

        return render_template('index.html', prediction_text=result_text, form_data=data)

    except Exception as e:
        return render_template('index.html', prediction_text=f"⚠️ Error: {e}", form_data=request.form)

if __name__ == '__main__':
    app.run(debug=True)
