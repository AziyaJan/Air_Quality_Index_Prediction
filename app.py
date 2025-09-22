from flask import Flask, render_template, request
import numpy as np
import pickle
import datetime

app = Flask(__name__)

# Load the trained model
model = pickle.load(open("aqi_model.pkl", "rb"))

# AQI categories
def aqi_category(aqi):
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"

@app.route('/')
def home():
    return render_template('welcome.html')

@app.route('/input')
def input_page():
    return render_template('input.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == "POST":
        # User inputs
        city = request.form['city']
        date_str = request.form['date']
        date = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        pm25 = float(request.form['pm25'])
        pm10 = float(request.form['pm10'])
        no2  = float(request.form['no2'])
        so2  = float(request.form['so2'])
        co   = float(request.form['co'])
        o3   = float(request.form['o3'])

        # Generate features in the same order as training
        day = date.day
        month = date.month
        weekday = date.weekday()
        is_weekday = 1 if weekday < 5 else 0

        # For simplicity, use current pollutants as lag features (can improve if past data available)
        AQI_lag1 = pm25  # approximate
        PM25_lag1 = pm25
        PM10_lag1 = pm10
        NO2_lag1 = no2
        SO2_lag1 = so2
        CO_lag1 = co
        O3_lag1 = o3
        PM25_roll3 = pm25  # approximate

        # Complete feature vector (20 features)
        input_vector = np.array([
            pm25, pm10, no2, so2, co, o3,
            day, month, weekday, is_weekday,
            AQI_lag1, PM25_lag1, PM10_lag1, NO2_lag1,
            SO2_lag1, CO_lag1, O3_lag1, PM25_roll3,
            0, 0  # last two features placeholders if model expects 20
        ]).reshape(1, -1)

        # Predict AQI
        prediction = model.predict(input_vector)[0]
        category = aqi_category(prediction)

        return render_template('output.html', city=city, date=date_str,
                               prediction=round(prediction,2), category=category)

if __name__ == "__main__":
    app.run(debug=True)
