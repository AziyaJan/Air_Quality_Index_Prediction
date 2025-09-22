from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load model
model = pickle.load(open("aqi_model.pkl", "rb"))

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
        city = request.form['city']   # currently only Delhi
        date = request.form['date']
        
        # Pollutant values
        pm25 = float(request.form['pm25'])
        pm10 = float(request.form['pm10'])
        no2  = float(request.form['no2'])
        so2  = float(request.form['so2'])
        co   = float(request.form['co'])
        o3   = float(request.form['o3'])

        # Example input (order must match training features!)
        # Here we simplify: just pass pollutants and dummy lag values
        input_data = (pm25, pm10, no2, so2, co, o3, 15, 8, 2, 1, 200, 180, 160, 100, 120, 2.5, 170, 80.0)
        input_array = np.array(input_data).reshape(1, -1)

        prediction = model.predict(input_array)[0]
        category = aqi_category(prediction)

        return render_template('output.html', city=city, date=date,
                               prediction=round(prediction,2), category=category)

if __name__ == "__main__":
    app.run(debug=True)
