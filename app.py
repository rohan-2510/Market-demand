from flask import Flask, render_template, request
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load model and scaler
model = joblib.load("crop_demand_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load training column names
with open("columns.txt", "r", encoding="utf-8") as f:
    feature_columns = f.read().splitlines()

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get numeric input
        input_data = {
            "Temperature (°C)": float(request.form["temperature"]),
            "Rainfall": float(request.form["rainfall"]),
            "Supply Volume": float(request.form["supply"]),
            "Demand Volume": float(request.form["demand"]),
            "Transportation Cost (₹/ton)": float(request.form["transport_cost"]),
            "Fertilizer Cost": float(request.form["fertilizer_cost"]),
            "Pest Level": float(request.form["pest_level"]),
        }

        # One-hot encode dropdowns
        input_data[f"Crop Type_{request.form['crop_type']}"] = 1
        input_data[f"Season_{request.form['season']}"] = 1
        input_data[f"City_{request.form['city']}"] = 1
        input_data[f"State_{request.form['state']}"] = 1

        # Fill in all missing features as 0
        for col in feature_columns:
            if col not in input_data:
                input_data[col] = 0

        # Create DataFrame and reorder columns
        df_input = pd.DataFrame([input_data])[feature_columns]
        df_scaled = scaler.transform(df_input)
        prediction = model.predict(df_scaled)[0]

        return render_template("index.html", prediction=f"Predicted Demand: {prediction}")

    except Exception as e:
        return render_template("index.html", prediction=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=True)