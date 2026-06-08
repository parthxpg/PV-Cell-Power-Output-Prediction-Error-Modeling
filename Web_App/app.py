"""
Flask Web App for PV Cell Power Prediction
Step 7 (Optional) from the project brief.
"""
import sys
import os
import numpy as np
import joblib

from flask import Flask, request, render_template, jsonify

app = Flask(__name__)

# Load the saved model once at startup
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.pkl')
model_info = joblib.load(MODEL_PATH)
model      = model_info['model']
features   = model_info['features']


def build_feature_vector(irradiation, ambient_temp, module_temp, hour):
    """Build the feature vector matching the training pipeline."""
    hour_sin      = np.sin(2 * np.pi * hour / 24)
    hour_cos      = np.cos(2 * np.pi * hour / 24)
    irrad_x_temp  = irradiation * module_temp
    irrad_squared = irradiation ** 2

    return [[irradiation, ambient_temp, module_temp,
             hour_sin, hour_cos, irrad_x_temp, irrad_squared]]


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        irradiation  = float(request.form['irradiation'])
        ambient_temp = float(request.form['ambient_temp'])
        module_temp  = float(request.form['module_temp'])
        hour         = float(request.form['hour'])

        X = build_feature_vector(irradiation, ambient_temp, module_temp, hour)
        prediction = model.predict(X)[0]
        prediction = max(0.0, float(prediction))   # power cannot be negative

        return jsonify({
            'success': True,
            'prediction': round(prediction, 2),
            'unit': 'kW'
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400


if __name__ == '__main__':
    print("🌞 PV Power Predictor running at http://127.0.0.1:5000")
    app.run(debug=True, host='0.0.0.0', port=5000)
