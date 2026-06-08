# ☀️ PV Cell Power Output Prediction & Error Modeling

A complete predictive modeling pipeline and real-time forecasting web interface for Photovoltaic (PV) solar power output. This project trains machine learning models on solar generation and weather datasets, validates results against an analytical single-diode physics model, and serves predictions via an interactive web dashboard.

---

## 📁 Repository Structure

```text
├── Data/
│   ├── Plant_1_Generation_Data.csv      # Solar power generation readings
│   └── Plant_1_Weather_Sensor_Data.csv  # Weather sensor readings (irradiance, temp)
├── Graphs/
│   ├── r2_all_models.png               # R² metrics comparison graph
│   ├── actual_vs_best.png              # Actual vs Best predictions comparison
│   ├── error_dist_top3.png             # Error distribution of top models
│   └── pvlib_validation.png            # ML vs pvlib single-diode validation plot
├── Physics_Validation/
│   └── pvlib_validation.py             # Single-diode physics validation script
├── Web_App/
│   ├── app.py                          # Flask backend server
│   ├── save_model.py                   # Script to train and serialize the best model
│   └── templates/
│       └── index.html                  # Premium HTML/CSS dashboard UI
├── PV_Cell_Ceiling_Test.ipynb          # Comprehensive Jupyter Notebook pipeline
├── requirements.txt                    # Project dependencies
└── .gitignore                          # Excludes large models, data and checkpoints
```

---

## 🚀 Setup & Execution

### 1. Install Dependencies
Make sure you have Python 3.8+ installed. Run:
```bash
pip install -r requirements.txt
```

### 2. Prepare the Model
Since the serialized model file (`model.pkl` ~70MB) is excluded from version control to keep the repository lightweight, you can generate it locally in a few seconds:
```bash
cd Web_App
python save_model.py
```
This trains the best-performing model (Tuned Random Forest, $R^2 \approx 0.9865$) and serializes it to `model.pkl`.

### 3. Launch the Web Application
Start the interactive dashboard locally:
```bash
python app.py
```
Open your browser and navigate to **[http://127.0.0.1:5000](http://127.0.0.1:5000)**.

---

## 🔬 Physics-Based Validation (`pvlib` vs. ML)
The project contains an analytical single-diode model simulation using `pvlib` (replicating the MATLAB Simscape / PV System Toolbox physics equations).

To run the standalone comparison script:
```bash
python Physics_Validation/pvlib_validation.py
```

### Key Validation Findings:
*   **ML Model MAE:** `4.35 kW`
*   **Physics Model MAE:** `117.14 kW`
*   **Conclusion:** The machine learning model outperforms the pure physical model by **over 25x**. This indicates that the ML model successfully learns secondary real-world factors (such as micro-shading, soiling, and sensor drift) which standard single-diode formulas do not capture without tedious manual calibration.

---

## 📊 Model Performance Comparison

The pipeline evaluates nine different model configurations (base, tuned, stacked, and residual error-corrected):

| Model Configuration | RMSE (kW) | MAE (kW) | $R^2$ Score |
| :--- | :---: | :---: | :---: |
| **Tuned Random Forest (Best)** | **101.44** | **31.25** | **0.9865** |
| Base Random Forest | 104.38 | 32.55 | 0.9857 |
| XGBoost Regression | 104.83 | 34.02 | 0.9856 |
| Stacking Ensemble (Weighted) | 104.12 | 32.22 | 0.9858 |
| Residual Correction (RF ➔ XGB) | 104.34 | 32.48 | 0.9857 |

*The convergence of all models around $R^2 \approx 0.986$ indicates an **irreducible error ceiling** caused by stochastic environmental noise (e.g. rapid cloud movements) not captured by 15-minute sensor intervals.*

---

## 🌟 Web Dashboard Features
*   **Dark Glassmorphism Interface:** Designed for premium aesthetic feel.
*   **Real-time AJAX Forecasts:** Update parameters (irradiation, temperature, time of day) and get prediction output instantly without page reloads.
*   **Educational Insights:** Explains the physical Photovoltaic Effect and outlines **8 actionable steps** for increasing PV efficiency (e.g. angle adjustments, micro-inverter optimization, cleaning, and ML-based maintenance).
