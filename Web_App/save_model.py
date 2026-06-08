"""
Script to train the best model and save it with joblib for the Flask web app.
Run this script ONCE after running the full notebook to generate model.pkl and scaler.pkl.
"""
import sys
sys.stdout.reconfigure(encoding='utf-8')

import pandas as pd
import numpy as np
import joblib
import warnings
import os
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

print("Loading data...")
gen_df     = pd.read_csv(os.path.join(BASE_DIR, '../Data/Plant_1_Generation_Data.csv'))
weather_df = pd.read_csv(os.path.join(BASE_DIR, '../Data/Plant_1_Weather_Sensor_Data.csv'))

gen_df['DATE_TIME']     = pd.to_datetime(gen_df['DATE_TIME'])
weather_df['DATE_TIME'] = pd.to_datetime(weather_df['DATE_TIME'])

df = pd.merge(gen_df, weather_df, on='DATE_TIME', how='inner')
df.ffill(inplace=True)

# IQR outlier removal
Q1  = df['AC_POWER'].quantile(0.25)
Q3  = df['AC_POWER'].quantile(0.75)
IQR = Q3 - Q1
df  = df[(df['AC_POWER'] >= Q1 - 1.5*IQR) & (df['AC_POWER'] <= Q3 + 1.5*IQR)]
print(f"Rows after cleaning: {len(df)}")

# Feature engineering
df['HOUR']          = df['DATE_TIME'].dt.hour
df['HOUR_SIN']      = np.sin(2 * np.pi * df['HOUR'] / 24)
df['HOUR_COS']      = np.cos(2 * np.pi * df['HOUR'] / 24)
df['IRRAD_X_TEMP']  = df['IRRADIATION'] * df['MODULE_TEMPERATURE']
df['IRRAD_SQUARED'] = df['IRRADIATION'] ** 2

features = ['IRRADIATION', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE',
            'HOUR_SIN', 'HOUR_COS', 'IRRAD_X_TEMP', 'IRRAD_SQUARED']

X = df[features]
y = df['AC_POWER']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

# Train the best model (Tuned RF - matches notebook results)
print("Training Tuned RF model (this may take a minute)...")
model = RandomForestRegressor(
    n_estimators=300, max_depth=20,
    min_samples_split=2, min_samples_leaf=1,
    max_features='sqrt', random_state=42
)
model.fit(X_train, y_train)

# Save model and feature list
model_info = {
    'model': model,
    'features': features,
    'feature_ranges': {
        'IRRADIATION':           (float(df['IRRADIATION'].min()),           float(df['IRRADIATION'].max())),
        'AMBIENT_TEMPERATURE':   (float(df['AMBIENT_TEMPERATURE'].min()),   float(df['AMBIENT_TEMPERATURE'].max())),
        'MODULE_TEMPERATURE':    (float(df['MODULE_TEMPERATURE'].min()),    float(df['MODULE_TEMPERATURE'].max())),
        'HOUR':                  (0, 23),
    }
}
model_path = os.path.join(BASE_DIR, 'model.pkl')
joblib.dump(model_info, model_path)
print(f"Model saved to {model_path}")

from sklearn.metrics import r2_score
y_pred = model.predict(X_test)
print(f"R² on test set: {r2_score(y_test, y_pred):.4f}")
