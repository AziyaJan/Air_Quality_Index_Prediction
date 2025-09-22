import numpy as np
import pandas as pd
import pickle
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# ------------------------------
# 1. Load dataset
# ------------------------------
aqi_data = pd.read_csv("C:/Users/janaz/OneDrive/Desktop/AQI Project/Delhi_AQI_Dataset.csv")  # Replace with your dataset path

# Parse Date safely
aqi_data['Date'] = pd.to_datetime(aqi_data['Date'], errors='coerce')
aqi_data = aqi_data.dropna(subset=['Date'])

# ------------------------------
# 2. Feature Engineering
# ------------------------------
aqi_data['day'] = aqi_data['Date'].dt.day
aqi_data['month'] = aqi_data['Date'].dt.month
aqi_data['weekday'] = aqi_data['Date'].dt.weekday
aqi_data['is_weekday'] = aqi_data['weekday'].apply(lambda x: 1 if x < 5 else 0)

# Target: Next day's AQI
aqi_data['AQI_target'] = aqi_data['AQI'].shift(-1)

# Lag features
aqi_data['AQI_lag1'] = aqi_data['AQI'].shift(1)
aqi_data['PM2.5_lag1'] = aqi_data['PM2.5'].shift(1)
aqi_data['PM10_lag1'] = aqi_data['PM10'].shift(1)
aqi_data['NO2_lag1'] = aqi_data['NO2'].shift(1)
aqi_data['SO2_lag1'] = aqi_data['SO2'].shift(1)
aqi_data['CO_lag1'] = aqi_data['CO'].shift(1)
aqi_data['O3_lag1'] = aqi_data['O3'].shift(1)

# Rolling mean (3-hour window to avoid empty dataset)
aqi_data['PM2.5_roll3'] = aqi_data['PM2.5'].rolling(3).mean()

# Drop rows with NaNs in required features
required_features = ['AQI_lag1','PM2.5_lag1','PM10_lag1','NO2_lag1','SO2_lag1',
                     'CO_lag1','O3_lag1','PM2.5_roll3','AQI_target']
aqi_data = aqi_data.dropna(subset=required_features)

print(f"Rows after feature engineering: {len(aqi_data)}")

# ------------------------------
# 3. Define features and target
# ------------------------------
X = aqi_data.drop(columns=['City','Date','AQI','AQI_target'])
Y = aqi_data['AQI_target']

print("Feature columns:", list(X.columns))
print("Number of features:", X.shape[1])

# ------------------------------
# 4. Split dataset
# ------------------------------
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

# ------------------------------
# 5. Train model
# ------------------------------
xgb_model = XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=2
)
xgb_model.fit(X_train, Y_train)

# ------------------------------
# 6. Evaluate
# ------------------------------
train_pred = xgb_model.predict(X_train)
test_pred = xgb_model.predict(X_test)

print(f"R2 (train): {r2_score(Y_train, train_pred):.3f}, RMSE (train): {np.sqrt(mean_squared_error(Y_train, train_pred)):.3f}")
print(f"R2 (test): {r2_score(Y_test, test_pred):.3f}, RMSE (test): {np.sqrt(mean_squared_error(Y_test, test_pred)):.3f}")

# ------------------------------
# 7. Save model
# ------------------------------
pickle.dump(xgb_model, open("aqi_model.pkl", "wb"))
print("✅ Model saved as aqi_model.pkl")
