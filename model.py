import numpy as np
import pandas as pd
import pickle
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import mean_squared_error

# Load your dataset (replace with correct path)
aqi_data = pd.read_csv("aqi_dataset.csv")

# Convert Date column
aqi_data['Date'] = pd.to_datetime(aqi_data['Date'])

# Feature extraction
aqi_data['day'] = aqi_data['Date'].dt.day
aqi_data['month'] = aqi_data['Date'].dt.month
aqi_data['weekday'] = aqi_data['Date'].dt.weekday
aqi_data['is_weekday'] = aqi_data['weekday'].apply(lambda x: 1 if x < 5 else 0)

# Target : Next day AQI
aqi_data['AQI_target'] = aqi_data['AQI'].shift(-1)

# Lag features
aqi_data['AQI_lag1'] = aqi_data['AQI'].shift(1)
aqi_data['PM2.5'] = aqi_data['PM2.5'].shift(1)
aqi_data['PM10_lag1'] = aqi_data['PM10'].shift(1)
aqi_data['NO2_lag1'] = aqi_data['NO2'].shift(1)
aqi_data['SO2_lag1'] = aqi_data['SO2'].shift(1)
aqi_data['CO_lag1'] = aqi_data['CO'].shift(1)
aqi_data['O3_lag1'] = aqi_data['O3'].shift(1)

# Rolling mean
aqi_data['PM2.5_roll24'] = aqi_data['PM2.5'].rolling(24).mean()

aqi_data = aqi_data.dropna()

X = aqi_data.drop(columns=['City','Date','AQI'], axis=1)
Y = aqi_data['AQI']

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, shuffle=False)

# Train model
xgb_model = XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=2
)

xgb_model.fit(X_train, Y_train)

# Save model
pickle.dump(xgb_model, open("aqi_model.pkl", "wb"))

print("✅ Model trained and saved as aqi_model.pkl")
