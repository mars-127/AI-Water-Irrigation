#CALCULATE DAILY AVERAGES
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


# Load the data
df = pd.read_csv('readings.csv')


# Calculate daily averages
daily_df = df.groupby('Day').agg({
    'Temperature': 'mean',
    'Humidity': 'mean'
}).reset_index()


print("Daily Averages:")
print(daily_df.head(10))


##CREATE LAGGED FEATURES FOR PREDICTION
# Define how many previous days to use for prediction
n_lags = 5  # Using past 5 days to predict next day


# Create lagged features for each parameter
features_df = pd.DataFrame()


for lag in range(1, n_lags + 1):
    features_df[f'Temperature_lag_{lag}'] = daily_df['Temperature'].shift(lag)
    features_df[f'Humidity_lag_{lag}'] = daily_df['Humidity'].shift(lag)


# Add the target (next day's values)
features_df['Target_Temperature'] = daily_df['Temperature']
features_df['Target_Humidity'] = daily_df['Humidity']


# Remove rows with missing values (first n_lags days)
model_data = features_df.dropna().reset_index(drop=True)


print("\nModel Training Data (first 5 rows):")
print(model_data.head())


# TRAIN TEST SPLIT AND MODEL TRAINING
# Split features and targets
X = model_data.drop(['Target_Temperature', 'Target_Humidity'], axis=1)
y_temp = model_data['Target_Temperature']
y_humidity = model_data['Target_Humidity']


# Split into training and testing sets (last 7 days for testing)
test_size = 7
X_train, X_test = X.iloc[:-test_size], X.iloc[-test_size:]
y_temp_train, y_temp_test = y_temp.iloc[:-test_size], y_temp.iloc[-test_size:]
y_humidity_train, y_humidity_test = y_humidity.iloc[:-test_size], y_humidity.iloc[-test_size:]


# Train model for Temperature
temp_model = RandomForestRegressor(n_estimators=100, random_state=42)
temp_model.fit(X_train, y_temp_train)


# Train model for Humidity
humidity_model = RandomForestRegressor(n_estimators=100, random_state=42)
humidity_model.fit(X_train, y_humidity_train)


# Make predictions
temp_pred = temp_model.predict(X_test)
humidity_pred = humidity_model.predict(X_test)


# Evaluate models
temp_mae = mean_absolute_error(y_temp_test, temp_pred)
humidity_mae = mean_absolute_error(y_humidity_test, humidity_pred)


print(f"Temperature Model MAE: {temp_mae:.2f}°C")
print(f"Humidity Model MAE: {humidity_mae:.2f}%")


#PREDICT NEXT DAY'S WEATHER
# Get the last n_lags days of data
last_n_days = daily_df.tail(n_lags)


# Prepare features for prediction
prediction_features = []
for lag in range(1, n_lags + 1):
    # Get data from lag days ago (reverse order)
    day_data = last_n_days.iloc[-lag]
    prediction_features.extend([day_data['Temperature'], day_data['Humidity']])


# Create feature names to match training data
feature_names = []
for lag in range(1, n_lags + 1):
    feature_names.extend([f'Temperature_lag_{lag}', f'Humidity_lag_{lag}'])


# Create prediction dataframe
X_next_day = pd.DataFrame([prediction_features], columns=feature_names)


# Make predictions
next_day_temp = temp_model.predict(X_next_day)[0]
next_day_humidity = humidity_model.predict(X_next_day)[0]


print(f"\nPrediction for Next Day (Day {daily_df['Day'].max() + 1}):")
print(f"Temperature: {next_day_temp:.1f}°C")
print(f"Humidity: {next_day_humidity:.1f}%")
