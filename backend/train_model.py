import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import math

# ---------------- Fetch Stock Data ----------------
data = yf.download("TCS.NS", start="2015-01-01", end="2024-12-31")
data = data[['Close']]

if data.empty:
    raise ValueError("No data found for the given stock symbol!")

print(f"✅ Downloaded {len(data)} records for TCS.NS")

# ---------------- Scale Data ----------------
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

# ---------------- Prepare Training Data ----------------
x_train, y_train = [], []
for i in range(60, len(scaled_data)):
    x_train.append(scaled_data[i-60:i, 0])
    y_train.append(scaled_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

print(f"✅ Training data shape: {x_train.shape}")

# ---------------- Build LSTM Model ----------------
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
model.add(LSTM(50))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')

# ---------------- Train Model ----------------
print("🚀 Training model...")
model.fit(x_train, y_train, epochs=10, batch_size=32, verbose=1)

# ---------------- Save Model ----------------
model.save('stock_model.h5')
print("💾 Model saved as stock_model.h5")

# ---------------- Evaluate Model ----------------
print("📊 Evaluating model performance...")

predicted_train = model.predict(x_train)
predicted_train = scaler.inverse_transform(predicted_train)
actual_train = scaler.inverse_transform(y_train.reshape(-1, 1))

# Calculate Metrics
mae = mean_absolute_error(actual_train, predicted_train)
mse = mean_squared_error(actual_train, predicted_train)
rmse = math.sqrt(mse)
r2 = r2_score(actual_train, predicted_train)
mape = mean_absolute_percentage_error(actual_train, predicted_train) * 100
accuracy = 100 - mape  # Approximate model prediction accuracy

# ---------------- Display Metrics ----------------
print("\n📈 Model Evaluation Metrics:")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"R² Score: {r2:.4f}")
print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
print(f"Model Prediction Accuracy: {accuracy:.2f}%")

# ---------------- Save Metrics ----------------
metrics = {
    "MAE": mae,
    "MSE": mse,
    "RMSE": rmse,
    "R2_Score": r2,
    "MAPE": mape,
    "Accuracy": accuracy
}
metrics_df = pd.DataFrame(metrics, index=[0])
metrics_df.to_csv("model_metrics.csv", index=False)

print("\n📁 Metrics saved to model_metrics.csv")
