from flask import Flask, request, jsonify
from flask_cors import CORS
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import load_model
from datetime import timedelta

app = Flask(__name__)
CORS(app)

# ------------------- ROUTE 1: Fetch Historical Data -------------------
@app.route('/api/history')
def get_history():
    symbol = request.args.get('symbol')
    if not symbol:
        return jsonify({"error": "Symbol is required"}), 400

    try:
        # Try to fetch with given symbol first
        data = yf.download(symbol, period="1y", interval="1d", progress=False)
        if data.empty:
            # Retry with alternate exchange (.NS or .BSE)
            if symbol.endswith(".BSE"):
                alt_symbol = symbol.replace(".BSE", ".NS")
            elif symbol.endswith(".NS"):
                alt_symbol = symbol.replace(".NS", ".BSE")
            else:
                alt_symbol = symbol + ".NS"

            data = yf.download(alt_symbol, period="1y", interval="1d", progress=False)
            if data.empty:
                return jsonify({"error": f"No data found for {symbol} or {alt_symbol}"}), 404
            else:
                symbol = alt_symbol

        data = data.reset_index()
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = ['_'.join(col).strip() if isinstance(col, tuple) else col for col in data.columns]

        date_col = next((col for col in data.columns if "date" in col.lower()), None)
        close_col = next((col for col in data.columns if "close" in col.lower()), None)
        if not date_col or not close_col:
            return jsonify({"error": "Date or Close column not found"}), 500

        dates = data[date_col].astype(str).tolist()
        prices = data[close_col].astype(float).tolist()

        return jsonify({"symbol": symbol, "dates": dates, "prices": prices})

    except Exception as e:
        return jsonify({"error": f"Error fetching history: {str(e)}"}), 500


# ------------------- ROUTE 2: Predict Future Prices -------------------
@app.route('/api/predict')
def predict():
    try:
        symbol = request.args.get('symbol', 'AAPL')
        data = yf.download(symbol, period="6mo", interval="1d")
        if data.empty:
            # Retry with alternate exchange if failed
            if symbol.endswith(".BSE"):
                alt_symbol = symbol.replace(".BSE", ".NS")
            elif symbol.endswith(".NS"):
                alt_symbol = symbol.replace(".NS", ".BSE")
            else:
                alt_symbol = symbol + ".NS"

            data = yf.download(alt_symbol, period="6mo", interval="1d")
            if data.empty:
                return jsonify({"error": f"No data found for {symbol} or {alt_symbol}"}), 404
            else:
                symbol = alt_symbol

        data = data[['Close']]
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)

        last_60_days = scaled_data[-60:]
        X_input = np.array(last_60_days).reshape(1, 60, 1)

        model = load_model("stock_model.h5")

        predictions = []
        current_input = X_input.copy()
        for _ in range(5):
            next_pred = model.predict(current_input, verbose=0)
            next_value = float(np.array(next_pred).flatten()[0])
            predictions.append(next_value)
            current_input = np.append(current_input[:, 1:, :], np.array(next_value).reshape(1, 1, 1), axis=1)

        predicted_prices = scaler.inverse_transform(np.array(predictions).reshape(-1, 1)).flatten()
        last_close = float(data['Close'].iloc[-1])
        diffs = [round(float(p - last_close), 2) for p in predicted_prices]

        last_date = data.index[-1]
        future_dates = [(last_date + timedelta(days=i+1)).strftime('%Y-%m-%d') for i in range(5)]

        return jsonify({
            "symbol": symbol,
            "last_close": last_close,
            "predicted_next_5_days": [round(float(p), 2) for p in predicted_prices],
            "difference_from_last_close": diffs,
            "future_dates": future_dates
        })

    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"}), 500


# ------------------- 🧠 ROUTE 3: Model Accuracy Metrics -------------------
@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    try:
        import os
        import pandas as pd
        file_path = "model_metrics.csv"

        if not os.path.exists(file_path):
            return jsonify({"error": "Metrics file not found. Please train model first."}), 404

        df = pd.read_csv(file_path)
        if df.empty:
            return jsonify({"error": "Metrics file is empty."}), 400

        latest = df.iloc[-1].to_dict()
        metrics = {
            "MAE": round(float(latest.get("MAE", 0)), 3),
            "MSE": round(float(latest.get("MSE", 0)), 3),
            "RMSE": round(float(latest.get("RMSE", 0)), 3),
            "R2_Score": round(float(latest.get("R2_Score", 0)), 3),
            "MAPE": round(float(latest.get("MAPE", 0)), 3),
            "Accuracy": round(float(latest.get("Accuracy", 0)), 2)
        }

        return jsonify(metrics)
    except Exception as e:
        return jsonify({"error": f"Error reading metrics: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)
