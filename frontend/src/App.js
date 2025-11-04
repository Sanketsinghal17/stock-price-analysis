import React, { useState } from "react";
import { Line } from "react-chartjs-2";
import "chart.js/auto";

function App() {
  const [symbol, setSymbol] = useState("TCS.NS");
  const [chartData, setChartData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [searchResults, setSearchResults] = useState([]);

  const API_KEY = "HJIEKYFCKTNMYVKS";

  // ------------------ Search Company → Symbol ------------------
  const handleSearch = async () => {
    if (!searchTerm.trim()) return;
    setError(null);
    setSearchResults([]);
    setLoading(true);

    try {
      const response = await fetch(
        `https://www.alphavantage.co/query?function=SYMBOL_SEARCH&keywords=${searchTerm}&apikey=${API_KEY}`
      );
      const data = await response.json();

      if (data.bestMatches && data.bestMatches.length > 0) {
        const results = data.bestMatches.map((match) => ({
          symbol: match["1. symbol"],
          name: match["2. name"],
          region: match["4. region"],
        }));
        setSearchResults(results);
      } else {
        setError("⚠️ No matching companies found!");
      }
    } catch (err) {
      console.error(err);
      setError("❌ Error fetching company symbols.");
    }

    setLoading(false);
  };

  // ------------------ Fetch Historical Data ------------------
  const fetchStockData = async () => {
    setLoading(true);
    setError(null);
    setChartData(null);
    setMetrics(null);

    try {
      const response = await fetch(
        `http://127.0.0.1:5000/api/history?symbol=${symbol}`
      );
      const data = await response.json();

      if (data.error) {
        setError(data.error);
        setLoading(false);
        return;
      }

      // ✅ Dynamic ₹ / $ symbol
      const currencySymbol =
        symbol.endsWith(".NS") || symbol.endsWith(".BSE") ? "₹" : "$";

      setChartData({
        labels: data.dates,
        datasets: [
          {
            label: `${symbol} Historical Prices (${currencySymbol})`,
            data: data.prices,
            borderColor: "blue",
            backgroundColor: "rgba(0,123,255,0.2)",
            fill: true,
            tension: 0.3,
          },
        ],
      });
    } catch (err) {
      console.error(err);
      setError("Failed to fetch historical data");
    }

    setLoading(false);
  };

  // ------------------ Fetch Predicted Data ------------------
  const fetchPredictedData = async () => {
  setLoading(true);
  setError(null);
  setChartData(null);

  try {
    // 🔹 Step 1: Fetch predicted prices
    const response = await fetch(
      `http://127.0.0.1:5000/api/predict?symbol=${symbol}`
    );
    const data = await response.json();

    if (data.error) {
      setError(data.error);
      setLoading(false);
      return;
    }

    // 🔹 Step 2: Dynamic ₹ / $ symbol
    const currencySymbol =
      symbol.endsWith(".NS") || symbol.endsWith(".BSE") ? "₹" : "$";

    // 🔹 Step 3: Display prediction chart
    setChartData({
      labels: data.future_dates,
      datasets: [
        {
          label: `${symbol} Predicted Prices (Next 5 Days) (${currencySymbol})`,
          data: data.predicted_next_5_days,
          borderColor: "orange",
          backgroundColor: "rgba(255,165,0,0.2)",
          fill: true,
          tension: 0.3,
        },
      ],
    });

    // 🔹 Step 4: Fetch model accuracy metrics
    const metricsResponse = await fetch("http://127.0.0.1:5000/api/metrics");
    const metricsData = await metricsResponse.json();

    if (metricsData.error) {
      console.warn("Metrics not found:", metricsData.error);
      setMetrics(null);
    } else {
      setMetrics(metricsData);
    }
  } catch (err) {
    console.error(err);
    setError("Failed to fetch prediction data or metrics");
  }

  setLoading(false);
};


  // ------------------ Render ------------------
  return (
    <div style={{ textAlign: "center", padding: "20px", position: "relative" }}>
      <h2>📊 Stock Price Prediction Dashboard</h2>

      {/* 🔍 Search Bar */}
      <div style={{ position: "absolute", top: "20px", right: "30px" }}>
        <input
          type="text"
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          placeholder="Search company name..."
          style={{
            padding: "8px",
            borderRadius: "6px",
            border: "1px solid #ccc",
            width: "220px",
          }}
        />
        <button
          onClick={handleSearch}
          style={{
            padding: "8px 12px",
            marginLeft: "6px",
            backgroundColor: "#28a745",
            color: "white",
            border: "none",
            borderRadius: "6px",
            cursor: "pointer",
          }}
        >
          Search
        </button>

        {searchResults.length > 0 && (
          <div
            style={{
              position: "absolute",
              backgroundColor: "white",
              boxShadow: "0 2px 8px rgba(0,0,0,0.2)",
              borderRadius: "8px",
              marginTop: "5px",
              width: "260px",
              zIndex: 999,
            }}
          >
            {searchResults.map((result, index) => (
              <div
                key={index}
                onClick={() => {
                  setSymbol(result.symbol);
                  setSearchTerm(result.name);
                  setSearchResults([]);
                }}
                style={{
                  padding: "8px",
                  cursor: "pointer",
                  borderBottom: "1px solid #f0f0f0",
                }}
              >
                <strong>{result.symbol}</strong> — {result.name}
                <div style={{ fontSize: "12px", color: "gray" }}>
                  {result.region}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>

      <div style={{ marginTop: "60px" }}>
        <input
          type="text"
          value={symbol}
          onChange={(e) => setSymbol(e.target.value)}
          placeholder="Enter Stock Symbol (e.g. TCS.NS)"
          style={{
            padding: "8px",
            marginRight: "10px",
            width: "220px",
            border: "1px solid #ccc",
            borderRadius: "6px",
          }}
        />

        <button
          onClick={fetchStockData}
          style={{
            marginRight: "10px",
            padding: "8px 14px",
            backgroundColor: "#007bff",
            color: "white",
            border: "none",
            borderRadius: "6px",
            cursor: "pointer",
          }}
        >
          Fetch Data
        </button>

        <button
          onClick={fetchPredictedData}
          style={{
            padding: "8px 14px",
            backgroundColor: "orange",
            color: "white",
            border: "none",
            borderRadius: "6px",
            cursor: "pointer",
          }}
        >
          Predict Prices
        </button>
      </div>

      {loading && <p>Loading...</p>}
      {error && <p style={{ color: "red" }}>{error}</p>}

      {chartData && (
        <div style={{ width: "90%", margin: "30px auto" }}>
          <Line data={chartData} />
        </div>
      )}

      {/* ✅ Display accuracy metrics */}
      {metrics && (
        <div
          style={{
            backgroundColor: "#f8f9fa",
            padding: "20px",
            borderRadius: "10px",
            width: "50%",
            margin: "20px auto",
            boxShadow: "0 3px 8px rgba(0,0,0,0.1)",
            textAlign: "left",
          }}
        >
          <h3 style={{ textAlign: "center", marginBottom: "10px" }}>
            📈 Model Evaluation Metrics ({symbol.endsWith(".NS") || symbol.endsWith(".BSE") ? "₹" : "$"})
          </h3>
          <ul style={{ listStyleType: "none", padding: 0, lineHeight: "1.8" }}>
            <li>
              <b>Mean Absolute Error (MAE):</b> {metrics.MAE}
            </li>
            <li>
              <b>Mean Squared Error (MSE):</b> {metrics.MSE}
            </li>
            <li>
              <b>Root Mean Squared Error (RMSE):</b> {metrics.RMSE}
            </li>
            <li>
              <b>R² Score:</b> {metrics.R2_Score}
            </li>
            <li>
              <b>Mean Absolute Percentage Error (MAPE):</b> {metrics.MAPE}%
            </li>
            <li>
              <b>Accuracy:</b> {metrics.Accuracy}%
            </li>          
            </ul>
        </div>
      )}
    </div>
  );
}

export default App;
