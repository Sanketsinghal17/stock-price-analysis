import React, { useState, useEffect, useMemo } from "react";
import { Line } from "react-chartjs-2";
import "chart.js/auto";

// --- Custom Component for Metric Cards ---
const MetricCard = ({ title, value, unit, color, tooltipText }) => (
  <div
    style={{
      flex: 1,
      minWidth: "150px",
      padding: "15px",
      margin: "0 10px",
      borderRadius: "10px",
      backgroundColor: "#ffffff",
      borderLeft: `5px solid ${color}`,
      boxShadow: "0 4px 12px rgba(0, 0, 0, 0.05)",
      cursor: "help",
      position: "relative",
    }}
    title={tooltipText} // Tooltip for explanation
  >
    <div style={{ fontSize: "12px", color: "#6c757d", marginBottom: "5px" }}>
      {title}
    </div>
    <div style={{ fontSize: "24px", fontWeight: "bold", color }}>
      {value}
      {unit}
    </div>
  </div>
);

// --- Main App Component ---
function App() {
  const [symbol, setSymbol] = useState("TCS.NS");
  const [chartData, setChartData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [searchTerm, setSearchTerm] = useState("");
  const [searchResults, setSearchResults] = useState([]);
  const [lastClosePrice, setLastClosePrice] = useState(null);
  const [predictedPrices, setPredictedPrices] = useState(null); // New state for prediction table

  const API_KEY = "HJIEKYFCKTNMYVKS";

  const currencySymbol = useMemo(
    () => (symbol.endsWith(".NS") || symbol.endsWith(".BSE") ? "₹" : "$"),
    [symbol]
  );

  // Helper to combine historical and predicted data for one chart
  const combinedChartData = useMemo(() => {
    if (!chartData || !predictedPrices) return null;

    const historicalDates = chartData.labels;
    const historicalPrices = chartData.datasets[0].data;

    const futureDates = predictedPrices.future_dates;
    const predictedValues = predictedPrices.predicted_next_5_days;

    const combinedLabels = [...historicalDates, ...futureDates];
    const combinedPrices = [...historicalPrices];

    // Pad the predicted data start with 'null' to align only the forecast line
    const forecastStart = new Array(historicalPrices.length - 1).fill(null);
    const forecastLine = [...forecastStart, historicalPrices[historicalPrices.length - 1], ...predictedValues];


    return {
      labels: combinedLabels,
      datasets: [
        {
          label: `${symbol} Historical Prices`,
          data: combinedPrices,
          borderColor: "#007bff",
          backgroundColor: "rgba(0, 123, 255, 0.05)",
          pointRadius: 1.5,
          tension: 0.2,
          fill: false,
          yAxisID: 'y',
        },
        {
          label: '5-Day Forecast',
          data: forecastLine,
          borderColor: "#28a745",
          borderDash: [5, 5],
          pointRadius: 4,
          pointBackgroundColor: "#28a745",
          tension: 0.2,
          fill: false,
          yAxisID: 'y',
        }
      ],
    };
  }, [chartData, predictedPrices, symbol]);

  // Chart options for better look
  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        position: 'top',
        labels: {
          font: {
            size: 14,
            family: 'Arial, sans-serif'
          }
        }
      },
      title: {
        display: true,
        text: `${symbol} Price History & 5-Day Prediction`,
        font: {
          size: 18,
          family: 'Arial, sans-serif'
        },
        padding: {
            bottom: 20
        }
      }
    },
    scales: {
      x: {
        grid: {
          display: false
        }
      },
      y: {
        title: {
          display: true,
          text: `Price (${currencySymbol})`
        },
        grid: {
          color: 'rgba(0, 0, 0, 0.05)'
        }
      }
    }
  };

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
        setError(`⚠️ No matching companies found for "${searchTerm}"`);
      }
    } catch (err) {
      console.error(err);
      setError("❌ Error fetching company symbols.");
    }

    setLoading(false);
  };

  // ------------------ Fetch Historical Data ------------------
  const fetchHistoricalData = async (targetSymbol) => {
    setLoading(true);
    setError(null);

    try {
      const response = await fetch(
        `http://127.0.0.1:5000/api/history?symbol=${targetSymbol}`
      );
      const data = await response.json();

      if (data.error) {
        setError(data.error);
        return;
      }

      // Store historical data in chartData state
      setChartData({
        labels: data.dates,
        datasets: [
          {
            data: data.prices,
          },
        ],
      });
      // Store last close price
      setLastClosePrice(data.prices[data.prices.length - 1]);
    } catch (err) {
      console.error(err);
      setError("Failed to fetch historical data.");
    } finally {
      setLoading(false);
    }
  };

  // ------------------ Fetch Predicted Data ------------------
  const fetchPredictedData = async () => {
    setLoading(true);
    setError(null);
    setMetrics(null);
    setPredictedPrices(null);

    try {
      // 🔹 Step 1: Ensure historical data is fetched first
      // This is a simplified pattern. In a real app, you'd fetch both in parallel or combine the calls.
      await fetchHistoricalData(symbol);
      
      // 🔹 Step 2: Fetch predicted prices
      const predictResponse = await fetch(
        `http://127.0.0.1:5000/api/predict?symbol=${symbol}`
      );
      const predictData = await predictResponse.json();

      if (predictData.error) {
        setError(predictData.error);
        return;
      }

      setPredictedPrices({
          future_dates: predictData.future_dates,
          predicted_next_5_days: predictData.predicted_next_5_days,
      });

      // 🔹 Step 3: Fetch model accuracy metrics
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
      setError("Failed to fetch prediction data or metrics.");
    } finally {
      setLoading(false);
    }
  };

  // Effect to load initial historical data on mount or symbol change
  useEffect(() => {
    fetchHistoricalData(symbol);
  }, [symbol]);

  // ------------------ Render ------------------
  return (
    <div style={{ fontFamily: "Arial, sans-serif", backgroundColor: "#f4f7fa", minHeight: "100vh" }}>

      {/* --- 1. Header & Search --- */}
      <header style={{ 
          backgroundColor: "#2c3e50", 
          color: "white", 
          padding: "15px 30px", 
          display: "flex", 
          justifyContent: "space-between", 
          alignItems: "center",
          boxShadow: "0 2px 4px rgba(0, 0, 0, 0.1)"
      }}>
        <h2 style={{ margin: 0 }}>📈 DL Stock Dashboard</h2>
        
        <div style={{ display: "flex", alignItems: "center", position: "relative" }}>
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            onKeyDown={(e) => { if (e.key === 'Enter') handleSearch(); }}
            placeholder="Search company name/symbol..."
            style={{ 
              padding: "10px", 
              borderRadius: "20px", 
              border: "none", 
              width: "250px",
              marginRight: "10px",
            }}
          />
          <button
            onClick={handleSearch}
            style={{ 
              padding: "10px 15px", 
              backgroundColor: "#28a745", 
              color: "white", 
              border: "none", 
              borderRadius: "20px", 
              cursor: "pointer", 
              fontWeight: "bold" 
            }}
          >
            Go
          </button>
          
          {searchResults.length > 0 && (
            <div 
              style={{
                position: "absolute",
                top: "100%", 
                right: "0",
                backgroundColor: "white",
                boxShadow: "0 4px 12px rgba(0,0,0,0.2)",
                borderRadius: "8px",
                marginTop: "5px",
                width: "300px",
                zIndex: 1000,
                textAlign: "left",
                color: "#333"
              }}
            >
              {searchResults.map((result, index) => (
                <div
                  key={index}
                  onClick={() => {
                    setSymbol(result.symbol);
                    setSearchTerm(result.name);
                    setSearchResults([]);
                    fetchHistoricalData(result.symbol); // Load data immediately
                  }}
                  style={{
                    padding: "10px 15px",
                    cursor: "pointer",
                    borderBottom: "1px solid #eee",
                    transition: "background-color 0.2s"
                  }}
                  onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#f0f0f0'}
                  onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'white'}
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
      </header>

      {/* --- 2. Main Content Area --- */}
      <div style={{ padding: "30px", display: "flex", gap: "20px", maxWidth: "1400px", margin: "0 auto" }}>
        
        {/* --- Left Column: Key Metrics & Prediction Control --- */}
        <div style={{ flex: "0 0 300px", minWidth: "300px" }}>
          <div style={{ 
              backgroundColor: "white", 
              padding: "20px", 
              borderRadius: "10px", 
              boxShadow: "0 4px 12px rgba(0, 0, 0, 0.05)",
              marginBottom: "20px"
          }}>
            <h3 style={{ margin: "0 0 15px 0", color: "#333" }}>{symbol} Analysis</h3>
            <div style={{ fontSize: "14px", color: "#6c757d", borderBottom: "1px solid #eee", paddingBottom: "10px", marginBottom: "15px" }}>
                Last Symbol Selected: <strong>{symbol}</strong>
            </div>

            <div style={{ marginBottom: "20px" }}>
                <h4 style={{ margin: "0 0 10px 0", color: "#333" }}>Last Close Price</h4>
                <div style={{ fontSize: "32px", fontWeight: "bold", color: "#007bff" }}>
                    {lastClosePrice ? `${currencySymbol} ${lastClosePrice.toFixed(2)}` : 'N/A'}
                </div>
            </div>

            <button
                onClick={fetchPredictedData}
                disabled={loading}
                style={{
                    width: "100%",
                    padding: "12px",
                    backgroundColor: loading ? "#6c757d" : "#ffc107",
                    color: "white",
                    border: "none",
                    borderRadius: "8px",
                    cursor: loading ? "not-allowed" : "pointer",
                    fontWeight: "bold",
                    fontSize: "16px",
                    transition: "background-color 0.3s"
                }}
            >
                {loading ? "Processing Model..." : "✨ Generate 5-Day Prediction"}
            </button>
          </div>

          {/* --- Model Performance (Metrics Cards) --- */}
          {metrics && (
            <div style={{ marginBottom: "20px" }}>
                <h4 style={{ color: "#333", borderBottom: "2px solid #eee", paddingBottom: "10px", marginBottom: "15px" }}>
                    Model Metrics
                </h4>
                <div style={{ display: "flex", flexDirection: "column", gap: "10px" }}>
                    <MetricCard 
                        title="Accuracy (100-MAPE)" 
                        value={parseFloat(metrics.Accuracy).toFixed(2)} 
                        unit="%" 
                        color="#28a745" // Green for high accuracy
                        tooltipText="Accuracy is calculated as 100% minus the Mean Absolute Percentage Error (MAPE). Higher is better."
                    />
                    <MetricCard 
                        title="RMSE (Error)" 
                        value={parseFloat(metrics.RMSE).toFixed(2)} 
                        unit={currencySymbol} 
                        color="#dc3545" // Red for errors
                        tooltipText="Root Mean Squared Error. Measures the standard deviation of prediction errors. Lower is better."
                    />
                    <MetricCard 
                        title="MAE (Avg. Deviation)" 
                        value={parseFloat(metrics.MAE).toFixed(2)} 
                        unit={currencySymbol} 
                        color="#ffc107" // Orange/Yellow
                        tooltipText="Mean Absolute Error. The average magnitude of errors in a set of forecasts. Lower is better."
                    />
                </div>
            </div>
          )}
          
          {/* --- Full Metrics List (Optional Detailed View) --- */}
          {metrics && (
            <div style={{ 
                backgroundColor: "white", 
                padding: "20px", 
                borderRadius: "10px", 
                boxShadow: "0 4px 12px rgba(0, 0, 0, 0.05)",
                marginTop: "20px"
            }}>
                <h4 style={{ margin: "0 0 10px 0", color: "#333" }}>Detailed Evaluation</h4>
                <ul style={{ listStyleType: "none", padding: 0, lineHeight: "2" }}>
                    <li style={{ borderBottom: "1px dotted #eee" }}><b>R² Score:</b> <span style={{ float: "right" }}>{parseFloat(metrics.R2_Score).toFixed(4)}</span></li>
                    <li style={{ borderBottom: "1px dotted #eee" }}><b>MAPE:</b> <span style={{ float: "right" }}>{parseFloat(metrics.MAPE).toFixed(2)}%</span></li>
                    <li style={{ borderBottom: "1px dotted #eee" }}><b>MSE:</b> <span style={{ float: "right" }}>{parseFloat(metrics.MSE).toFixed(2)}</span></li>
                </ul>
            </div>
          )}

        </div>

        {/* --- Right Column: Chart & Prediction Table --- */}
        <div style={{ flex: 1 }}>
          {error && <div style={{ color: "white", backgroundColor: "#dc3545", padding: "10px", borderRadius: "5px", marginBottom: "15px" }}>{error}</div>}

          {/* Chart Area */}
          <div style={{ 
              backgroundColor: "white", 
              padding: "20px", 
              borderRadius: "10px", 
              boxShadow: "0 4px 12px rgba(0, 0, 0, 0.05)",
              height: "500px", // Fixed height for chart
              marginBottom: "20px"
          }}>
            {combinedChartData ? (
              <Line data={combinedChartData} options={chartOptions} />
            ) : (
              <div style={{ height: "100%", display: "flex", justifyContent: "center", alignItems: "center", color: "#6c757d" }}>
                {loading ? "Loading chart data..." : "Click 'Generate 5-Day Prediction' to view the full forecast."}
              </div>
            )}
          </div>
          
          {/* Prediction Table */}
          {predictedPrices && (
            <div style={{ 
                backgroundColor: "white", 
                padding: "20px", 
                borderRadius: "10px", 
                boxShadow: "0 4px 12px rgba(0, 0, 0, 0.05)",
            }}>
                <h4 style={{ margin: "0 0 15px 0", color: "#333" }}>🔮 Next 5-Day Price Forecast</h4>
                <table style={{ width: "100%", borderCollapse: "collapse" }}>
                    <thead>
                        <tr style={{ backgroundColor: "#f8f9fa" }}>
                            <th style={{ padding: "10px", borderBottom: "2px solid #ddd", textAlign: "left" }}>Date</th>
                            <th style={{ padding: "10px", borderBottom: "2px solid #ddd", textAlign: "right" }}>Predicted Price</th>
                        </tr>
                    </thead>
                    <tbody>
                        {predictedPrices.future_dates.map((date, index) => (
                            <tr key={index} style={{ borderBottom: "1px solid #eee", transition: "background-color 0.2s" }} onMouseEnter={(e) => e.currentTarget.style.backgroundColor = '#fcfcfc'} onMouseLeave={(e) => e.currentTarget.style.backgroundColor = 'white'}>
                                <td style={{ padding: "10px", textAlign: "left" }}>{date}</td>
                                <td style={{ padding: "10px", textAlign: "right", fontWeight: "bold", color: "#28a745" }}>
                                    {currencySymbol} {predictedPrices.predicted_next_5_days[index].toFixed(2)}
                                </td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
          )}

        </div>
      </div>
    </div>
  );
}

export default App;