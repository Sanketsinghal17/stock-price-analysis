📊 Stock Price Prediction Dashboard

An AI-powered stock market analysis web application that allows users to:

Search companies by name
Fetch historical stock prices
Predict future stock prices for the next 5 days using LSTM
View model evaluation metrics
Track frequently searched stocks
Visualize stock trends using interactive charts

This project uses React.js for frontend, Flask (Python) for backend, and an LSTM deep learning model for stock price prediction.

🚀 Features
🔍 Company Search

Users can search companies by name using the Alpha Vantage API.

Example:

Search "TCS"
Search "Apple"
Search "Reliance"

The app automatically fetches stock symbols.

📈 Historical Stock Data

Fetches 1 year of historical stock prices using Yahoo Finance API.

Features:

Supports Indian stocks (.NS, .BSE)
Supports global stocks
Automatically switches exchange if stock isn't found

Example:

TCS.NS
RELIANCE.NS
AAPL
GOOGL
🤖 Stock Price Prediction

Uses an LSTM neural network model to predict stock prices for the next 5 days.

Prediction includes:

Future dates
Predicted prices
Difference from current stock price
📊 Model Evaluation Metrics

Displays model performance metrics:

MAE
MSE
RMSE
R² Score
MAPE
Accuracy %

These metrics are stored in model_metrics.csv.

🔥 Frequently Searched Stocks

Tracks most searched stocks using SQLite database.

Users can quickly access popular stocks from homepage.

📉 Interactive Charts

Built using Chart.js:

Historical price visualization
Predicted price visualization
🛠 Tech Stack
Frontend
React.js
Chart.js
JavaScript
CSS
Backend
Flask
Flask-CORS
Python
Machine Learning
TensorFlow / Keras
LSTM Model
Scikit-learn
NumPy
Pandas
APIs
Yahoo Finance (yfinance)
Alpha Vantage API
Database
SQLite (for frequent searches tracking)
📂 Project Structure
stock-price-analysis/
│
├── backend/
│   ├── app.py
│   ├── train_model.py
│   ├── stock_model.h5
│   ├── model_metrics.csv
│   ├── requirements.txt
│
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── App.js
│   │   ├── components/
│   │   │   ├── FrequentSearches.jsx
│   │
│   ├── package.json
│
├── research_Paper/
│   ├── research_paper.docx
│
├── README.md
└── requirements.txt
⚙️ Installation & Setup
1 Clone Repository
git clone https://github.com/your-username/stock-price-analysis.git
cd stock-price-analysis
Backend Setup

Move into backend folder:

cd backend

Create virtual environment:

python -m venv venv

Activate virtual environment:

Windows
venv\Scripts\activate
Mac/Linux
source venv/bin/activate

Install dependencies:

pip install -r requirements.txt

Run Flask backend:

python app.py

Backend runs on:

http://127.0.0.1:5000
Frontend Setup

Move into frontend folder:

cd frontend

Install dependencies:

npm install

Start React app:

npm start

Frontend runs on:

http://localhost:3000
Machine Learning Model Training

To retrain the LSTM model:

cd backend
python train_model.py

This generates:

stock_model.h5
model_metrics.csv
API Endpoints
Fetch Historical Stock Data
GET /api/history?symbol=TCS.NS
Response
{
  "symbol": "TCS.NS",
  "dates": [...],
  "prices": [...]
}
Predict Future Prices
GET /api/predict?symbol=TCS.NS
Response
{
  "symbol": "TCS.NS",
  "predicted_next_5_days": [...],
  "future_dates": [...]
}
Get Model Metrics
GET /api/metrics
Record Search
POST /api/record-search
Top Searches
GET /api/top-searches
How It Works
Step 1

User searches company name

↓

Step 2

Alpha Vantage API returns matching symbols

↓

Step 3

User selects stock symbol

↓

Step 4

Historical data fetched using Yahoo Finance

↓

Step 5

LSTM model predicts future prices

↓

Step 6

Results displayed using charts

↓

Step 7

Searches stored for frequent stock tracking

Model Architecture

The project uses an LSTM Neural Network for time series forecasting.

Workflow:
Data Collection
Data Preprocessing
Feature Scaling
Sequence Creation
LSTM Training
Prediction
Evaluation
Future Improvements
Real-time stock updates
Candlestick charts
News sentiment analysis
Portfolio tracking
User authentication
Dark mode UI
Cloud deployment

Research Paper

This project also includes a research paper:

research_Paper/research_paper.docx
Challenges Faced
Handling invalid stock symbols
Supporting Indian + international stocks
Improving prediction accuracy
Managing API limitations
Integrating ML model with frontend
Learning Outcomes

Through this project, I learned:

Full Stack Development
API Integration
Machine Learning Deployment
Time Series Forecasting
React + Flask Integration
Data Visualization
Author

Sanket Singhal

B.Tech CSE (Data Science)
ABES Engineering College

License

This project is licensed under the MIT License.