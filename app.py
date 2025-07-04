import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="📈 Stock Price Predictor", layout="centered")

# 🧾 App title
st.title("💹 Stock Price Predictor")
st.markdown("_Predict the next day's closing price using ML_")

# 📥 User input
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, INFY)", "AAPL")
n_days = st.slider("Days of historical data to use", 60, 365, 180)

# ⏳ Load data
@st.cache_data
def load_data(ticker, n_days):
    data = yf.download(ticker, period=f"{n_days}d")
    data['Previous_Close'] = data['Close'].shift(1)
    data['5_day_MA'] = data['Close'].rolling(5).mean()
    data['10_day_MA'] = data['Close'].rolling(10).mean()
    data['Target'] = data['Close']
    data.dropna(inplace=True)
    return data

data = load_data(ticker, n_days)

# 📈 Show chart
st.subheader(f"{ticker} Closing Price")
st.line_chart(data['Close'])

# ✅ Train model
features = ['Previous_Close', '5_day_MA', '10_day_MA', 'Volume']
X = data[features]
y = data['Target']

model = LinearRegression()
model.fit(X, y)

# Predict next day's price
latest = X.iloc[-1].values.reshape(1, -1)
predicted_price = float(model.predict(latest)[0])
current_price = float(data['Close'].iloc[-1])

st.markdown("### 📊 Prediction")
st.success(f"📈 Predicted Price for Next Day: **${predicted_price:.2f}**")
st.info(f"📍 Current Price: **${current_price:.2f}**")
