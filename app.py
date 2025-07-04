import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from textblob import TextBlob

# -----------------------------
# ✅ Manual RSI Function
# -----------------------------
def compute_RSI(data, time_window=14):
    diff = data['Close'].diff(1)
    gain = diff.where(diff > 0, 0)
    loss = -diff.where(diff < 0, 0)
    avg_gain = gain.rolling(window=time_window).mean()
    avg_loss = loss.rolling(window=time_window).mean()
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# -----------------------------
# ✅ Streamlit Page Config
# -----------------------------
st.set_page_config(page_title="📈 Stock Intelligence Dashboard", layout="wide")
st.title("🧠 Smart Stock Intelligence Dashboard")

# -----------------------------
# Sidebar: Select stock
# -----------------------------
st.sidebar.header("📌 Choose Your Stock")

stock_options = {
    "Apple Inc. (AAPL)": "AAPL",
    "Tesla Inc. (TSLA)": "TSLA",
    "Infosys (INFY.NS)": "INFY.NS",
    "Reliance (RELIANCE.NS)": "RELIANCE.NS",
    "HDFC Bank (HDFCBANK.NS)": "HDFCBANK.NS",
    "Tata Motors (TATAMOTORS.NS)": "TATAMOTORS.NS",
    "Amazon (AMZN)": "AMZN"
}

selected_name = st.sidebar.selectbox("Select a stock", list(stock_options.keys()))
ticker = stock_options[selected_name]
days = st.sidebar.slider("Days of historical data", 60, 365, 180)

# -----------------------------
# Load stock data
# -----------------------------
@st.cache_data
def load_data(ticker, days):
    data = yf.download(ticker, period=f"{days}d")
    data['Previous_Close'] = data['Close'].shift(1)
    data['5_day_MA'] = data['Close'].rolling(5).mean()
    data['10_day_MA'] = data['Close'].rolling(10).mean()
    data['RSI'] = compute_RSI(data)
    data['Target'] = data['Close']
    return data.dropna()

data = load_data(ticker, days)

# -----------------------------
# ML Prediction
# -----------------------------
features = ['Previous_Close', '5_day_MA', '10_day_MA', 'Volume']
X = data[features]
y = data['Target']
model = LinearRegression()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
model.fit(X_train, y_train)

latest = X.iloc[-1].values.reshape(1, -1)
predicted_price = float(model.predict(latest)[0])
current_price = float(data['Close'].iloc[-1])
trend = "UP" if predicted_price > current_price else "DOWN"

# -----------------------------
# Display Metrics
# -----------------------------
st.subheader("📊 Market Overview")
col1, col2, col3 = st.columns(3)
col1.metric("Current Price", f"${current_price:.2f}")
col2.metric("Predicted Price", f"${predicted_price:.2f}", f"${predicted_price - current_price:.2f}")
col3.metric("Trend", "📈 UP" if trend == "UP" else "📉 DOWN")

# -----------------------------
# RSI Signal
# -----------------------------
st.subheader("📌 RSI (Relative Strength Index)")
rsi_value = data['RSI'].iloc[-1]
if rsi_value < 30:
    st.success(f"RSI: {rsi_value:.2f} — 🔽 **Oversold**: Possible buy signal.")
elif rsi_value > 70:
    st.warning(f"RSI: {rsi_value:.2f} — 🔼 **Overbought**: Caution advised.")
else:
    st.info(f"RSI: {rsi_value:.2f} — ⚖️ Neutral zone.")

# -----------------------------
# Candlestick Chart
# -----------------------------
st.subheader("🕯️ Candlestick Chart")
candlestick = go.Figure(data=[go.Candlestick(
    x=data.index,
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close']
)])
candlestick.update_layout(xaxis_rangeslider_visible=False, height=400)
st.plotly_chart(candlestick, use_container_width=True)

# -----------------------------
# Actual vs Predicted
# -----------------------------
st.subheader("📈 Model Validation: Actual vs Predicted")
y_pred = model.predict(X_test)
acc_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
st.line_chart(acc_df)

# -----------------------------
# Sentiment Analysis (Demo)
# -----------------------------
st.subheader("📰 News Sentiment (Demo Headlines)")
headlines = [
    f"{selected_name.split()[0]} reports strong quarterly earnings",
    f"{selected_name.split()[0]} faces legal scrutiny over data",
    f"Market outlook for {selected_name.split()[0]} improves as demand rises"
]

for h in headlines:
    score = TextBlob(h).sentiment.polarity
    sentiment = "🔵 Positive" if score > 0 else "🔴 Negative" if score < 0 else "⚪ Neutral"
    st.markdown(f"- {h} — **{sentiment}** (Score: {score:.2f})")

# -----------------------------
# Save Watchlist Option
# -----------------------------
st.sidebar.markdown("---")
if st.sidebar.button("📥 Save to Watchlist"):
    pd.DataFrame([[selected_name, ticker, current_price, predicted_price]]).to_csv("watchlist.csv", index=False, header=["Company", "Ticker", "Current", "Predicted"])
    st.sidebar.success("✅ Watchlist saved as `watchlist.csv`")
