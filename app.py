import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Stock Prediction Dashboard",
    page_icon="📈",
    layout="wide"
)

# -----------------------------
# Title
# -----------------------------
st.title("📈 Stock Movement Prediction Dashboard")
st.markdown("Predict the next day's stock movement using Machine Learning.")

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("models/stock_model.pkl")

# -----------------------------
# Load Dataset
# -----------------------------
stock = pd.read_csv(
    "data/reliance_improved_dataset.csv",
    index_col=0,
    parse_dates=True
)

# -----------------------------
# Latest Row
# -----------------------------
latest = stock.iloc[-1]

# -----------------------------
# Features Used
# -----------------------------
features = [
    'Return',
    'MA_5',
    'MA_20',
    'EMA_20',
    'Volatility_20',
    'Lag_1',
    'Lag_2',
    'Lag_3',
    'Momentum',
    'Price_Range'
]

X_latest = latest[features].values.reshape(1, -1)

# -----------------------------
# Prediction
# -----------------------------
prediction = model.predict(X_latest)[0]

probability = model.predict_proba(X_latest)[0]

confidence = max(probability) * 100

prediction_text = "📈 UP" if prediction == 1 else "📉 DOWN"

# -----------------------------
# Risk
# -----------------------------
volatility = latest["Volatility_20"]

if volatility < 0.015:
    risk = "🟢 LOW"

elif volatility < 0.03:
    risk = "🟡 MEDIUM"

else:
    risk = "🔴 HIGH"

# -----------------------------
# Trend
# -----------------------------
if latest["MA_5"] > latest["MA_20"]:
    trend = "📈 Bullish"

else:
    trend = "📉 Bearish"

# -----------------------------
# Dashboard Summary
# -----------------------------
st.subheader("📊 Market Summary")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Prediction",
        prediction_text
    )

with col2:
    st.metric(
        "Confidence",
        f"{confidence:.2f}%"
    )

with col3:
    st.metric(
        "Risk Level",
        risk
    )

with col4:
    st.metric(
        "Market Trend",
        trend
    )

st.divider()

# -----------------------------
# Charts
# -----------------------------
left, right = st.columns(2)

with left:

    st.subheader("📉 Closing Price")

    st.line_chart(stock["Close"])

with right:

    st.subheader("📊 Moving Averages")

    st.line_chart(stock[["MA_5", "MA_20"]])

st.divider()

# -----------------------------
# Technical Indicators
# -----------------------------
st.subheader("📌 Technical Indicators")

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "EMA 20",
        f"{latest['EMA_20']:.2f}"
    )

with c2:
    st.metric(
        "MA 20",
        f"{latest['MA_20']:.2f}"
    )

with c3:
    st.metric(
        "Momentum",
        f"{latest['Momentum']:.2f}"
    )

with c4:
    st.metric(
        "Volatility",
        f"{latest['Volatility_20']:.4f}"
    )

st.divider()

# -----------------------------
# Model Information
# -----------------------------
st.subheader("🤖 Model Information")

info1, info2, info3 = st.columns(3)

with info1:
    st.metric(
        "Model",
        "Gradient Boosting"
    )

with info2:
    st.metric(
        "Accuracy",
        "52.00%"
    )

with info3:
    st.metric(
        "ROC-AUC",
        "0.52"
    )

st.divider()

# -----------------------------
# Recent Dataset
# -----------------------------
st.subheader("📋 Recent Stock Data")

st.dataframe(stock.tail())