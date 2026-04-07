import streamlit as st
import pandas as pd
import plotly.express as px
import os

st.set_page_config(page_title="Stock Dashboard", layout="wide")

st.title("📊 Stock Analytics Dashboard")

# Load data
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
file_path = os.path.join(BASE_DIR, "output", "monthly_ohlcv_data.xlsx")

df = pd.read_excel(file_path)

# Sidebar
st.sidebar.header("Filters")

ticker = st.sidebar.selectbox("Select Ticker", df["Ticker"].unique())

date_range = st.sidebar.date_input(
    "Select Date Range",
    [df["Date"].min(), df["Date"].max()]
)

# Filter data
df["Date"] = pd.to_datetime(df["Date"])

filtered_df = df[
    (df["Ticker"] == ticker) &
    (df["Date"] >= pd.to_datetime(date_range[0])) &
    (df["Date"] <= pd.to_datetime(date_range[1]))
]

# Layout
col1, col2 = st.columns(2)

# Price Chart
with col1:
    st.subheader("Price Trend")
    fig = px.line(filtered_df, x="Date", y="Close", title=f"{ticker} Price")
    st.plotly_chart(fig, width='stretch')

# Returns Chart
with col2:
    st.subheader("Monthly Returns")
    fig2 = px.bar(filtered_df, x="Date", y="Monthly_Return", title=f"{ticker} Returns")
    st.plotly_chart(fig2, width='stretch')

# Moving Averages
st.subheader("Moving Averages")

fig3 = px.line(
    filtered_df,
    x="Date",
    y=["Close", "MA_3", "MA_6", "MA_12"],
    title=f"{ticker} Moving Averages"
)

st.plotly_chart(fig3, width='stretch')

# Metrics
st.subheader("Key Metrics")

latest = filtered_df.iloc[-1]

col3, col4, col5 = st.columns(3)

col3.metric("Last Close", f"{latest['Close']:.2f}")
col4.metric("Volatility (%)", f"{latest['Volatility']:.2f}")
col5.metric("CAGR (%)", f"{latest['CAGR']:.2f}")

# ===============================
# 🤖 LINEAR REGRESSION
# ===============================
from sklearn.linear_model import LinearRegression
import numpy as np

st.subheader("🤖 AI Price Prediction (Linear Regression)")

model_df = filtered_df.copy().dropna(subset=["Close"])
model_df["Date_Ordinal"] = model_df["Date"].map(pd.Timestamp.toordinal)

X_lr = model_df[["Date_Ordinal"]]
y_lr = model_df["Close"]

lr_model = LinearRegression()
lr_model.fit(X_lr, y_lr)

# Future prediction
last_date = model_df["Date"].max()
future_dates = pd.date_range(last_date, periods=6, freq="ME")

future_ordinal = pd.DataFrame(
    [d.toordinal() for d in future_dates],
    columns=["Date_Ordinal"]
)

predictions = lr_model.predict(future_ordinal)

future_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted_Close": predictions
})

import plotly.graph_objects as go

fig_pred = go.Figure()

fig_pred.add_trace(go.Scatter(
    x=model_df["Date"],
    y=model_df["Close"],
    mode='lines',
    name='Actual'
))

fig_pred.add_trace(go.Scatter(
    x=future_df["Date"],
    y=future_df["Predicted_Close"],
    mode='lines+markers',
    name='LR Predicted'
))

st.plotly_chart(fig_pred, width='stretch')

# ===============================
# 🧠 LSTM MODEL
# ===============================
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error

st.subheader("🧠 LSTM Deep Learning Prediction")

lstm_df = filtered_df.copy().dropna(subset=["Close"])
data = lstm_df["Close"].values.reshape(-1, 1)

# Normalize
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data)

# Create sequences
def create_sequences(data, seq_length=12):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])
    return np.array(X), np.array(y)

X_lstm, y_lstm = create_sequences(scaled_data)

# Train-test split
split = int(len(X_lstm) * 0.8)
X_train, X_test = X_lstm[:split], X_lstm[split:]
y_train, y_test = y_lstm[:split], y_lstm[split:]

# Build model
lstm_model = Sequential()
lstm_model.add(LSTM(50, return_sequences=True, input_shape=(X_lstm.shape[1], 1)))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(50))
lstm_model.add(Dense(1))

lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# Train
lstm_model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0)

# RMSE
y_pred_test = lstm_model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
st.write(f"📉 LSTM RMSE: {rmse:.4f}")

# Future prediction
last_sequence = scaled_data[-12:]
future_preds = []

current_seq = last_sequence.copy()

for _ in range(6):
    pred = lstm_model.predict(current_seq.reshape(1, 12, 1), verbose=0)
    future_preds.append(pred[0][0])
    current_seq = np.append(current_seq[1:], pred, axis=0)

future_preds = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

future_dates_lstm = pd.date_range(lstm_df["Date"].max(), periods=6, freq="ME")

future_lstm_df = pd.DataFrame({
    "Date": future_dates_lstm,
    "LSTM_Predicted": future_preds.flatten()
})

# Plot
fig_lstm = go.Figure()

fig_lstm.add_trace(go.Scatter(
    x=lstm_df["Date"],
    y=lstm_df["Close"],
    mode='lines',
    name='Actual'
))

fig_lstm.add_trace(go.Scatter(
    x=future_lstm_df["Date"],
    y=future_lstm_df["LSTM_Predicted"],
    mode='lines+markers',
    name='LSTM Predicted'
))

st.plotly_chart(fig_lstm, width='stretch')