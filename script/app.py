import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from streamlit_autorefresh import st_autorefresh

st.set_page_config(page_title="💻 AI Trading Terminal", layout="wide")

# =========================
# 🔄 AUTO REFRESH
# =========================
refresh_rate = st.sidebar.slider("🔄 Refresh Rate (sec)", 1, 60, 5)
st_autorefresh(interval=refresh_rate * 1000, key="live_data")

# =========================
# 🎨 LIGHT UI
# =========================
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: #f8fafc;
    color: #0f172a;
}
[data-testid="stSidebar"] {
    background-color: #e2e8f0;
}
h1, h2, h3 {
    color: #1e293b;
}
[data-testid="stMetricValue"] {
    color: #020617;
    font-weight: bold;
}
input {
    background-color: white !important;
    color: black !important;
}
</style>
""", unsafe_allow_html=True)

st.title("💻 AI Trading Terminal")

# =========================
# 🔴 LIVE STATUS
# =========================
st.markdown(f"""
<div style="
    background: #e0f2fe;
    padding: 10px;
    border-radius: 10px;
    color: #0369a1;
    font-weight: 600;
">
    🔴 LIVE MODE — Updating every {refresh_rate} sec
</div>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR INPUTS
# =========================
st.sidebar.header("⚙️ Controls")

ticker = st.sidebar.text_input("Ticker", "AAPL", key="ticker")
compare_tickers = st.sidebar.text_input("Compare", "MSFT,GOOG,TSLA", key="compare")
start = st.sidebar.date_input("Start", pd.to_datetime("2020-01-01"), key="start")
end = st.sidebar.date_input("End", pd.to_datetime("today"), key="end")

# =========================
# SAFE DOWNLOAD FUNCTION
# =========================
def load_data(symbol):
    df = yf.download(symbol, start=start, end=end, auto_adjust=True)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty:
        return df

    for col in ['Open','High','Low','Close','Volume']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df.dropna(inplace=True)
    return df

# =========================
# LOAD DATA
# =========================
df = load_data(ticker)

if df.empty:
    st.error("No data found")
    st.stop()

df.reset_index(inplace=True)

# =========================
# INDICATORS
# =========================
def add_indicators(df):
    df = df.copy()

    df['50_DMA'] = df['Close'].rolling(50).mean()
    df['200_DMA'] = df['Close'].rolling(200).mean()

    delta = df['Close'].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))

    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    df['Signal'] = df['MACD'].ewm(span=9).mean()

    return df

df = add_indicators(df)
df.dropna(inplace=True)

# =========================
# KPI
# =========================
latest = df.iloc[-1]

price = float(latest['Close'])
rsi = float(latest['RSI'])

trend = "🚀 Bullish" if latest['50_DMA'] > latest['200_DMA'] else "📉 Bearish"

score = 0
if rsi < 30: score += 1
if latest['MACD'] > latest['Signal']: score += 1

signal = "🔥 BUY" if score == 2 else "⚖️ HOLD" if score == 1 else "❌ SELL"
confidence = int((score / 2) * 100)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Price", f"${price:.2f}")
col2.metric("RSI", f"{rsi:.2f}")
col3.metric("Trend", trend)
col4.metric("Signal", signal)

# =========================
# AI STATUS
# =========================
st.markdown(f"""
<div style="
    background: linear-gradient(90deg, #ecfdf5, #d1fae5);
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #10b981;
    font-weight: 600;
    color: #065f46;
">
    🚀 AI Engine Active | Confidence: {confidence}%
</div>
""", unsafe_allow_html=True)

# =========================
# DARK LAYOUT FIX 🔥
# =========================
def dark_layout(fig, h=500):
    fig.update_layout(
        template="plotly_dark",
        height=h,
        plot_bgcolor="#020617",
        paper_bgcolor="#020617",
        xaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(showgrid=True, gridcolor="rgba(255,255,255,0.08)"),
        font=dict(color="#e2e8f0"),
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

# =========================
# MAIN CHART
# =========================
fig = go.Figure()

fig.add_trace(go.Candlestick(
    x=df['Date'],
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    increasing_line_color='#22c55e',
    decreasing_line_color='#ef4444'
))

buy = df[df['RSI'] < 30]
sell = df[df['RSI'] > 70]

fig.add_trace(go.Scatter(
    x=buy['Date'], y=buy['Close'],
    mode='markers',
    marker=dict(color='#22c55e', size=10),
    name="BUY"
))

fig.add_trace(go.Scatter(
    x=sell['Date'], y=sell['Close'],
    mode='markers',
    marker=dict(color='#ef4444', size=10),
    name="SELL"
))

st.plotly_chart(dark_layout(fig, 600), use_container_width=True)

# =========================
# MULTI STOCK
# =========================
st.subheader("📊 Multi-Stock Comparison")

fig2 = go.Figure()
colors = ["#38bdf8", "#f97316", "#a78bfa", "#34d399"]

for i, t in enumerate([x.strip().upper() for x in compare_tickers.split(",") if x.strip()]):
    try:
        temp = load_data(t)
        if not temp.empty:
            fig2.add_trace(go.Scatter(
                x=temp.index,
                y=temp['Close'],
                name=t,
                line=dict(color=colors[i % len(colors)], width=2)
            ))
    except:
        st.warning(f"⚠️ Error loading {t}")

st.plotly_chart(dark_layout(fig2, 400), use_container_width=True)

# =========================
# LSTM
# =========================
st.subheader("🧠 AI Forecast Engine")

data = df['Close'].values.reshape(-1,1)

if len(data) > 20:
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(data)

    def seq(data, n=12):
        X, y = [], []
        for i in range(len(data)-n):
            X.append(data[i:i+n])
            y.append(data[i+n])
        return np.array(X), np.array(y)

    X, y = seq(scaled)

    split = int(len(X)*0.8)
    X_train, X_test = X[:split], X[split:]

    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=(X.shape[1],1)))
    model.add(Dropout(0.2))
    model.add(LSTM(50))
    model.add(Dense(1))

    model.compile(optimizer='adam', loss='mse')
    model.fit(X_train, y[:split], epochs=5, verbose=0)

    preds = model.predict(X_test)

    preds = scaler.inverse_transform(preds)
    actual = scaler.inverse_transform(y[split:])

    fig3 = go.Figure()

    fig3.add_trace(go.Scatter(
        y=actual.flatten(),
        name="Actual",
        line=dict(color="#38bdf8", width=3)
    ))

    fig3.add_trace(go.Scatter(
        y=preds.flatten(),
        name="Predicted",
        line=dict(color="#f97316", width=3)
    ))

    st.plotly_chart(dark_layout(fig3, 400), use_container_width=True)