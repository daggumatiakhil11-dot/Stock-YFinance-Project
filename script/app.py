import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from streamlit_autorefresh import st_autorefresh
from sklearn.linear_model import LogisticRegression, LinearRegression

st.set_page_config(page_title="💻 AI Trading Terminal", layout="wide")

refresh_rate = st.sidebar.slider("🔄 Refresh Rate (sec)", 1, 60, 5)
st_autorefresh(interval=refresh_rate * 1000, key="live_data")

st.title("💻 AI Trading Terminal")

st.sidebar.header("⚙️ Controls")

ticker = st.sidebar.text_input("Ticker", "AAPL")
compare_tickers = st.sidebar.text_input("Compare", "MSFT,GOOG,TSLA")
start = st.sidebar.date_input("Start", pd.to_datetime("2020-01-01"))
end = st.sidebar.date_input("End", pd.to_datetime("today"))

@st.cache_data
def load_data(symbol, start, end):
    try:
        df = yf.download(symbol, start=start, end=end, auto_adjust=True)

        if df.empty:
            return df

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
        df.dropna(inplace=True)
        df.reset_index(inplace=True)

        return df
    except:
        return pd.DataFrame()

df = load_data(ticker, start, end)

if df.empty:
    st.error("❌ No data found for this ticker")
    st.stop()

def add_indicators(df):
    df = df.copy()

    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA30'] = df['Close'].rolling(30).mean()
    df['50_DMA'] = df['Close'].rolling(50).mean()
    df['200_DMA'] = df['Close'].rolling(200).mean()

    rolling_std = df['Close'].rolling(20).std()
    df['BB_Upper'] = df['MA20'] + (rolling_std * 2)
    df['BB_Lower'] = df['MA20'] - (rolling_std * 2)

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

@st.cache_resource
def train_model(df):
    df = df.copy()

    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)

    features = ['Close', 'MA20', 'RSI', 'MACD']
    X = df[features]
    y = df['Target']

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model

model = train_model(df)

latest = df.iloc[-1]

features = np.array([
    latest['Close'],
    latest['MA20'],
    latest['RSI'],
    latest['MACD']
]).reshape(1, -1)

prediction = model.predict(features)[0]
proba = model.predict_proba(features)[0]
confidence = round(max(proba) * 100, 2)

price = float(latest['Close'])
rsi = float(latest['RSI'])

trend = "🚀 Bullish" if latest['50_DMA'] > latest['200_DMA'] else "📉 Bearish"

score = 0
if prediction == 1:
    score += 1
if rsi < 30:
    score += 1
if latest['MACD'] > latest['Signal']:
    score += 1

if score >= 2:
    signal = "🔥 STRONG BUY"
elif score == 1:
    signal = "⚖️ HOLD"
else:
    signal = "❌ SELL"

col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
col1.metric("Price", f"${price:.2f}")
col2.metric("RSI (14)", f"{rsi:.2f}")
col3.metric("MA30", f"{latest['MA30']:.2f}")
col4.metric("50 DMA", f"{latest['50_DMA']:.2f}")
col5.metric("200 DMA", f"{latest['200_DMA']:.2f}")
col6.metric("AI Signal", signal)
col7.metric("Confidence", f"{confidence}%")


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

fig.add_trace(go.Scatter(x=df['Date'], y=df['MA30'], name="MA30"))
fig.add_trace(go.Scatter(x=df['Date'], y=df['50_DMA'], name="50 DMA"))
fig.add_trace(go.Scatter(x=df['Date'], y=df['200_DMA'], name="200 DMA"))

st.plotly_chart(fig, use_container_width=True)


st.subheader("📊 RSI Indicator (14)")

fig_rsi = go.Figure()
fig_rsi.add_trace(go.Scatter(x=df['Date'], y=df['RSI'], name="RSI"))
fig_rsi.add_hline(y=70)
fig_rsi.add_hline(y=30)

st.plotly_chart(fig_rsi, use_container_width=True)


st.subheader("📅 Monthly Data View")

df_monthly = df.copy()

df_monthly['Date'] = pd.to_datetime(df_monthly['Date'])
df_monthly.set_index('Date', inplace=True)

df_monthly = df_monthly.resample('MS').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
})

df_monthly.reset_index(inplace=True)

df_monthly['MA3'] = df_monthly['Close'].rolling(3).mean()
df_monthly['MA6'] = df_monthly['Close'].rolling(6).mean()

st.subheader("📋 Monthly Market Data")

st.dataframe(
    df_monthly[['Date', 'Close', 'MA3', 'MA6']].tail(12),
    use_container_width=True
)

st.subheader("📊 Monthly Trend")

fig_m = go.Figure()

fig_m.add_trace(go.Scatter(x=df_monthly['Date'], y=df_monthly['Close'], name="Close"))
fig_m.add_trace(go.Scatter(x=df_monthly['Date'], y=df_monthly['MA3'], name="MA3"))
fig_m.add_trace(go.Scatter(x=df_monthly['Date'], y=df_monthly['MA6'], name="MA6"))

st.plotly_chart(fig_m, use_container_width=True)

st.subheader("📋 Monthly OHLC Data (Full View)")

st.dataframe(
    df_monthly[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].tail(24),
    use_container_width=True
)

st.subheader("📊 Monthly Indicators")

st.dataframe(
    df_monthly[['Date', 'Close', 'MA3', 'MA6']].tail(24),
    use_container_width=True
)

# =========================
# 💰 BACKTESTING ENGINE (NEW 🔥)
# =========================
st.subheader("💰 Strategy Backtesting")

# Prepare dataset
df_bt = df.copy()

# Use same features
features = ['Close', 'MA20', 'RSI', 'MACD']

# Generate predictions
df_bt['Prediction'] = model.predict(df_bt[features])

# Strategy logic:
# 1 = Buy (long)
# 0 = Sell (exit / no position)
df_bt['Position'] = df_bt['Prediction']

# Calculate returns
df_bt['Returns'] = df_bt['Close'].pct_change()

# Strategy returns (shift to avoid lookahead bias)
df_bt['Strategy_Returns'] = df_bt['Position'].shift(1) * df_bt['Returns']

# Equity curve
initial_capital = 1000
df_bt['Equity'] = (1 + df_bt['Strategy_Returns']).cumprod() * initial_capital

# =========================
# 📊 METRICS
# =========================
total_return = (df_bt['Equity'].iloc[-1] / initial_capital - 1) * 100

win_trades = df_bt[df_bt['Strategy_Returns'] > 0]
total_trades = df_bt[df_bt['Strategy_Returns'] != 0]

win_rate = (len(win_trades) / len(total_trades)) * 100 if len(total_trades) > 0 else 0

# Max Drawdown
rolling_max = df_bt['Equity'].cummax()
drawdown = (df_bt['Equity'] - rolling_max) / rolling_max
max_drawdown = drawdown.min() * 100

# =========================
# 📊 DISPLAY METRICS
# =========================
col1, col2, col3 = st.columns(3)

col1.metric("Total Return %", f"{total_return:.2f}%")
col2.metric("Win Rate %", f"{win_rate:.2f}%")
col3.metric("Max Drawdown %", f"{max_drawdown:.2f}%")

st.subheader("📈 Equity Curve")

fig_bt = go.Figure()

fig_bt.add_trace(go.Scatter(
    x=df_bt['Date'],
    y=df_bt['Equity'],
    name="Strategy Equity",
    line=dict(color='green', width=2)
))

st.plotly_chart(fig_bt, use_container_width=True)


st.subheader("📊 Strategy vs Buy & Hold")

df_bt['BuyHold'] = (1 + df_bt['Returns']).cumprod() * initial_capital

fig_compare = go.Figure()

fig_compare.add_trace(go.Scatter(
    x=df_bt['Date'],
    y=df_bt['Equity'],
    name="AI Strategy",
    line=dict(color='green')
))

fig_compare.add_trace(go.Scatter(
    x=df_bt['Date'],
    y=df_bt['BuyHold'],
    name="Buy & Hold",
    line=dict(color='blue')
))

st.plotly_chart(fig_compare, use_container_width=True)

# =========================
# 🔥 ADD LOGISTIC REGRESSION MODEL (NEW)
# =========================
@st.cache_resource
def train_logistic_model(df):
    df = df.copy()

    df['Target'] = (df['Close'].shift(-1) > df['Close']).astype(int)
    df.dropna(inplace=True)

    features = ['Close', 'MA20', 'RSI', 'MACD']
    X = df[features]
    y = df['Target']

    log_model = LogisticRegression()
    log_model.fit(X, y)

    return log_model

# Train Logistic model
log_model = train_logistic_model(df)

# =========================
# 🔮 LOGISTIC PREDICTION (FIXED ✅)
# =========================

log_features = np.array([
    latest['Close'],
    latest['MA20'],
    latest['RSI'],
    latest['MACD']
]).reshape(1, -1)

log_prediction = log_model.predict(log_features)[0]
log_proba = log_model.predict_proba(log_features)[0]
log_confidence = round(max(log_proba) * 100, 2)