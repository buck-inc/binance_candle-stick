import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import requests
from sklearn.linear_model import LinearRegression
import numpy as np

st.set_page_config(page_title="Prediksi Harga Bitcoin", layout="wide")
st.title("📈 Prediksi Harga Bitcoin (Data Binance Real-time)")

@st.cache_data(ttl=60)
def fetch_data():
    url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": "BTCUSDT",
        "interval": "1m",
        "limit": 100
    }
    response = requests.get(url, timeout=10)
    data = response.json()
    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore"
    ])
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df

try:
    df = fetch_data()
    df = df.tail(5).reset_index(drop=True)
    df["target"] = df["close"].shift(-1)
    st.subheader("🧾 Data Terbaru (5 baris)")
    st.dataframe(df)

    # Candlestick chart
    fig = go.Figure(data=[go.Candlestick(
        x=df['timestamp'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        increasing_line_color='green',
        decreasing_line_color='red'
    )])
    fig.update_layout(title="Grafik Candlestick Harga BTC/USDT",
                      xaxis_title="Waktu", yaxis_title="Harga",
                      xaxis_rangeslider_visible=False,
                      template="plotly_dark")
    st.plotly_chart(fig, use_container_width=True)

    # Prediksi
    model = LinearRegression()
    features = ["open", "high", "low", "close", "volume"]
    df.dropna(inplace=True)
    X = df[features]
    y = df["target"]
    model.fit(X, y)
    pred = model.predict([df[features].iloc[-1]])[0]
    st.success(f"💹 Prediksi Harga Selanjutnya: ${pred:,.2f}")

    # Simpan data ke CSV
    df.to_csv("data_binance_log.csv", index=False)

except Exception as e:
    st.error(f"Gagal mengambil data: {e}")
