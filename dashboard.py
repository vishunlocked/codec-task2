# dashboard.py

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.graph_objs as go
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from streamlit_autorefresh import st_autorefresh

@st.cache_data(ttl=3600)
def download_stock_data(symbol, period="7d", interval="15m"):
    data = yf.download(symbol, period=period, interval=interval)
    return data

@st.cache_resource
def train_lstm_model(X, y, epochs=3):
    model = Sequential([
        LSTM(50, return_sequences=False, input_shape=(X.shape[1], 1)),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X, y, epochs=epochs, batch_size=1, verbose=0)
    return model

def run_dashboard():
    # --- Header ---
    col1, col2 = st.columns([6, 1])
    with col1:
        st.title("📈 Real-Time Stock Market Dashboard")

    # --- Stock Input ---
    if "stock_symbol" not in st.session_state:
        st.session_state.stock_symbol = "AAPL"

    def on_change_symbol():
        st.session_state.stock_symbol = st.session_state.input_symbol.strip().upper()

    stock_symbol = st.text_input(
        "Enter Stock Symbol (e.g., AAPL, TSLA):",
        value=st.session_state.stock_symbol,
        key="input_symbol",
        on_change=on_change_symbol
    )

    stock_symbol = stock_symbol.strip().upper()

    if not stock_symbol:
        st.warning("Please enter a valid stock symbol.")
        st.stop()

    if st.checkbox("🔄 Auto-refresh every 60 seconds"):
        st_autorefresh(interval=60_000, key="refresh")

    # --- Download Data ---
    stock_data = download_stock_data(stock_symbol)

    if stock_data.empty:
        st.error(f"❌ No data found for symbol '{stock_symbol}'.")
        st.stop()

    stock_data.index = pd.to_datetime(stock_data.index)
    latest = stock_data.iloc[-1]

    # --- Metrics ---
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Price", f"${float(latest['Close']):.2f}")
    col2.metric("Day High", f"${float(stock_data['High'].max()):.2f}")
    col3.metric("Day Low", f"${float(stock_data['Low'].min()):.2f}")

    # --- Quick View ---
    st.markdown("### 📊 Quick View: Stocks")
    stocks = {"AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Google"}
    cols = st.columns(len(stocks))

    for col, (symbol, name) in zip(cols, stocks.items()):
        data = download_stock_data(symbol)
        if data.empty:
            col.markdown(f"### {name}\n_No data available_")
            continue

        close_prices = data['Close'].tail(20).values
        times = data.index[-20:]
        col.markdown(f"#### {name}")
        col.metric("Price", f"${float(close_prices[-1]):.2f}")
        fig = go.Figure(go.Scatter(
            x=times, y=close_prices,
            mode="lines", line=dict(color="#22c55e")
        ))
        fig.update_layout(height=120, margin=dict(l=0, r=0, t=0, b=0),
                          xaxis=dict(visible=False), yaxis=dict(visible=False),
                          plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
        col.plotly_chart(fig, use_container_width=True)

    # --- Portfolio Performance (Mock) ---
    st.markdown("### 📈 Performance")
    perf_fig = go.Figure()
    days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
    values_2022 = [100, 150, 120, 180, 160, 200, 190]
    values_2023 = [140, 170, 130, 210, 180, 220, 200]

    perf_fig.add_trace(go.Bar(name='2022', x=days, y=values_2022, marker_color='gray'))
    perf_fig.add_trace(go.Bar(name='2023', x=days, y=values_2023, marker_color='limegreen'))
    perf_fig.update_layout(barmode='group', height=300)
    st.plotly_chart(perf_fig, use_container_width=True)

    # --- Dividend Chart ---
    st.markdown("### Dividend")

    months = ["Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    dividend_values = [8000, 12000, 9000, 25254, 15000, 18000]
    highlight_idx = 3  # October highlighted

    dividend_trace = go.Scatter(
        x=months,
        y=dividend_values,
        mode='lines+markers',
        line=dict(color='black', width=2, shape='spline'),
        marker=dict(
            size=10,
            color='white',
            line=dict(color='black', width=2),
            symbol='circle'
        ),
        hovertemplate='%{x}: $%{y:,}<extra></extra>'
    )

    highlight_trace = go.Scatter(
        x=[months[highlight_idx]],
        y=[dividend_values[highlight_idx]],
        mode='markers+text',
        marker=dict(
            size=16,
            color='lightgreen',
            line=dict(color='darkgreen', width=2),
            symbol='circle'
        ),
        text=[f"${dividend_values[highlight_idx]:,}"],
        textposition='top center',
        hovertemplate='%{x}: $%{y:,}<extra></extra>'
    )

    dividend_fig = go.Figure(data=[dividend_trace, highlight_trace])

    dividend_fig.update_layout(
        title=dict(text="Monthly Dividend Payouts", x=0.05, xanchor='left', font=dict(size=18, color='black')),
        xaxis=dict(
            title="Month",
            showgrid=True,
            gridcolor='lightgray',
            zeroline=False,
            tickmode='array',
            tickvals=months,
            tickfont=dict(size=12),
        ),
        yaxis=dict(
            title="Dividend Amount (USD)",
            showgrid=True,
            gridcolor='lightgray',
            zeroline=False,
            tickprefix="$",
            tickformat=",",
            tickfont=dict(size=12),
        ),
        plot_bgcolor='white',
        margin=dict(l=40, r=20, t=40, b=30),
        hovermode='x unified',
        showlegend=False,
        height=350
    )

    st.plotly_chart(dividend_fig, use_container_width=True)

    # --- LSTM Multi-Step Forecast ---
    st.markdown(f"### 📈 {stock_symbol} Price Forecast (Next 10 Intervals)")

    look_back = 10
    future_steps = 10  # How many intervals to predict

    if len(stock_data) >= look_back + 1:
        close_data = stock_data[['Close']].values
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(close_data)

        X, y = [], []
        for i in range(look_back, len(scaled_data)):
            X.append(scaled_data[i - look_back:i, 0])
            y.append(scaled_data[i, 0])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))

        model = train_lstm_model(X, y, epochs=3)

        # --- Forecast Future Intervals ---
        forecast_input = scaled_data[-look_back:].reshape(1, look_back, 1)
        predictions = []

        for _ in range(future_steps):
            pred_scaled = model.predict(forecast_input, verbose=0)
            predictions.append(pred_scaled[0, 0])
            forecast_input = np.append(forecast_input[:, 1:, :], [[[pred_scaled[0, 0]]]], axis=1)

        predictions = np.array(predictions).reshape(-1, 1)
        predicted_prices = scaler.inverse_transform(predictions).flatten()

        # --- Time Axis for Forecast ---
        last_date = stock_data.index[-1]
        future_dates = [last_date + pd.Timedelta(minutes=15 * i) for i in range(1, future_steps + 1)]

        st.metric("Next Predicted Price", f"${predicted_prices[0]:.2f}")

        # --- Forecast Chart ---
        forecast_fig = go.Figure()

        forecast_fig.add_trace(go.Scatter(
            x=stock_data.index,
            y=stock_data['Close'],
            mode='lines+markers',
            name="Actual",
            line=dict(color='blue', width=3),
            marker=dict(size=6, color='blue'),
            hoverinfo='x+y'
        ))

        forecast_fig.add_trace(go.Scatter(
            x=future_dates,
            y=predicted_prices,
            mode='lines+markers',
            name="Forecast",
            line=dict(dash='dash', color='orange', width=3),
            marker=dict(symbol='circle', size=8, color='orange', line=dict(width=2, color='darkorange')),
            hoverinfo='x+y'
        ))

        forecast_fig.update_layout(
            title=f"{stock_symbol} Price Forecast (Next {future_steps} Intervals)",
            height=400,
            xaxis_title="Time",
            yaxis_title="Price",
            template='plotly_dark',
            hovermode='x unified',
        )

        st.plotly_chart(forecast_fig, use_container_width=True)

    # --- Data Table ---
    st.markdown("### 📄 Recent Data")
    st.dataframe(stock_data.tail(10), use_container_width=True)
