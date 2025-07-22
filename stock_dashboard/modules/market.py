import streamlit as st

def show_market():
    st.title("📊 Market Overview")
    st.write("Insights into today's market movements and major indices.")
    st.success("Top Gainers: AAPL, MSFT, GOOGL")
    st.warning("Top Losers: TSLA, NFLX, AMD")
