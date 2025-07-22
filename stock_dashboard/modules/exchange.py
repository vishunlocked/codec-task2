import streamlit as st

def show_exchange():
    st.title("🔄 Exchange Activity")
    st.write("Details of stock exchanges and volume stats.")
    st.metric("NYSE Volume", "1.5B Shares")
    st.metric("NASDAQ Volume", "2.1B Shares")
