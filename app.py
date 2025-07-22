import streamlit as st
from streamlit_option_menu import option_menu

# --- Page Config ---
st.set_page_config(page_title="Stock Dashboard", layout="wide")

# --- Load Custom CSS ---
try:
    with open("static/style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    st.warning("⚠️ style.css not found. Using default theme.")

# --- Sidebar Navigation ---
with st.sidebar:
    selected = option_menu(
        menu_title="Layers",
        options=["Dashboard", "Market", "Exchange", "Wallet", "Blog & News"],
        icons=["speedometer", "graph-up", "repeat", "wallet", "newspaper"],
        menu_icon="layers",
        default_index=0,
    )
    st.markdown("---")
    st.text("🧑🏻‍💼 V I S H N U\nAdmin")

# --- Page Routing ---
if selected == "Dashboard":
    from dashboard import run_dashboard
    run_dashboard()

elif selected == "Market":
    from modules.market import show_market
    show_market()

elif selected == "Exchange":
    from modules.exchange import show_exchange
    show_exchange()

elif selected == "Wallet":
    from modules.wallet import show_wallet
    show_wallet()

elif selected == "Blog & News":
    from modules.blog_news import show_blog_news
    show_blog_news()
