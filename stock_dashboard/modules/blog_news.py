import streamlit as st

def show_blog_news():
    st.title("📰 Blog & News")
    st.write("Latest market headlines and blog insights.")
    st.markdown("""
    - [Bloomberg: Market Updates](https://www.bloomberg.com/)
    - [Reuters: Global Economy](https://www.reuters.com/)
    - [CNBC: Stock News](https://www.cnbc.com/)
    """)
