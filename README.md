# Real-Time Stock Market Dashboard

A dynamic and interactive dashboard that tracks and visualizes live stock market data in real-time.

## Description

This project is a real-time stock market dashboard built with Python. It fetches live stock market data from an API, processes it, and displays interactive visualizations and key financial indicators for selected stocks. The dashboard enables users to monitor market trends and stock performance instantly.

## Features

- Real-time data fetching and updates for selected stocks
- Interactive graphs and charts using Plotly
- Key financial indicators and metrics display
- User-friendly interface built with Streamlit
- Easy selection of stocks to track

## Technologies Used

- **Python**: Core programming language  
- **Pandas**: Data processing and manipulation  
- **Plotly**: Interactive data visualization  
- **Requests**: API calls to fetch live stock data  
- **Streamlit**: Web app framework for creating the dashboard UI

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/vishunlocked/codec-task2.git
   cd codec-task2
2. Create and activate a virtual environment (optional but recommended):

python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install required dependencies:

pip install -r requirements.txt

Usage
1. Run the Streamlit app:
   streamlit run app.py
2.Open the provided local URL in your browser (usually http://localhost:8501).

3.Select one or more stocks to view their real-time graphs and financial indicators.

API and Data Source
The dashboard fetches live stock data using a public API through the requests library. Ensure you have internet access for live updates.


