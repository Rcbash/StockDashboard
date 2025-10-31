# --- FIX FOR MACOS MUTEX ERROR ---
from tensorflow.keras.models import load_model
# --- END OF FIX ---

import streamlit as st
import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import joblib
import plotly.graph_objects as go
import os
import streamlit.components.v1 as components
from newsapi import NewsApiClient
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime, timedelta

# --- 1. Custom CSS Styling ---
st.markdown("""
<style>
    /* ... (Your CSS code is unchanged) ... */
</style>
""", unsafe_allow_html=True)


# --- 2. Load Models and Scalers ---
@st.cache_resource
def load_all_models():
    """Loads all trained models and scalers from disk."""
    print("Loading REGRESSION models...")
    try:
        lstm_model = load_model("price_model.keras", compile=False) 
        x_scaler = joblib.load("x_scaler.joblib")
        y_scaler = joblib.load("y_scaler.joblib")
        print("Models loaded successfully.")
        return lstm_model, x_scaler, y_scaler
    except FileNotFoundError:
        st.error("ERROR: Model files not found. Please run 'train_price_model.py' first in your terminal.")
        return None, None, None
    except Exception as e:
        st.error(f"Error loading models: {e}. You may need to retrain.")
        return None, None, None

lstm_model, x_scaler, y_scaler = load_all_models()

# --- 3. NEW: Live Sentiment Function ---
@st.cache_data(ttl=900) # Cache for 15 minutes
def get_live_news_sentiment(api_key, query):
    """Fetches news for the last 30 days and calculates sentiment."""
    newsapi = NewsApiClient(api_key=api_key)
    sia = SentimentIntensityAnalyzer()
    
    try:
        all_articles = newsapi.get_everything(
            q=query,
            language='en',
            sort_by='publishedAt',
            page_size=100
        )
    except Exception as e:
        print(f"NewsAPI Error: {e}")
        st.warning("Could not fetch live news sentiment. Prediction will be based on price only.")
        return pd.DataFrame(columns=['Sentiment'])

    daily_scores = {}
    for article in all_articles['articles']:
        date = article['publishedAt'][:10]
        if date not in daily_scores:
            daily_scores[date] = []
        score = sia.polarity_scores(article['title'] or '')['compound']
        if score != 0:
            daily_scores[date].append(score)

    avg_scores = {}
    for date, scores in daily_scores.items():
        if scores:
            avg_scores[date] = sum(scores) / len(scores)

    sentiment_df = pd.DataFrame.from_dict(avg_scores, orient='index', columns=['Sentiment'])
    sentiment_df.index = pd.to_datetime(sentiment_df.index)
    return sentiment_df

# --- 4. Data Caching ---
@st.cache_data(ttl=900)
def get_stock_data(ticker, api_key):
    """Fetches all YFinance data and merges live sentiment."""
    print(f"Fetching new data for {ticker}")
    stock = yf.Ticker(ticker)
    info = stock.info
    df = stock.history(period="1y", interval="1d")
    financials = stock.financials
    
    if df.empty:
        st.error(f"Could not fetch price data for '{ticker}'. Is it a valid ticker?")
        return None, None, None, None
    
    # --- THIS IS THE FIX ---
    # Convert 'df' index to timezone-NAIVE before merging
    df.index = df.index.tz_localize(None)
    # -----------------------
    
    # --- NEW: Get and merge live sentiment ---
    news_query = info.get('longName', ticker) 
    sentiment_df = get_live_news_sentiment(api_key, news_query)
    
    if not sentiment_df.empty:
        df = pd.merge(df, sentiment_df, left_index=True, right_index=True, how='left')
        df['Sentiment'] = df['Sentiment'].ffill().fillna(0.0) 
    else:
        df['Sentiment'] = 0.0
        
    return info, df, financials, sentiment_df.tail(1).iloc[0,0] if not sentiment_df.empty else 0.0

# --- 5. Charting & Prediction Functions ---
def create_groww_chart(df):
    fig = go.Figure()
    fig.add_trace(go.Candlestick(x=df.index,
                    open=df['Open'], high=df['High'],
                    low=df['Low'], close=df['Close'], name='Price'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], line=dict(color='yellow', width=1), name='SMA 20'))
    fig.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], line=dict(color='orange', width=1), name='SMA 50'))
    fig.update_layout(title='Price History (Candlestick Chart)', yaxis_title='Stock Price', xaxis_rangeslider_visible=False,
                      plot_bgcolor='#0E1117', paper_bgcolor='#0E1117', font_color='white')
    return fig

def create_sentiment_gauge(signal_value, title="Model Signal"):
    gauge_value = (signal_value + 1) * 50
    fig = go.Figure(go.Indicator(
        mode = "gauge+number", value = gauge_value,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': title, 'font': {'size': 20}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "rgba(0,0,0,0)"},
            'steps' : [
                 {'range': [0, 40], 'color': '#FF4B4B'}, # Red (Negative)
                 {'range': [40, 60], 'color': 'gray'},  # Gray (Neutral)
                 {'range': [60, 100], 'color': '#28A745'}], # Green (Positive)
            'threshold' : {'line': {'color': "white", 'width': 4}, 'thickness': 0.75, 'value': 50}
        }))
    fig.update_layout(paper_bgcolor = "#0E1117", font = {'color': "white", 'family': "Arial"})
    return fig

def get_prediction(df):
    """Uses the loaded model to make a price prediction."""
    if lstm_model is None or x_scaler is None or y_scaler is None:
        return None
    
    feature_cols = ['Close', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist', 'Sentiment']
    TIME_STEPS = 60
    
    if len(df) < TIME_STEPS:
        st.error(f"Need at least {TIME_STEPS} days of data, only have {len(df)}.")
        return None
        
    last_60_days_features = df[feature_cols].tail(TIME_STEPS)
    scaled_features = x_scaler.transform(last_60_days_features)
    live_sequence = np.array([scaled_features])
    
    predicted_price_scaled = lstm_model.predict(live_sequence)[0][0]
    predicted_price = y_scaler.inverse_transform([[predicted_price_scaled]])[0][0]
    return predicted_price

# =======================================================
# --- Streamlit UI (The Website) ---
# =======================================================
st.title("📈 Real-Time Stock Analysis & Forecast Dashboard")

# --- The Search Bar ---
ticker_input = st.text_input("Enter Stock Ticker (e.g., AAPL, RELIANCE.NS)", value="RELIANCE.NS")

# --- I have added your API key here ---
NEWS_API_KEY = "fd47ec806a054ff2aeb3a3b5adb57b42"
# ----------------------------------------

if st.button(f"Analyze {ticker_input}"):
    st.warning("**Disclaimer:** This is an educational project. Model predictions are not financial advice. **Do not trade based on this data.**", icon="⚠️")
    
    try:
        info, df, financials, latest_sentiment = get_stock_data(ticker_input, NEWS_API_KEY)
    except Exception as e:
        st.error(f"Error fetching data: {e}")
        st.stop()
    if info is None:
        st.stop()

    # --- Process Data for Charting and Prediction ---
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df['SMA_20'] = ta.sma(df['Close'], length=20)
    df['SMA_50'] = ta.sma(df['Close'], length=50)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        df['MACD'] = macd['MACD_12_26_9']
        df['MACD_signal'] = macd['MACDs_12_26_9']
        df['MACD_hist'] = macd['MACDh_12_26_9']
    df_pred = df.dropna()

    # --- Create Tabs ---
    tab1, tab2, tab3 = st.tabs(["📊 **Forecast & Chart**", "🏢 **Company Profile**", "💰 **Financials & Backtest**"])

    # --- TAB 1: Forecast & Chart ---
    with tab1:
        st.header(f"Price Forecast for {info.get('longName', ticker_input)}")
        predicted_price = get_prediction(df_pred)
        col1, col2 = st.columns([1, 1])
        with col1:
            if predicted_price is not None:
                last_close_price = df['Close'].iloc[-1]
                delta = predicted_price - last_close_price
                st.metric(label="Predicted Next Day Close",
                          value=f"{predicted_price:,.2f}",
                          delta=f"{delta:,.2f} ({delta/last_close_price*100:.2f}%)")
                
                signal_value = 80 if delta > 0 else 20
                st.plotly_chart(create_sentiment_gauge(signal_value, title="Model Price Signal"), use_container_width=True)
            else:
                st.error("Could not generate prediction.")
        
        with col2:
            st.subheader("Live News Sentiment")
            st.plotly_chart(create_sentiment_gauge(latest_sentiment, title="Today's News Sentiment"), use_container_width=True)
            st.caption("Based on 'compound' sentiment score of news from the last 24 hours.")

        st.header(f"Interactive Chart for {ticker_input}")
        fig = create_groww_chart(df.tail(180))
        st.plotly_chart(fig, use_container_width=True)

    # --- TAB 2: Company Profile ---
    with tab2:
        st.header(f"Profile for {info.get('longName', ticker_input)}")
        col1, col2 = st.columns([1, 4])
        logo_url = info.get('logo_url', '')
        with col1:
            if logo_url: st.image(logo_url, width=150)
            else: st.write("No Logo Found")
        with col2:
            st.write(f"**Sector:** {info.get('sector', 'N/A')}")
            st.write(f"**Industry:** {info.get('industry', 'N/A')}")
            st.write(f"**Website:** {info.get('website', 'N/A')}")
            st.write(f"**Employees:** {info.get('fullTimeEmployees', 'N/A')}")
        st.info(info.get('longBusinessSummary', 'No summary available.'))
        st.subheader("Key Statistics")
        stats_col1, stats_col2, stats_col3 = st.columns(3)
        with stats_col1:
            st.metric("Market Cap", f"{info.get('marketCap', 0):,}")
            st.metric("P/E Ratio", f"{info.get('trailingPE', 0):.2f}")
        with stats_col2:
            st.metric("52-Week High", f"{info.get('fiftyTwoWeekHigh', 0):.2f}")
            st.metric("52-Week Low", f"{info.get('fiftyTwoWeekLow', 0):.2f}")
        with stats_col3:
            st.metric("Avg. Volume", f"{info.get('averageVolume', 0):,}")
            st.metric("Dividend Yield", f"{info.get('dividendYield', 0) * 100:.2f}%")

    # --- TAB 3: Financials & Backtest ---
    with tab3:
        st.header("Yearly Profits (Net Income)")
        if financials is not None and not financials.empty and 'Net Income' in financials.index:
            profits = financials.loc['Net Income']
            profits.index = profits.index.strftime('%Y')
            st.bar_chart(profits)
        else:
            st.write("Yearly profit data (Net Income) is not available for this asset.")
        with st.expander("Show Raw Financials Table"):
            st.dataframe(financials)
        
        st.header("Strategy Backtest Report")
        backtest_file = "backtest_report.html"
        if os.path.exists(backtest_file):
            st.success("A backtest report was found! This shows how a simple SMA Crossover strategy would have performed on our training data.")
            with open(backtest_file, "r", encoding='utf-8') as f:
                html_content = f.read()
            components.html(html_content, height=600, scrolling=True)
            st.download_button(label="Download Full Report", data=html_content, file_name="backtest_report.html", mime="text/html")
        else:
            st.info("No backtest report found. Run `python3 backtest.py` in your terminal to generate one.")