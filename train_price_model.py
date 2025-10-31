# --- FIX FOR MACOS MUTEX ERROR ---
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import save_model
# --- END OF FIX ---

import yfinance as yf
import pandas as pd
import pandas_ta as ta
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from newsapi import NewsApiClient
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from datetime import datetime, timedelta

# --- 1. NEW: Sentiment Analysis Functions ---

def get_news_sentiment(api_key, start_date, end_date):
    """
    Fetches news from NewsAPI and calculates daily average sentiment.
    """
    newsapi = NewsApiClient(api_key=api_key)
    sia = SentimentIntensityAnalyzer()
    
    sentiment_scores = {}
    
    print("Fetching news for sentiment...")
    try:
        all_articles = newsapi.get_everything(
            q='NIFTY OR "Indian economy" OR "Reserve Bank of India"',
            language='en',
            sort_by='publishedAt',
            page_size=100
        )
    except Exception as e:
        print(f"NewsAPI Error: {e}")
        print("Using 0.0 for sentiment. To fix, check your API key.")
        return pd.DataFrame(index=pd.date_range(start=start_date, end=end_date), columns=['Sentiment']).fillna(0.0)

    daily_scores = {}
    for article in all_articles['articles']:
        date = article['publishedAt'][:10] # Get YYYY-MM-DD
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
    
    historical_avg_sentiment = sentiment_df['Sentiment'].mean()
    if pd.isna(historical_avg_sentiment):
        historical_avg_sentiment = 0.0 # Default to neutral

    print(f"Using historical average sentiment of: {historical_avg_sentiment:.3f}")
    
    full_date_range = pd.date_range(start=start_date, end=end_date)
    final_sentiment_df = pd.DataFrame(index=full_date_range, columns=['Sentiment'])
    final_sentiment_df.update(sentiment_df)
    final_sentiment_df['Sentiment'] = final_sentiment_df['Sentiment'].fillna(historical_avg_sentiment)
    
    return final_sentiment_df

# --- 2. Standard Helper Functions ---

def create_sequences(data, target, time_steps=60):
    Xs, ys = [], []
    for i in range(len(data) - time_steps):
        v = data[i:(i + time_steps)]
        Xs.append(v)
        ys.append(target[i + time_steps])
    return np.array(Xs), np.array(ys)

# --- 3. Main Training Function ---
def train_model():
    NEWS_API_KEY = "fd47ec806a054ff2aeb3a3b5adb57b42"

    print("Starting REGRESSION model training (v2 with Sentiment)...")
    
    start_date = '2015-01-01'
    end_date = '2025-10-01'

    # 1. Get Price Data
    print("Fetching price data for ^NSEBANK...")
    df = yf.download('^NSEBANK', start=start_date, end=end_date)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # --- THIS IS THE FIX ---
    # Convert 'df' index to timezone-NAIVE before merging
    df.index = df.index.tz_localize(None)
    # -----------------------

    # 2. Get Sentiment Data
    sentiment_df = get_news_sentiment(NEWS_API_KEY, start_date=start_date, end_date=end_date)
    
    # 3. Merge Price and Sentiment
    print("Merging price and sentiment data...")
    df = pd.merge(df, sentiment_df, left_index=True, right_index=True, how='left')
    
    # 4. Add Technical Features
    print("Calculating technical features...")
    df['SMA_20'] = ta.sma(df['Close'], length=20)
    df['SMA_50'] = ta.sma(df['Close'], length=50)
    df['RSI'] = ta.rsi(df['Close'], length=14)
    macd = ta.macd(df['Close'], fast=12, slow=26, signal=9)
    if macd is not None and not macd.empty:
        df['MACD'] = macd['MACD_12_26_9']
        df['MACD_signal'] = macd['MACDs_12_26_9']
        df['MACD_hist'] = macd['MACDh_12_26_9']

    # 5. Create the Target and Clean Data
    df['Target'] = df['Close'].shift(-1)
    df = df.dropna()

    # 6. Define Features (X) and Target (y)
    target_col = 'Target'
    feature_cols = ['Close', 'SMA_20', 'SMA_50', 'RSI', 'MACD', 'MACD_signal', 'MACD_hist', 'Sentiment']
    
    X = df[feature_cols]
    y = df[[target_col]]

    # 7. Scale ALL data
    print("Scaling data...")
    x_scaler = MinMaxScaler(feature_range=(0, 1))
    y_scaler = MinMaxScaler(feature_range=(0, 1))
    X_scaled = x_scaler.fit_transform(X)
    y_scaled = y_scaler.fit_transform(y)

    # 8. Split and Create Sequences
    split_index = int(len(X_scaled) * 0.8)
    X_train, X_test = X_scaled[:split_index], X_scaled[split_index:]
    y_train, y_test = y_scaled[:split_index], y_scaled[split_index:]
    TIME_STEPS = 60
    X_train_seq, y_train_seq = create_sequences(X_train, y_train, TIME_STEPS)
    X_test_seq, y_test_seq = create_sequences(X_test, y_test, TIME_STEPS)

    if X_train_seq.shape[0] == 0:
        print("Not enough data to create LSTM sequences.")
        return

    # 9. Build Keras Regression Model
    print("Building Keras LSTM Regression Model...")
    lstm_model = Sequential()
    lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(TIME_STEPS, len(feature_cols))))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(LSTM(units=50, return_sequences=False))
    lstm_model.add(Dropout(0.2))
    lstm_model.add(Dense(units=25))
    lstm_model.add(Dense(units=1)) 
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')

    # 10. Train the model
    print("Training LSTM... this may take a few minutes.")
    lstm_model.fit(X_train_seq, y_train_seq, epochs=25, batch_size=32, validation_split=0.1, shuffle=False)

    # 11. Save the new model AND the scalers
    print("Saving models...")
    save_model(lstm_model, "price_model.keras") 
    joblib.dump(x_scaler, "x_scaler.joblib")
    joblib.dump(y_scaler, "y_scaler.joblib")

    print("\n--- REGRESSION model (v2 with Sentiment) training complete. ---")

if __name__ == "__main__":
    train_model()