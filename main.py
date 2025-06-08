import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import os
from datetime import datetime, timedelta
import logging
from keras.models import load_model  # type: ignore
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from statsmodels.tsa.arima.model import ARIMA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from newsapi.newsapi_client import NewsApiClient
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import pytz

# ---------------- SETUP ----------------
st.set_page_config(page_title="Stock Price Predictor", page_icon="📈", layout="wide")
st.title("📈 Stock Price Predictor")
PREDICTIONS_FILE = "predictions.csv"
STOCK_DATA_FILE = "stock_data.csv"
MODEL_PATH = "Latest_stock_price_model.keras"
NEWS_API_KEY = "d7fcd0aaf8024281854dea5350623f64"
DEFAULT_SENTIMENT_MODEL = "ProsusAI/finbert"
keras_available = False
model = None
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(_name_)

# ---------------- SENTIMENT ANALYSIS FUNCTIONS ----------------
@st.cache_resource
def get_sentiment_pipeline(model_name=DEFAULT_SENTIMENT_MODEL):
    """Load and cache FinBERT sentiment model."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        pipeline_instance = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        logger.info(f"Loaded sentiment model: {model_name}")
        return pipeline_instance
    except Exception as e:
        logger.error(f"Failed to load sentiment model {model_name}: {e}")
        st.error(f"⚠ Failed to load sentiment model: {e}. Using neutral sentiment.")
        return None

sentiment_pipeline = get_sentiment_pipeline()

def analyze_sentiment_for_ticker(articles, sentiment_pipeline, df):
    """Analyze sentiment of news articles and align with stock data."""
    if not articles or not sentiment_pipeline or df is None or df.empty:
        logger.warning("No articles, pipeline, or stock data provided. Using neutral sentiment.")
        return 0.0, [], pd.Series(0.0, index=df.index)
    try:
        analyzed_details = []
        scores = []
        news_df = pd.DataFrame(articles)
        news_df['publishedAt'] = pd.to_datetime(news_df['publishedAt'], errors='coerce', utc=True)
        news_df['text'] = news_df['title'].fillna('') + ' ' + news_df['description'].fillna('')
        for _, article in news_df.iterrows():
            text = article['text']
            if not text.strip():
                continue
            result = sentiment_pipeline(text, truncation=True, max_length=512)[0]
            label = result['label'].lower()
            score = result['score']
            sentiment_score = score if label == 'positive' else -score if label == 'negative' else 0.0
            analyzed_details.append({
                'headline': article.get('title', ''),
                'url': article.get('url', ''),
                'label': label,
                'score': sentiment_score,
                'publishedAt': article.get('publishedAt')
            })
            scores.append(sentiment_score)
        aggregated_score = np.mean(scores) if scores else 0.0
        # Aggregate daily sentiment with recency weighting
        news_df['sentiment_score'] = [detail['score'] for detail in analyzed_details]
        news_df['source_weight'] = news_df['source'].apply(
            lambda x: 1.0 if x.get('name') in ["Reuters", "Bloomberg", "CNBC"] else 0.8
        )
        current_time = datetime.now(pytz.UTC)
        news_df['days_old'] = (current_time - news_df['publishedAt']).dt.total_seconds() / (24 * 3600)
        news_df['recency_weight'] = np.exp(-news_df['days_old'].fillna(7.0) / 7)
        news_df['weighted_sentiment'] = news_df['sentiment_score'] * news_df['source_weight'] * news_df['recency_weight']
        daily_sentiment = news_df.groupby(news_df['publishedAt'].dt.date)['weighted_sentiment'].mean().reset_index()
        daily_sentiment['publishedAt'] = pd.to_datetime(daily_sentiment['publishedAt'])
        sentiment_series = pd.Series(index=df.index, dtype=float)
        for date in df.index:
            date_sentiment = daily_sentiment[
                daily_sentiment['publishedAt'].dt.date == date.date()
            ]['weighted_sentiment']
            sentiment_series[date] = date_sentiment.mean() if not date_sentiment.empty else 0.0
        sentiment_series = sentiment_series.fillna(0.0)
        logger.info(f"Sentiment series stats: mean={sentiment_series.mean():.4f}, std={sentiment_series.std():.4f}")
        return aggregated_score, analyzed_details, sentiment_series
    except Exception as e:
        logger.error(f"Error analyzing sentiment: {e}")
        st.warning(f"⚠ Error processing sentiment data: {e}. Using neutral sentiment.")
        return 0.0, [], pd.Series(0.0, index=df.index)

def get_suggestion(score):
    """Generate investment suggestion based on sentiment score."""
    if score > 0.3:
        return "Strong Buy"
    elif score > 0.1:
        return "Buy"
    elif score < -0.3:
        return "Strong Sell"
    elif score < -0.1:
        return "Sell"
    else:
        return "Hold"

def get_validation_points(analyzed_details):
    """Generate validation points from news sentiment analysis."""
    points = []
    for item in analyzed_details[:5]:
        emoji = "🟢" if item['label'] == 'positive' else "🔴" if item['label'] == 'negative' else "⚪"
        points.append(
            f"{emoji} {item['headline']} "
            f"(<a href='{item['url']}' target='_blank'>Source</a>, Sentiment: {item['score']:.3f})"
        )
    return points

@st.cache_data(ttl=3600)
def get_us_news(query):
    """Fetch U.S. news articles using NewsAPI."""
    newsapi = NewsApiClient(api_key=NEWS_API_KEY)
    try:
        from_date = (datetime.now(pytz.UTC) - timedelta(days=7)).strftime('%Y-%m-%d')
        articles = newsapi.get_everything(
            q=query,
            from_param=from_date,
            language='en',
            sort_by='relevancy',
            page_size=50
        )['articles']
        logger.info(f"Fetched {len(articles)} articles for query: {query}")
        return articles
    except Exception as e:
        logger.error(f"Error fetching news for {query}: {e}")
        return []

def fetch_news(ticker, company_name=None):
    """Fetch news for ticker, using company name for better coverage."""
    query = f"{ticker}"
    if company_name:
        query += f" OR \"{company_name}\""
    with st.spinner(f"Fetching news for {ticker}..."):
        articles = get_us_news(query)
    if not articles:
        st.warning(f"⚠ No recent news found for {ticker}.")
    return articles

# ---------------- HELPER FUNCTIONS ----------------
def init_csv_files():
    """Initialize CSV files for predictions and stock data."""
    if not os.path.exists(PREDICTIONS_FILE):
        pd.DataFrame(columns=["ticker", "date", "predictions", "model"]).to_csv(PREDICTIONS_FILE, index=False)
    if not os.path.exists(STOCK_DATA_FILE):
        pd.DataFrame(columns=["ticker", "date", "Open", "High", "Low", "Close", "MA_7", "MA_14", "MA_21"]).to_csv(STOCK_DATA_FILE, index=False)

@st.cache_data(ttl=3600)
def fetch_stock_data(stock_ticker):
    """Fetch stock data and company info using Yahoo Finance."""
    end_date = datetime.now(pytz.UTC)
    start_date = datetime(end_date.year - 20, end_date.month, end_date.day)
    try:
        with st.spinner(f"Fetching stock data for {stock_ticker}..."):
            ticker = yf.Ticker(stock_ticker)
            df = ticker.history(start=start_date, end=end_date)
            info = ticker.info
        if df.empty:
            logger.error(f"No stock data available for {stock_ticker}.")
            st.error(f"⚠ No stock data available for {stock_ticker}.")
            return None, None
        for days in [7, 14, 21]:
            df[f'MA_{days}'] = df['Close'].rolling(days).mean()
        df_to_save = df.copy()
        df_to_save['ticker'] = stock_ticker
        df_to_save['date'] = df_to_save.index
        df_to_save.reset_index(drop=True, inplace=True)
        df_to_save.to_csv(STOCK_DATA_FILE, mode='a', header=not os.path.exists(STOCK_DATA_FILE), index=False)
        company_name = info.get('longName', stock_ticker)
        return df, company_name
    except Exception as e:
        logger.error(f"Error fetching stock data for {stock_ticker}: {e}")
        st.error(f"⚠ Error fetching stock data for {stock_ticker}: {e}")
        return None, None

def display_stock_data(df, stock_info):
    """Display stock data, info, and charts."""
    if df is not None:
        st.subheader("Stock Data")
        st.markdown(f"*Company*: {stock_info.get('longName', 'N/A')}")
        st.markdown(f"*Sector*: {stock_info.get('sector', 'N/A')}")
        st.markdown(f"*Industry*: {stock_info.get('industry', 'N/A')}")
        market_cap = stock_info.get('marketCap')
        if market_cap:
            st.markdown(f"*Market Cap*: ${market_cap:,}")
        current_price = df['Close'].iloc[-1] if not df.empty else 'N/A'
        st.metric(label="Last Price", value=f"{current_price:.2f}" if isinstance(current_price, (int, float)) else "N/A")
        fig, ax = plt.subplots(figsize=(15, 6))
        ax.plot(df.index, df['Close'], label='Close', color='blue')
        ax.plot(df.index, df['Open'], label='Open', color='green', alpha=0.6)
        ax.plot(df.index, df['High'], label='High', color='red', alpha=0.6)
        ax.plot(df.index, df['Low'], label='Low', color='orange', alpha=0.6)
        ax.plot(df.index, df['MA_7'], label='MA 7', color='purple', linestyle='--')
        ax.plot(df.index, df['MA_14'], label='MA 14', color='cyan', linestyle='--')
        ax.plot(df.index, df['MA_21'], label='MA 21', color='magenta', linestyle='--')
        ax.legend()
        ax.set_title('Historical Stock Prices with Moving Averages')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        st.pyplot(fig)
        st.write("Raw Data (including Moving Averages):")
        st.write(df[['Open', 'High', 'Low', 'Close', 'MA_7', 'MA_14', 'MA_21']])

# ---------------- MODEL FUNCTIONS ----------------
def load_keras_model():
    global model, keras_available
    try:
        model = load_model(MODEL_PATH)
        keras_available = True
        st.success("✅ Keras model loaded successfully!")
    except Exception as e:
        logger.error(f"Keras model not loaded: {e}")
        st.warning(f"⚠ Keras model not loaded: {e}. ARIMA model will be used.")
        keras_available = False

def predict_arima(df, stock_ticker):
    if df is None:
        return None, None
    try:
        with st.spinner("Running ARIMA prediction..."):
            arima_model = ARIMA(df['Close'].dropna(), order=(5, 1, 0))
            arima_fit = arima_model.fit()
            forecast_steps = st.slider("Select number of days to forecast (ARIMA)", min_value=1, max_value=30, value=7, key="arima")
            arima_forecast = arima_fit.forecast(steps=forecast_steps)
            arima_dates = pd.date_range(df['Close'].dropna().index[-1], periods=forecast_steps, freq='B')
            arima_predictions_df = pd.DataFrame({'Date': arima_dates, 'Predicted Price': arima_forecast})
            predictions_to_save = arima_predictions_df.copy()
            predictions_to_save['ticker'] = stock_ticker
            predictions_to_save['date'] = datetime.now(pytz.UTC).strftime('%Y-%m-%d')
            predictions_to_save['model'] = 'ARIMA'
            predictions_to_save.to_csv(PREDICTIONS_FILE, mode='a', header=not os.path.exists(PREDICTIONS_FILE), index=False)
            return arima_predictions_df, arima_fit
    except Exception as e:
        logger.error(f"Error with ARIMA prediction: {e}")
        st.error(f"⚠ Error with ARIMA prediction: {e}")
        return None, None

def predict_keras(df, model, stock_ticker):
    if not keras_available or model is None or df is None:
        return None, None, None
    try:
        with st.spinner("Running Keras model prediction..."):
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(df[['Close']])
            time_step = 100
            x_data, y_data = [], []
            for i in range(time_step, len(scaled_data)):
                x_data.append(scaled_data[i - time_step:i])
                y_data.append(scaled_data[i])
            x_data, y_data = np.array(x_data), np.array(y_data)
            predictions = model.predict(x_data, verbose=0)
            inv_pre = scaler.inverse_transform(predictions)
            inv_y_test = scaler.inverse_transform(y_data)
            ploting_data = pd.DataFrame({'Original': inv_y_test.flatten(), 'Predicted': inv_pre.flatten()},
                                       index=df.index[len(df) - len(y_data):])
            future_predictions = []
            last_sequence = scaled_data[-time_step:]
            current_input = last_sequence.reshape(1, time_step, 1)
            future_days = st.slider("Select number of days to forecast (Keras)", min_value=1, max_value=30, value=7, key="keras")
            for _ in range(future_days):
                predicted_price = model.predict(current_input, verbose=0)
                future_predictions.append(predicted_price[0, 0])
                current_input = np.append(current_input[:, 1:, :], [[[predicted_price[0, 0]]]], axis=1)
            future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))
            future_dates = pd.date_range(df.index[-1], periods=future_days + 1, freq='B')[1:]
            future_df = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_predictions.flatten()})
            predictions_to_save = future_df.copy()
            predictions_to_save['ticker'] = stock_ticker
            predictions_to_save['date'] = datetime.now(pytz.UTC).strftime('%Y-%m-%d')
            predictions_to_save['model'] = 'Keras'
            predictions_to_save.to_csv(PREDICTIONS_FILE, mode='a', header=not os.path.exists(PREDICTIONS_FILE), index=False)
            return ploting_data, future_df, df
    except Exception as e:
        logger.error(f"Error with Keras model prediction: {e}")
        st.error(f"⚠ Error with Keras model prediction: {e}")
        return None, None, None

def predict_ensemble(df, stock_ticker):
    if df is None:
        logger.warning("No data provided for ensemble prediction.")
        st.error("⚠ No stock data provided.")
        return None, None, None
    try:
        with st.spinner("Running Ensemble model prediction..."):
            models = {
                'Linear': LinearRegression(),
                'KNN': KNeighborsRegressor(n_neighbors=5),
                'Ridge': Ridge(alpha=1.0),
                'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42)
            }
            df_features = pd.DataFrame(index=df.index)
            df_features['Close'] = df['Close'].copy()
            lags = 5
            for i in range(1, lags + 1):
                df_features[f'Close_lag{i}'] = df_features['Close'].shift(i)
            df_features['Target'] = df_features['Close'].shift(-1)
            df_features.dropna(inplace=True)
            if len(df_features) < lags + 1:
                raise ValueError(f"Insufficient data points ({len(df_features)}) after creating lags")
            feature_columns = [f'Close_lag{i}' for i in range(1, lags + 1)]
            X = df_features[feature_columns].values
            y = df_features['Target'].values
            if len(X) < 20:
                raise ValueError(f"Not enough data points ({len(X)}) for ensemble modeling")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, shuffle=False
            )
            predictions = {}
            scores = {}
            for name, model in models.items():
                model.fit(X_train, y_train)
                if name == 'RandomForest':
                    logger.info("RandomForestRegressor fitted successfully")
                pred = model.predict(X_test)
                predictions[name] = pred
                mse = mean_squared_error(y_test, pred)
                scores[name] = 1 / (1 + mse)
            total_score = sum(scores.values())
            weights = {name: score / total_score for name, score in scores.items()}
            ensemble_pred = np.zeros_like(y_test)
            for name, pred in predictions.items():
                ensemble_pred += pred * weights[name]
            test_dates = df_features.index[-len(y_test):]
            predictions_df = pd.DataFrame({
                'Date': test_dates,
                'Actual': y_test,
                'Ensemble': ensemble_pred
            }, index=test_dates)
            future_days = st.slider("Select number of days to forecast (Ensemble)", min_value=1, max_value=30, value=7, key="ensemble")
            future_predictions = []
            current_features = X[-1].copy()
            for _ in range(future_days):
                pred_dict = {}
                current_features_scaled = scaler.transform(current_features.reshape(1, -1))
                for name, model in models.items():
                    pred = model.predict(current_features_scaled)[0]
                    pred_dict[name] = pred
                ensemble_pred = sum(pred * weights[name] for name, pred in pred_dict.items())
                future_predictions.append(ensemble_pred)
                current_features = np.roll(current_features, -1)
                current_features[-1] = ensemble_pred
            future_dates = pd.date_range(df.index[-1], periods=future_days + 1, freq='B')[1:]
            future_df = pd.DataFrame({'Date': future_dates, 'Predicted': future_predictions})
            predictions_to_save = future_df.copy()
            predictions_to_save['ticker'] = stock_ticker
            predictions_to_save['model'] = 'Ensemble'
            predictions_to_save['date'] = datetime.now(pytz.UTC).strftime('%Y-%m-%d')
            predictions_to_save.to_csv(PREDICTIONS_FILE, mode='a', header=not os.path.exists(PREDICTIONS_FILE), index=False)
            return predictions_df, future_df, scores
    except Exception as e:
        logger.error(f"Error with Ensemble prediction: {str(e)}")
        st.error(f"⚠ Error with Ensemble prediction: {str(e)}")
        return None, None, None

def predict_sentiment_adjusted_ensemble(df, stock_ticker, aggregated_score):
    """Predict stock prices using an ensemble adjusted by sentiment score."""
    if df is None or aggregated_score is None:
        logger.warning("No data or sentiment score provided for sentiment-adjusted ensemble prediction.")
        st.error("⚠ No stock data or sentiment score available.")
        return None, None, None
    try:
        with st.spinner("Running Sentiment-Adjusted Ensemble prediction..."):
            models = {
                'Linear': LinearRegression(),
                'KNN': KNeighborsRegressor(n_neighbors=5),
                'Ridge': Ridge(alpha=1.0),
                'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42)
            }
            df_features = pd.DataFrame(index=df.index)
            df_features['Close'] = df['Close'].copy()
            lags = 5
            for i in range(1, lags + 1):
                df_features[f'Close_lag{i}'] = df_features['Close'].shift(i)
            df_features['Target'] = df_features['Close'].shift(-1)
            df_features.dropna(inplace=True)
            if len(df_features) < lags + 1:
                raise ValueError(f"Insufficient data points ({len(df_features)}) after creating lags")
            feature_columns = [f'Close_lag{i}' for i in range(1, lags + 1)]
            X = df_features[feature_columns].values
            y = df_features['Target'].values
            if len(X) < 20:
                raise ValueError(f"Not enough data points ({len(X)}) for sentiment-adjusted ensemble modeling")
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, shuffle=False
            )
            predictions = {}
            scores = {}
            for name, model in models.items():
                model.fit(X_train, y_train)
                if name == 'RandomForest':
                    logger.info("RandomForestRegressor fitted successfully for sentiment-adjusted ensemble")
                pred = model.predict(X_test)
                predictions[name] = pred
                mse = mean_squared_error(y_test, pred)
                scores[name] = 1 / (1 + mse)
            total_score = sum(scores.values())
            weights = {name: score / total_score for name, score in scores.items()}
            ensemble_pred = np.zeros_like(y_test)
            for name, pred in predictions.items():
                ensemble_pred += pred * weights[name]
            sentiment_factor = 1 + (aggregated_score * 0.1)  # 10% adjustment per sentiment unit
            adjusted_ensemble_pred = ensemble_pred * sentiment_factor
            test_dates = df_features.index[-len(y_test):]
            predictions_df = pd.DataFrame({
                'Date': test_dates,
                'Actual': y_test,
                'Sentiment-Adjusted Ensemble': adjusted_ensemble_pred
            }, index=test_dates)
            future_days = st.slider("Select number of days to forecast (Sentiment-Adjusted Ensemble)", min_value=1, max_value=30, value=7, key="sentiment_adjusted_ensemble")
            future_predictions = []
            current_features = X[-1].copy()
            for _ in range(future_days):
                pred_dict = {}
                for name, model in models.items():
                    pred = model.predict(current_features.reshape(1, -1))[0]
                    pred_dict[name] = pred
                ensemble_pred = sum(pred * weights[name] for name, pred in pred_dict.items())
                adjusted_pred = ensemble_pred * sentiment_factor
                future_predictions.append(adjusted_pred)
                current_features = np.roll(current_features, -1)
                current_features[-1] = adjusted_pred
            future_dates = pd.date_range(df.index[-1], periods=future_days + 1, freq='B')[1:]
            future_df = pd.DataFrame({'Date': future_dates, 'Predicted': future_predictions})
            predictions_to_save = future_df.copy()
            predictions_to_save['ticker'] = stock_ticker
            predictions_to_save['model'] = 'Sentiment-Adjusted Ensemble'
            predictions_to_save['date'] = datetime.now(pytz.UTC).strftime('%Y-%m-%d')
            predictions_to_save.to_csv(PREDICTIONS_FILE, mode='a', header=not os.path.exists(PREDICTIONS_FILE), index=False)
            return predictions_df, future_df, scores
    except Exception as e:
        logger.error(f"Error with Sentiment-Adjusted Ensemble prediction: {str(e)}")
        st.error(f"⚠ Error with Sentiment-Adjusted Ensemble prediction: {str(e)}")
        return None, None, None

# ---------------- MAIN APP FLOW ----------------
def main():
    init_csv_files()
    load_keras_model()

    if sentiment_pipeline is None:
        st.error(f"⚠ Failed to load sentiment model '{DEFAULT_SENTIMENT_MODEL}'. Cannot perform sentiment analysis.")
        return

    stock_ticker = st.text_input("Enter Stock ID (e.g., AAPL, TCS.NS)", "AAPL").upper().strip()
    if not stock_ticker:
        st.warning("Please enter a stock ticker symbol.")
        return

    df, company_name = fetch_stock_data(stock_ticker)
    stock_info = yf.Ticker(stock_ticker).info if df is not None else {}
    display_stock_data(df, stock_info)

    # Sentiment Analysis
    st.subheader("📰 News Sentiment Analysis")
    articles = fetch_news(stock_ticker, company_name)
    aggregated_score, analyzed_details, sentiment_series = analyze_sentiment_for_ticker(articles, sentiment_pipeline, df)
    suggestion = get_suggestion(aggregated_score)
    validation_points = get_validation_points(analyzed_details)

    if analyzed_details:
        st.write(f"*Average Sentiment Score*: {aggregated_score:.3f}")
        st.write(f"*AI Suggestion*: {suggestion}")
        st.markdown("*Justification based on news:*")
        if validation_points:
            for point in validation_points:
                st.markdown(point, unsafe_allow_html=True)
        else:
            st.write("No specific validation points extracted.")
        st.write("*Recent News Headlines Analyzed:*")
        news_df = pd.DataFrame(analyzed_details)
        news_df['source'] = [article.get('source', {}).get('name', 'Unknown') for article in articles[:len(analyzed_details)]]
        news_df['publishedAt'] = [article.get('publishedAt') for article in articles[:len(analyzed_details)]]
        display_df = news_df[['headline', 'source', 'publishedAt', 'score', 'label']].rename(columns={
            'headline': 'Title', 'source': 'Source', 'publishedAt': 'Date', 'score': 'Sentiment Score', 'label': 'Sentiment Label'
        })
        st.write(display_df.head(10))
        news_csv = display_df.to_csv(index=False)
        st.download_button(
            label="Download News Data as CSV",
            data=news_csv,
            file_name=f"{stock_ticker}_news.csv",
            mime="text/csv"
        )
        # Sentiment Trend
        if sentiment_series is not None:
            st.subheader("Sentiment Trend")
            fig, ax = plt.subplots(figsize=(15, 4))
            ax.plot(sentiment_series.index, sentiment_series, label='Sentiment Score', color='teal')
            ax.set_title(f'Daily Sentiment Trend for {stock_ticker} (News-Based)')
            ax.set_xlabel('Date')
            ax.set_ylabel('Sentiment Score')
            ax.legend()
            st.pyplot(fig)
            # Sentiment Statistics
            st.write("Sentiment Statistics:")
            sentiment_stats = {
                "Mean": sentiment_series.mean().round(4),
                "Std Dev": sentiment_series.std().round(4),
                "Min": sentiment_series.min().round(4),
                "Max": sentiment_series.max().round(4)
            }
            st.write(pd.DataFrame([sentiment_stats]))
    else:
        st.warning(f"No sentiment data available for {stock_ticker}. Using neutral sentiment.")
        aggregated_score = 0.0
        suggestion = "Hold"

    # ARIMA Predictions
    arima_predictions_df, _ = predict_arima(df, stock_ticker)
    if arima_predictions_df is not None:
        st.subheader("ARIMA Predictions")
        st.write(arima_predictions_df)
        st.line_chart(arima_predictions_df.set_index("Date"), height=300, use_container_width=True)
        arima_csv = arima_predictions_df.to_csv(index=False)
        st.download_button(
            label="Download ARIMA Predictions as CSV",
            data=arima_csv,
            file_name=f"{stock_ticker}_arima_predictions.csv",
            mime="text/csv"
        )

    # Keras Predictions
    keras_future_df = None
    if keras_available:
        ploting_data, keras_future_df, _ = predict_keras(df, model, stock_ticker)
        if ploting_data is not None and keras_future_df is not None:
            st.subheader("Keras Model Predictions")
            st.write(ploting_data)
            st.line_chart(ploting_data[["Original", "Predicted"]], height=300, use_container_width=True)
            keras_hist_csv = ploting_data.to_csv(index=True)
            st.download_button(
                label="Download Keras Historical Predictions as CSV",
                data=keras_hist_csv,
                file_name=f"{stock_ticker}_keras_historical_predictions.csv",
                mime="text/csv"
            )
            st.subheader("Keras Model - Future Predictions")
            st.write(keras_future_df)
            st.line_chart(keras_future_df.set_index("Date"), height=300, use_container_width=True)
            keras_future_csv = keras_future_df.to_csv(index=False)
            st.download_button(
                label="Download Keras Future Predictions as CSV",
                data=keras_future_csv,
                file_name=f"{stock_ticker}_keras_future_predictions.csv",
                mime="text/csv"
            )

    # Ensemble Predictions
    ensemble_predictions_df, ensemble_future_df, scores = predict_ensemble(df, stock_ticker)
    if ensemble_predictions_df is not None and ensemble_future_df is not None:
        st.subheader("Ensemble Model Predictions")
        st.write("Model Scores:", scores)
        st.write(ensemble_predictions_df)
        st.line_chart(ensemble_predictions_df.set_index("Date")[["Actual", "Ensemble"]], height=300, use_container_width=True)
        ensemble_hist_csv = ensemble_predictions_df.to_csv(index=True)
        st.download_button(
            label="Download Ensemble Historical Predictions as CSV",
            data=ensemble_hist_csv,
            file_name=f"{stock_ticker}_ensemble_historical_predictions.csv",
            mime="text/csv"
        )
        st.subheader("Ensemble Model - Future Predictions")
        st.write(ensemble_future_df)
        st.line_chart(ensemble_future_df.set_index("Date"), height=300, use_container_width=True)
        ensemble_future_csv = ensemble_future_df.to_csv(index=False)
        st.download_button(
            label="Download Ensemble Future Predictions as CSV",
            data=ensemble_future_csv,
            file_name=f"{stock_ticker}_ensemble_future_predictions.csv",
            mime="text/csv"
        )

    # Sentiment-Adjusted Ensemble Predictions
    if aggregated_score is None:
        st.error("⚠ Sentiment score is unavailable. Skipping sentiment-adjusted predictions.")
        sentiment_adj_predictions_df, sentiment_adj_future_df, sentiment_adj_scores = None, None, None
    else:
        sentiment_adj_predictions_df, sentiment_adj_future_df, sentiment_adj_scores = predict_sentiment_adjusted_ensemble(df, stock_ticker, aggregated_score)
    if sentiment_adj_predictions_df is not None and sentiment_adj_future_df is not None:
        st.subheader("Sentiment-Adjusted Ensemble Model Predictions")
        st.write("Model Scores:", sentiment_adj_scores)
        st.write(sentiment_adj_predictions_df)
        st.line_chart(sentiment_adj_predictions_df.set_index("Date")[["Actual", "Sentiment-Adjusted Ensemble"]], height=300, use_container_width=True)
        sentiment_adj_hist_csv = sentiment_adj_predictions_df.to_csv(index=True)
        st.download_button(
            label="Download Sentiment-Adjusted Ensemble Historical Predictions as CSV",
            data=sentiment_adj_hist_csv,
            file_name=f"{stock_ticker}_sentiment_adj_ensemble_historical_predictions.csv",
            mime="text/csv"
        )
        st.subheader("Sentiment-Adjusted Ensemble Model - Future Predictions")
        st.write(sentiment_adj_future_df)
        st.line_chart(sentiment_adj_future_df.set_index("Date"), height=300, use_container_width=True)
        sentiment_adj_future_csv = sentiment_adj_future_df.to_csv(index=False)
        st.download_button(
            label="Download Sentiment-Adjusted Ensemble Future Predictions as CSV",
            data=sentiment_adj_future_csv,
            file_name=f"{stock_ticker}_sentiment_adj_ensemble_future_predictions.csv",
            mime="text/csv"
        )

    # Model Comparison
    st.subheader("📊 Model Comparison")
    comparison_models = []
    if arima_predictions_df is not None:
        comparison_models.append(('ARIMA', arima_predictions_df))
    if keras_future_df is not None:
        comparison_models.append(('Keras', keras_future_df))
    if ensemble_future_df is not None:
        comparison_models.append(('Ensemble', ensemble_future_df))
    if sentiment_adj_future_df is not None:
        comparison_models.append(('Sentiment-Adjusted Ensemble', sentiment_adj_future_df))
    if comparison_models:
        min_forecast_days = min(len(df) for _, df in comparison_models)
        if min_forecast_days == 0:
            st.warning("No predictions available to compare.")
            return
        comparison_df = pd.DataFrame(index=pd.date_range(df.index[-1], periods=min_forecast_days + 1, freq='B')[1:])
        last_close = df['Close'].iloc[-1].item() if (df is not None and not df.empty and 'Close' in df.columns) else None
        for model_name, predictions_df in comparison_models:
            col_name = 'Predicted Price' if model_name in ['ARIMA', 'Keras'] else 'Predicted'
            comparison_df[model_name] = predictions_df[col_name].values[:min_forecast_days]
        comparison_df['Sentiment Score'] = aggregated_score
        st.write(f"Future Predictions Comparison (First {min_forecast_days} Days)")
        st.write(comparison_df)

        # Prediction Statistics
        st.subheader("📈 Prediction Statistics")
        if comparison_df.empty:
            st.write("No prediction data available for statistics.")
        else:
            st.write(f"Last Closing Price: {last_close if last_close is not None else 'N/A'}")
            means = comparison_df.mean().round(2)
            stds = comparison_df.std().round(2)
            mins = comparison_df.min().round(2)
            maxs = comparison_df.max().round(2)
            ranges = (maxs - mins).round(2)
            ci_lower = (means - 1.96 * stds / np.sqrt(min_forecast_days)).round(2)
            ci_upper = (means + 1.96 * stds / np.sqrt(min_forecast_days)).round(2)
            pct_change = {}
            if last_close is not None and last_close != 0:
                for col in comparison_df.columns:
                    if col != 'Sentiment Score':
                        pct_change[col] = ((means[col] - last_close) / last_close * 100).round(2)
            else:
                pct_change = {col: None for col in comparison_df.columns if col != 'Sentiment Score'}
            stats_df = pd.DataFrame({
                'Mean': means,
                'Std Dev': stds,
                'Min': mins,
                'Max': maxs,
                'Range': ranges,
                '95% CI Lower': ci_lower,
                '95% CI Upper': ci_upper,
                '% Change': pd.Series(pct_change)
            })
            with st.expander("What do these statistics mean?"):
                st.markdown("""
                - *Mean*: Average predicted price. Higher = more optimistic.
                - *Std Dev*: Variability of predictions. Higher = more uncertainty.
                - *Min*: Lowest predicted price (worst-case).
                - *Max*: Highest predicted price (best-case).
                - *Range*: Max - Min (prediction spread).
                - *95% CI Lower/Upper*: 95% confidence interval for the mean.
                - *% Change*: Expected percentage change from the last close (green = positive, red = negative).
                - *Sentiment Score*: News sentiment (-1 to 1). Positive (>0.1) = Buy, Negative (<-0.1) = Sell.
                """)
            def color_pct_change(val):
                if val is None or pd.isna(val):
                    return ''
                color = 'green' if val > 0 else 'red' if val < 0 else 'black'
                return f'color: {color}'
            st.write(stats_df.style.applymap(color_pct_change, subset=['% Change']))
            stats_csv = stats_df.to_csv(index=True)
            st.download_button(
                label="Download Statistics as CSV",
                data=stats_csv,
                file_name=f"{stock_ticker}_prediction_stats.csv",
                mime="text/csv"
            )

if _name_ == "_main_":
    main()