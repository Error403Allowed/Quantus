import yfinance as yf
import finnhub
from datetime import datetime, timedelta, timezone
import pandas as pd

def fetch_price_data(ticker: str, period="6mo", interval="1d"):
    data = yf.download(ticker, period=period, interval=interval, auto_adjust=True, progress=False)

    # Validate data
    if data is None or data.empty:
        raise ValueError(f"Could not fetch price data for {ticker}")

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    return data # Only return data if validation passes

def fetch_news_data(ticker: str, api_key: str, days_back: int = 7):
    client = finnhub.Client(api_key=api_key)
    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=days_back)
    start_str = start_date.isoformat()
    end_str = end_date.isoformat()
    news = client.company_news(ticker, _from=start_str, to=end_str)
    headlines = [article["headline"] for article in news if "headline" in article]
    return headlines
