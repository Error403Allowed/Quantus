from datetime import datetime, timedelta, timezone

import finnhub
import pandas as pd
import yfinance as yf

def fetch_news_data(ticker: str, api_key: str, days_back: int = 7) -> list[str]:
    client = finnhub.Client(api_key=api_key)
    end_date = datetime.now(timezone.utc).date()
    start_date = end_date - timedelta(days=days_back)

    news = client.company_news(ticker, _from=start_date.isoformat(), to=end_date.isoformat())
    return [article["headline"] for article in news if "headline" in article]