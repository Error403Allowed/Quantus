#!/usr/bin/env python3
"""Quantus"""

import sys
from pathlib import Path
from typing import Optional
import warnings
import os
import pickle
import yaml
from typing import Any

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from pipeline.data_fetcher import fetch_price_data
from pipeline.indicators import compute_indicators, get_latest_data
from train.model import StockClassifier
import torch
import pandas as pd

LOCAL_CONFIG = PROJECT_ROOT / 'local.yaml'


def load_local_config() -> dict:
    if not LOCAL_CONFIG.exists():
        return {}
    try:
        return yaml.safe_load(LOCAL_CONFIG.read_text(encoding='utf-8')) or {}
    except Exception:
        return {}


GROQ_AVAILABLE = False
GROQ_IMPORT_ERROR = None
Groq: Any = None

try:
    from groq import Groq as _Groq
    Groq = _Groq
    GROQ_AVAILABLE = True
except Exception as e:
    GROQ_IMPORT_ERROR = repr(e)



CACHE_DIR = PROJECT_ROOT / '.cache'
CACHE_DIR.mkdir(exist_ok=True)


def get_cache_key(ticker: str, period: str, interval: str) -> str:
    return f"{ticker}_{period}_{interval}"


def load_cache(key: str) -> Optional[pd.DataFrame]:
    cache_file = CACHE_DIR / f"{key}.pkl"
    if not cache_file.exists():
        return None

    try:
        with cache_file.open('rb') as f:
            data = pickle.load(f)
        if not isinstance(data, pd.DataFrame):
            return None
        return data
    except Exception:
        return None


def save_cache(key: str, data: pd.DataFrame) -> None:
    try:
        cache_file = CACHE_DIR / f"{key}.pkl"
        with cache_file.open('wb') as f:
            pickle.dump(data, f)
    except Exception:
        pass


def predict_stock(
    ticker: str,
    period: str = '5y',
    interval: str = '1d',
    model_path: str = 'models/stock_classifier.pt',
    groq_api_key: Optional[str] = None,
    no_cache: bool = False,
) -> None:
    ticker = ticker.upper()

    GREEN = '\x1b[32m'
    RED = '\x1b[31m'
    YELLOW = '\x1b[33m'
    CYAN = '\x1b[36m'
    WHITE = '\x1b[37m'
    BOLD = '\x1b[1m'
    RESET = '\x1b[0m'

    def c(text, color):
        return f"{color}{text}{RESET}"

    print(c(f'\n🚀 {ticker}', CYAN) + c(f' | {period} | {interval}', WHITE))

    cache_key = get_cache_key(ticker, period, interval)
    if not no_cache:
        cached = load_cache(cache_key)
        if cached is not None:
            print(c('✅ Cached data', YELLOW))
            price_data = cached
        else:
            price_data = fetch_price_data(ticker, period, interval)
            save_cache(cache_key, price_data)
            print(c('✅ Downloaded (saved to cache)', GREEN))
    else:
        price_data = fetch_price_data(ticker, period, interval)

    print(c(f'✅ {len(price_data)} OHLCV rows', GREEN))

    indicators = compute_indicators(price_data)
    features = get_latest_data(indicators)

    model = StockClassifier()
    model.load_state_dict(torch.load(model_path, weights_only=True, map_location='cpu'))
    model.eval()

    input_array = features.values.copy()
    input_tensor = torch.FloatTensor(input_array).unsqueeze(0)

    with torch.no_grad():
        out = model(input_tensor)
        pred = int(torch.argmax(out, dim=1)[0])
        conf = torch.softmax(out, dim=1)[0][pred].item()

    actions = {0: ('SELL', '📉', RED), 1: ('HOLD', '➡️', YELLOW), 2: ('BUY', '📈', GREEN)}
    action, emoji, color = actions[pred]

    print()
    print(c(BOLD + f'🎯 {action} {emoji}', color) + BOLD)
    print(c(f'   Neural Network Confidence: {conf*100:.1f}%', WHITE))
    print(c(f'   Model: MLP (11→64→32→3)', CYAN))

    if groq_api_key and GROQ_AVAILABLE:
        print(c('\nQuantus AI Analysis...', CYAN))
        try:
            client = Groq(api_key=groq_api_key)
            features_str = '\n'.join(f'  {n}: {v:.2f}' for n, v in features.items())
            prompt = (
                f"Ticker: {ticker}\n"
                f"Prediction: {action} ({conf*100:.1f}%)\n"
                f"Indicators:\n{features_str}\n\n"
                "3 reasons + 3 risks. Max 150 words."
            )
            resp = client.chat.completions.create(
                model='openai/gpt-oss-20b',
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=500,
            )
            content = resp.choices[0].message.content
            if content:
                print(c('  ' + content, WHITE))
        except Exception as e:
            print(c(f'  ⚠️  {e}', YELLOW))
    elif groq_api_key and not GROQ_AVAILABLE:
        print(c(f'  ⚠️  Groq import failed: {GROQ_IMPORT_ERROR}', YELLOW))

    print(c(f'\n✅ {ticker} → {action} {emoji}', GREEN) + '\n')


def main() -> None:
    GREEN = '\x1b[32m'
    RED = '\x1b[31m'
    YELLOW = '\x1b[33m'
    CYAN = '\x1b[36m'
    WHITE = '\x1b[37m'
    BOLD = '\x1b[1m'
    RESET = '\x1b[0m'

    config = load_local_config()
    groq_api_key = config.get('groq', {}).get('api_key') or os.getenv('GROQ_API_KEY')

    if groq_api_key and GROQ_AVAILABLE:
        status = 'Groq: enabled'
    elif groq_api_key and not GROQ_AVAILABLE:
        status = 'Groq: import failed'
    else:
        status = 'Groq: disabled'

    print(CYAN + BOLD + 'Quantus' + RESET + WHITE + f' | MLP | {status}' + RESET)
    print()

    period = '5y'
    interval = '1d'
    model_path = 'models/stock_classifier.pt'

    while True:
        try:
            user_input = input(CYAN + 'Enter ticker: ' + RESET).strip()
            if not user_input:
                continue
            if user_input.lower() == 'quit':
                print(CYAN + '\nBye! 👋' + RESET)
                break
            if user_input.startswith('period='):
                period = user_input.split('=', 1)[1]
                print(CYAN + f'Period: {period}' + RESET)
                continue
            if user_input.startswith('interval='):
                interval = user_input.split('=', 1)[1]
                print(CYAN + f'Interval: {interval}' + RESET)
                continue
            if user_input.startswith('no-cache '):
                ticker = user_input.split(maxsplit=1)[1]
                predict_stock(ticker, period, interval, model_path, groq_api_key, no_cache=True)
                continue
            predict_stock(user_input, period, interval, model_path, groq_api_key)
        except KeyboardInterrupt:
            print(CYAN + '\nBye! 👋' + RESET)
            break
        except Exception as e:
            print(RED + f'{e}' + RESET)


if __name__ == '__main__':
    main()
