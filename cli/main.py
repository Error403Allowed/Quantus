#!/usr/bin/env python3
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Optional

import yaml

warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
# config paths like models/alpha/weights/... are relative, so always run from the project root
os.chdir(PROJECT_ROOT)

from config.schema import load_config
from pipeline.features import build_features
from pipeline.orchestrator import QuantusOrchestrator
from pipeline.price_fetcher import fetch_price_data

LOCAL_CONFIG = PROJECT_ROOT / 'config' / 'local.yaml'

GREEN = '\x1b[32m'
RED = '\x1b[31m'
YELLOW = '\x1b[33m'
CYAN = '\x1b[36m'
WHITE = '\x1b[37m'
BOLD = '\x1b[1m'
RESET = '\x1b[0m'

ACTIONS = {
    'short': ('SELL', '📉', RED),
    'flat': ('HOLD', '➡️', YELLOW),
    'long': ('BUY', '📈', GREEN),
}


def c(text, color):
    return f"{color}{text}{RESET}"


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


def predict_stock(
    ticker: str,
    period: str,
    interval: str,
    orchestrator: QuantusOrchestrator,
    groq_api_key: Optional[str] = None,
) -> None:
    ticker = ticker.upper()
    print(c(f'\n🚀 {ticker}', CYAN) + c(f' | {period} | {interval}', WHITE))

    price_data = fetch_price_data(ticker, period, interval)
    print(c(f'✅ {len(price_data)} OHLCV rows', GREEN))

    features = build_features(price_data)
    decision = orchestrator.decide_from_alpha_only(features)

    action, emoji, color = ACTIONS[decision.action]
    conf = decision.confidence
    lookback = orchestrator.config.features.lookback

    print()
    print(c(BOLD + f'🎯 {action} {emoji}', color) + BOLD)
    print(c(f'   Neural Network Confidence: {conf * 100:.1f}%', WHITE))
    print(c(f'   Model: GRU ({lookback}-day window)', CYAN))

    if groq_api_key and GROQ_AVAILABLE:
        print(c('\nQuantus AI Analysis...', CYAN))
        try:
            client = Groq(api_key=groq_api_key)
            # unscaled values, so the LLM sees readable numbers
            latest = features[orchestrator.alpha_service.feature_columns].iloc[-1]
            features_str = '\n'.join(f'  {n}: {v:.4f}' for n, v in latest.items())
            prompt = (
                f"Ticker: {ticker}\n"
                f"Prediction: {action} ({conf * 100:.1f}%)\n"
                f"Indicators:\n{features_str}\n\n"
                "3 reasons + 3 risks. Max 150 words. Do not say how many words at end of response."
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

    print(c(f'\n✅ {ticker} -> {action} {emoji}', GREEN) + '\n')


def main() -> None:
    local = load_local_config()
    groq_api_key = local.get('groq_api_key') or os.getenv('GROQ_API_KEY')

    if groq_api_key and GROQ_AVAILABLE:
        status = 'Groq: enabled'
    elif groq_api_key and not GROQ_AVAILABLE:
        status = 'Groq: import failed'
    else:
        status = 'Groq: disabled'

    # built once so the model and scaler aren't reloaded on every ticker
    config = load_config(str(PROJECT_ROOT / 'config' / 'default.yaml'))
    orchestrator = QuantusOrchestrator(config)

    print(CYAN + BOLD + 'Quantus' + RESET + WHITE + f' | GRU | {status}' + RESET)
    print()

    period = config.data.period
    interval = config.data.interval

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
            predict_stock(user_input, period, interval, orchestrator, groq_api_key)
        except KeyboardInterrupt:
            print(CYAN + '\nBye! 👋' + RESET)
            break
        except Exception as e:
            print(RED + f'{e}' + RESET)


if __name__ == '__main__':
    main()
