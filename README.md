# Quantus

## 🚀 Quick Start

```bash
# 1. Install dependencies (first time)
pip install torch yfinance finnhub-python pandas ta numpy scikit-learn click groq pyyaml pydantic

# 2. Train model (first time)
PYTHONPATH=. python train/model.py

# 3. Run CLI
python cli/main.py

# 4. Use it
📝 AAPL                       # predict stock
📝 TSLA                       # predict another
📝 quit                       # exit
```

> **Note**: `train/model.py` imports package modules, so run it with `PYTHONPATH=.`
> from the project root. On some systems use `python3` instead of `python`.

---

## 🧠 Groq AI Setup (Optional)

Groq explanations are **optional**. The CLI works without it, just no AI analysis.

**To enable Groq AI:**

1. Get free API key: (https://groq.com)
2. Set environment variable:
   ```bash
   echo 'export GROQ_API_KEY=your_key_here' >> ~/.zshrc
   source ~/.zshrc
   ```
   (or add `groq_api_key: your_key_here` to `config/local.yaml`)
3. Run CLI — Groq auto-works ✅

**Without Groq:** CLI still predicts stocks (BUY/HOLD/SELL), just no explanation.

---

## 📊 Features

- **Neural Network**: MLP (11->64->32->3) for 3-class classification (BUY/HOLD/SELL)
- **Technical Indicators**: RSI, MACD, EMA (20/50/200), volatility, momentum
- **Groq AI**: 3 reasons + 3 risks explanation (neutral analysis) [optional]
- **Caching**: OHLCV data cached to `.cache/` (never expires; use `no-cache TICKER` to force a fresh fetch)
- **Interactive CLI**: Type multiple tickers without restarting

---

## 🏗️ Project Structure

```
Quantus/
├── cli/
│   └── main.py              # Interactive CLI (entry point)
├── pipeline/
│   ├── data_fetcher.py      # yfinance data fetching
│   ├── indicators.py        # Technical indicator computation
├── train/
│   ├── model.py             # MLP training + evaluation
│   └── dataset.py           # Dataset preparation
├── models/
│   └── stock_classifier.pt  # Trained model (after training)
├── config/                  # Configuration files
├── .cache/                  # Auto-generated data cache
└── main.py                  # Empty placeholder
```

---

## 🧮 Model Architecture

```
Input (11 features) -> Hidden1 (64) -> Hidden2 (32) -> Output (3)
                    ↓           ↓
                  Dropout       Dropout
```

**Features**: RSI, MACD, EMA_20/50/200, Price_Change, Vol_Change, Volatility, MA_20/50/200

**Output**: 
- `0` -> SELL 📉
- `1` -> HOLD ➡️
- `2` -> BUY 📈

---

## 📖 CLI Commands

| Command          | Description |
|------------------|-------------|
| `AAPL`           | Predict stock |
| `period=1y`      | Set data period (default: 5y) |
| `interval=1h`    | Set data interval (default: 1d) |
| `no-cache AAPL`  | Predict with a fresh data fetch (skip cache) |
| `quit`           | Exit CLI |

---

## 🔧 Setup

```bash
# Install dependencies
pip install torch yfinance finnhub-python pandas ta numpy scikit-learn click groq pyyaml pydantic

# Train model
PYTHONPATH=. python train/model.py

# Run CLI
python cli/main.py
```

---

## 🎯 Performance

- **Train Accuracy**: ~79-83%
- **Test Accuracy**: ~73-77%
- **Confidence**: 50-90% (varies by stock)

---

## 🧠 Groq AI Integration

Get concise explanations with **3 reasons** + **3 risks**:

```
🧠 Groq AI Analysis...
  ✓ RSI above 50 shows bullish momentum
  ✓ MACD positive crossover indicates upward trend
  ✓ EMA_20 > EMA_50 confirms short-term strength
  
  ⚠️ Volatility elevated - potential reversal
  ⚠️ Volume declining - weak conviction
  ⚠️ Market sentiment uncertain
```

**Setup**: Set `GROQ_API_KEY` env var (or `config/local.yaml`), then run the CLI.

---

## 🚦 Usage Example

```bash
$ python cli/main.py

  Quantus | MLP | Groq: disabled

  Enter ticker: AAPL
  🚀 AAPL | 5y | 1d
  ✅ 1255 OHLCV rows

  🎯 BUY 📈
     Neural Network Confidence: 88.7%
     Model: MLP (11->64->32->3)

  ✅ AAPL -> BUY 📈

  Enter ticker: quit
  Bye! 👋
```

---

## 🛠️ Tech Stack

- **Python 3.14**
- **PyTorch** (Neural Network)
- **yfinance** (Price data)
- **Finnhub** (News sentiment)
- **ta** (Technical indicators)
- **Groq** (AI explanations)
- **Click** (CLI)

---

## 📝 License

Apache 2.0 License - See [LICENSE]