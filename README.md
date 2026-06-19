# Quantus

## 🚀 Quick Start

```bash
# 1. Train model (first time)
cd /Users/shrravan/Documents/Quantus
python train/model.py

# 2. Run CLI
python cli/main.py

# 3. Use it
📝 AAPL                       # predict stock
📝 TSLA                       # predict another
📝 quit                       # exit
```

---

## 🧠 Groq AI Setup (Optional)

Groq explanations are **optional**. The CLI works without it, just no AI analysis.

**To enable Groq AI:**

1. Get free API key: [https://groq.com](https://groq.com)
2. Set environment variable:
   ```bash
   echo 'export GROQ_API_KEY=your_key_here' >> ~/.zshrc
   source ~/.zshrc
   ```
3. Run CLI — Groq auto-works ✅

**Without Groq:** CLI still predicts stocks (BUY/HOLD/SELL), just no explanation.

---

## 📊 Features

- **Neural Network**: MLP (11→64→32→3) for 3-class classification (BUY/HOLD/SELL)
- **Technical Indicators**: RSI, MACD, EMA (20/50/200), volatility, momentum
- **Groq AI**: 3 reasons + 3 risks explanation (neutral analysis) [optional]
- **Caching**: 24h data cache (skip API fetch)
- **Interactive CLI**: Type multiple tickers without restarting

---

## 🏗️ Project Structure

```
Quantus/
├── cli/
│   └── main.py              # Interactive CLI (entry point)
├── pipeline/
│   ├── data_fetcher.py      # yfinance + Finnhub API
│   ├── indicators.py        # Technical indicator computation
│   └── dataset.py           # Dataset preparation
├── train/
│   └── model.py             # MLP training + evaluation
├── models/
│   └── stock_classifier.pt  # Trained model (after training)
├── config/                  # Configuration files
├── tests/                   # Unit tests
├── .cache/                  # Auto-generated cache (24h)
```

---

## 🧮 Model Architecture

```
Input (11 features) → Hidden1 (64) → Hidden2 (32) → Output (3)
                    ↓           ↓
                  Dropout       Dropout
```

**Features**: RSI, MACD, EMA_20/50/200, Price_Change, Vol_Change, Volatility, MA_20/50/200

**Output**: 
- `0` → SELL 📉
- `1` → HOLD ➡️
- `2` → BUY 📈

---

## 📖 CLI Commands

| Command | Description |
|---------|-------------|
| `AAPL` | Predict stock |
| `period=1y` | Set data period (default: 5y) |
| `interval=1h` | Set data interval (default: 1d) |
| `groq=KEY` | Set Groq API key |
| `no-cache` | Fetch fresh data (skip cache) |
| `quit` | Exit CLI |
| `help` | Show commands |

---

## 🔧 Setup

```bash
# Install dependencies
pip install torch yfinance finnhub-python pandas ta numpy click groq

# Train model
python train/model.py

# Run CLI
python cli/main.py
```

---

## 🎯 Performance

- **Train Accuracy**: ~73-80%
- **Test Accuracy**: ~71-73%
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

**Setup**: `groq=your_api_key_here` in CLI

---

## 🚦 Usage Example

```bash
$ python cli/main.py

  Welcome to Quantus, a quantative trading machine learning program using
  networks built in pytorch that uses neural networks to predict stock outcomes.

  Type a ticker (or "quit" to exit)
  Commands: ticker, period=X, interval=X, groq=KEY, quit

  📝 groq=abc123xyz
  ✅ Groq API: set

  📝 AAPL
  🚀 AAPL | 5y | 1d
  ✅ 1255 OHLCV rows

  🎯 BUY 📈
     Neural Network Confidence: 74.2%
     Model: MLP (11→64→32→3)

  🧠 Groq AI Analysis...
    RSI momentum bullish, MACD crossover positive, EMA trend up
    Risks: Volatility high, Volume low, Market uncertain

  ✅ AAPL → BUY 📈

  📝 quit
  ✅ Bye! 👋
```

---

## 🛠️ Tech Stack

- **Python 3.14**
- **PyTorch** (Neural Network)
- **yfinance** (Price data)
- **Finnhub** (News sentiment)
- **TA-Lib** (Technical indicators)
- **Groq** (AI explanations)
- **Click** (CLI)

---

## 📝 License

MIT License - See [LICENSE](LICENSE)
