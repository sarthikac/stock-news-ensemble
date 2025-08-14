# 📈 Stock Price Movement Prediction with News + Technical Indicators

Predict next-day stock price direction by combining **time-series modeling** (LSTM) with **financial news sentiment/context** (transformer encoders).  
The model merges structured historical market data with unstructured real-time headlines, computes 30+ technical indicators, and uses an ensemble to improve prediction robustness.

---

## 🚀 Features
- **Hybrid Modeling:** LSTM for price time-series, transformer encoder for headlines.
- **Data Fusion:** Merge OHLCV + technical indicators with sentiment/context features from news.
- **Custom Technical Indicators:** Over 30 indicators (e.g., RSI, MACD, Bollinger Bands).
- **Ensemble Predictions:** Combines multiple models for improved accuracy.
- **Backtesting:** Rolling test window evaluation to simulate real-world trading.
- **Performance:** Achieved **68% directional accuracy** in validation.

---

## 📦 Quickstart

### 1️⃣ Clone & Setup
```bash
git clone https://github.com/yourusername/stock-price-news-prediction.git
cd stock-price-news-prediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```
### 2️⃣ API Key (Optional)
If you want live headlines from NewsAPI:

Sign up at newsapi.org/register to get a free API key.

Copy .env.example to .env:
```bash
cp .env.example .env
```
Edit .env and add your key:
```
NEWSAPI_KEY=your_api_key_here
```
If you skip this step, you can alternatively load headlines from a CSV (see data/README.md).

### 3️⃣ Download Data & Headlines
Example for Apple stock:
```
python src/data/download_data.py --ticker AAPL --start 2018-01-01 --end 2024-12-31
```
This saves:

data/raw/price_AAPL.csv – OHLCV + technical indicators

data/raw/news_AAPL.csv – Date, headline

### 4️⃣ Train the Model
```bash
python src/train.py --config configs/default.yaml
```
Results and logs will be saved to outputs/.

### 📂 Project Structure
```
.
├── configs/           # Model & training configs
├── data/              # Raw & processed datasets
│   └── README.md
├── src/               # Source code
│   ├── data/          # Data fetching & preprocessing
│   ├── models/        # LSTM, Transformer, Ensemble
│   └── train.py       # Training script
├── requirements.txt   # Dependencies
├── .env.example       # Environment variables template
└── README.md
```
### 📄 Data Folder Guide
See data/README.md for details.

### 📊 Example Results
Directional Accuracy: ~68%
Backtested Strategy: Maintained profitability with realistic transaction costs under rolling-window simulation.

### ⚠️ Disclaimer
This project is for educational and research purposes only.
It is not financial advice. Trading in financial markets involves significant risk.
