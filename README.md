# StockPredictorRL

End-to-end experimentation repo for forecasting MSFT intraday prices with an LSTM, enriching with social sentiment, and evaluating simple rule-based and PPO reinforcement-learning trading strategies.

## Key ideas
- Combine market microstructure (60‑minute candles) with daily social sentiment features.
- Train an LSTM for next-step price prediction and visualize results.
- Use predictions + sentiment in a simple rules engine to simulate trades and PnL.
- Explore RL with PPO to learn a trading policy (notebook-driven, with logs included).

## Project structure
```
.
├── aggr_data/                     # Aggregated and intermediate datasets
│   ├── all_data.csv               # Joined price, predictions, and sentiment (produced by notebooks)
│   ├── sentiments_2016_to_2024.csv# Daily sentiment scores (often moved here after generation)
│   └── stockprices.csv            # Example aggregated prices
├── graphs/                        # Visualizations
│   ├── lstm_testing_results.png
│   └── trading_bot.png
├── price_data/                    # 60-min MSFT candles (2016–2024, Alpha Vantage format)
├── sentiment_data/                # Raw JSON sentiment slices (StockGeist API)
├── trained_models/
│   └── LSTM_Price_Predictor_Trial.keras  # Example saved LSTM model
├── Training/
│   └── Logs/                      # PPO training logs (multiple runs)
├── LSTM.ipynb                     # LSTM training & evaluation pipeline
├── RL.ipynb                       # PPO RL experiments (env/prototype and logs)
├── sentiment_extractor.py         # Fetch StockGeist metrics and compute daily sentiment score
├── trader_bot.py                  # Rule-based strategy using LSTM preds + sentiment + ATR position sizing
├── PPO.zip, PPO2.zip              # Saved PPO artifacts (reference)
└── README.md                      # This file
```

## Data sources
- Price data: Alpha Vantage intraday 60‑minute candles for `MSFT` (2016–2024). Files live in `price_data/` and are formatted as CSV with a `timestamp` column in UTC.
- Sentiment data: StockGeist message metrics (daily timeframe). Pulled via `sentiment_extractor.py` into `sentiment_data/` as JSON, then aggregated to daily scores. The script currently writes `sentiments_2016_to_2024.csv` to the project root; move it into `aggr_data/` to match the notebooks.

## Environment setup
- Recommended: Python 3.10+
- Core libraries used across notebooks/scripts:
  - pandas, numpy, matplotlib
  - tensorflow/keras
  - scikit-learn
  - vmdpy (Variational Mode Decomposition for signal processing)
  - gym / gymnasium and stable-baselines3 (PPO)
  - requests, jupyter

Quick setup:
```bash
# from the repo root
python -m venv .venv
source .venv/bin/activate  # on Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install pandas numpy matplotlib tensorflow scikit-learn vmdpy requests jupyter gym gymnasium "stable-baselines3[extra]"
```

Set your StockGeist API token (required for sentiment fetch):
```bash
export STOCKGEIST_API_TOKEN="your_actual_api_token_here"
```

Optional: If you want to refresh price data from Alpha Vantage, you will also need an Alpha Vantage API key (see commented cells in `LSTM.ipynb`).

## Reproducibility workflow
1) Fetch or refresh sentiment data
```bash
python sentiment_extractor.py
# writes:
#  - sentiment_data/YYYY_01_to_YYYY_06.json
#  - sentiment_data/YYYY_07_to_YYYY_12.json
#  - sentiments_2016_to_2024.csv   (project root)

# move to align with notebooks
mv sentiments_2016_to_2024.csv aggr_data/
```
- The script computes a normalized daily sentiment score in [-1, 1] from StockGeist metrics, weighting emotional vs non‑emotional messages.

2) Train/evaluate LSTM and build the modeling dataset
- Open `LSTM.ipynb` and run all cells:
  - Loads 60‑minute price data from `price_data/` and aligns to UTC dates.
  - Loads daily sentiment from `aggr_data/sentiments_2016_to_2024.csv`.
  - Trains an LSTM for price prediction (optionally using VMD for signal decomposition).
  - Saves model as `trained_models/LSTM_Price_Predictor_Trial.keras`.
  - Produces `graphs/lstm_testing_results.png`.
  - Exports a joined dataset (e.g., `aggr_data/all_data.csv`) with columns such as `timestamp`, `close`, `Predictions_lstm`, `sentiment_score`.

3) Run the rule-based trading bot
```bash
python trader_bot.py
# reads:  aggr_data/all_data.csv
# writes: graphs/trading_bot.png
```
Strategy summary in `trader_bot.py`:
- Moving averages: 25 vs 50 periods on `close`.
- Long signal: short MA > long MA, LSTM predicted price > last price, and sentiment > 0.
- Short/sell signal: short MA < long MA, predicted < last, and sentiment < 0.
- Position sizing: risk-based using a modified ATR (SMA of absolute price changes), with stop amount ≈ 2×ATR.

4) Explore reinforcement learning (optional)
- Open `RL.ipynb` to experiment with PPO using `stable-baselines3`.
- A prototype `Portfolio` class is included; PPO logs are under `Training/Logs/` (multiple runs: `PPO_1`, `PPO_2`, …).
- Saved RL artifacts (`PPO.zip`, `PPO2.zip`) are provided as references.

## Notes and assumptions
- Timezones: price data `timestamp` are handled in UTC, daily sentiment is aligned by date.
- Symbols: repository focuses on `MSFT`. Generalization requires adjusting symbols and data loaders.
- Data completeness: `price_data/` and `sentiment_data/` are bundled for convenience; you can refresh them from the respective APIs.
- Hardware: TensorFlow models may benefit from GPU acceleration; CPU works but is slower.

## Troubleshooting
- StockGeist token missing: `sentiment_extractor.py` will fail without `STOCKGEIST_API_TOKEN` set.
- Package versions: If TensorFlow/Keras or SB3 versions conflict, try pinning versions compatible with your Python (e.g., TF 2.12–2.15 with Python 3.10/3.11).
- File paths: Scripts assume execution from the repo root. Adjust paths if running elsewhere.

## Acknowledgments
- Alpha Vantage for intraday price data.
- StockGeist for message-metrics sentiment API.
- TensorFlow/Keras, scikit-learn, vmdpy for modeling.
- Gym/Gymnasium and Stable-Baselines3 for RL.
