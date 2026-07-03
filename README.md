# Trader Performance vs Market Sentiment Analysis

## Objective

This project analyzes the relationship between Bitcoin market sentiment (Fear & Greed Index) and trader performance using Hyperliquid historical trading data.

---

## Dataset

- fear_greed_index.csv
- historical_data.csv

---

## Technologies

- Python
- Pandas
- NumPy
- Matplotlib

---

## Project Workflow

1. Load datasets
2. Data preprocessing
3. Merge datasets
4. Feature engineering
5. Exploratory Data Analysis
6. Trader Segmentation
7. Business Insights
8. Strategy Recommendations

---

## Visualizations

- Average Closed PnL by Market Sentiment
- Win Rate by Market Sentiment
- Average Trade Size by Market Sentiment
- Number of Trades Per Day
- Long / Short Ratio

---

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python analysis.py
```

or open

```
Trader_Sentiment_Analysis.ipynb
```

using Jupyter Notebook.

---

## Key Findings

- Traders achieve higher profitability during Greed and Extreme Greed periods.
- Win rate increases under positive market sentiment.
- Traders execute larger trades during bullish markets.
- Frequent traders participate more consistently.

---

## Recommendations

- Reduce position size during Fear periods.
- Increase trading activity during Greed periods.
- Follow disciplined risk management practices.
