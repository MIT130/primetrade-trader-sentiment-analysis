import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# Load Datasets
# ==========================================================
fear_greed = pd.read_csv("fear_greed_index.csv")
historical = pd.read_csv("historical_data.csv")

# ==========================================================
# Dataset Information
# ==========================================================
print("=" * 50)
print("Dataset Information")
print("=" * 50)

print(f"Fear & Greed Dataset Shape : {fear_greed.shape}")
print(f"Historical Dataset Shape   : {historical.shape}")

print("\nMissing Values (Fear & Greed)")
print(fear_greed.isnull().sum())

print("\nMissing Values (Historical)")
print(historical.isnull().sum())

print("\nDuplicate Rows")
print(f"Fear & Greed : {fear_greed.duplicated().sum()}")
print(f"Historical   : {historical.duplicated().sum()}")

# ==========================================================
# Data Preprocessing
# ==========================================================
historical["Timestamp IST"] = pd.to_datetime(
    historical["Timestamp IST"],
    errors="coerce"
)
historical["date"] = historical["Timestamp IST"].dt.date

fear_greed["date"] = pd.to_datetime(
    fear_greed["date"],
    errors="coerce"
).dt.date

# ==========================================================
# Merge Datasets
# ==========================================================
merged_data = historical.merge(
    fear_greed[["date", "classification"]],
    on="date",
    how="left"
)

print("\nMerged Dataset Shape:", merged_data.shape)

# ==========================================================
# 1. Average Closed PnL
# ==========================================================
avg_pnl = (
    merged_data
    .groupby(["classification"])["Closed PnL"]
    .mean()
    .reset_index()
)

print("\nAverage PnL")
print(avg_pnl.head())

plt.figure(figsize=(8, 5))
plt.bar(
    avg_pnl["classification"],
    avg_pnl["Closed PnL"],
    color="skyblue"
)
plt.title("Average Closed PnL by Market Sentiment")
plt.xlabel("Market Sentiment")
plt.ylabel("Average Closed PnL")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.show()
# plt.savefig("Average_Closed_PnL_by_Market_Sentiment.png", dpi=300)

# ==========================================================
# 2. Win Rate
# ==========================================================
win_rate = (
    merged_data
    .assign(win=merged_data["Closed PnL"] > 0)
    .groupby("classification")["win"]
    .mean()
    .mul(100)
    .reset_index(name="Win Rate (%)")
)

print("\nWin Rate")
print(win_rate)

plt.figure(figsize=(8, 5))
plt.bar(
    win_rate["classification"],
    win_rate["Win Rate (%)"],
    color="lightgreen"
)
plt.title("Win Rate by Market Sentiment")
plt.xlabel("Market Sentiment")
plt.ylabel("Win Rate (%)")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.show()
# plt.savefig("Win_Rate_by_Market_Sentiment.png", dpi=300)

# ==========================================================
# 3. Average Trade Size
# ==========================================================
avg_trade_size = (
    merged_data
    .groupby("classification")["Size Tokens"]
    .mean()
    .reset_index(name="Average Trade Size")
)

print("\nAverage Trade Size")
print(avg_trade_size)

plt.figure(figsize=(8, 5))
plt.bar(
    avg_trade_size["classification"],
    avg_trade_size["Average Trade Size"],
    color="orange"
)
plt.title("Average Trade Size by Market Sentiment")
plt.xlabel("Market Sentiment")
plt.ylabel("Average Size (Tokens)")
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.show()
# plt.savefig("Average_Trade_Size_by_Market_Sentiment.png", dpi=300)

# ==========================================================
# 4. Trades Per Day
# ==========================================================
trades_per_day = (
    merged_data
    .groupby("date")
    .size()
    .reset_index(name="Trades")
)

print("\nTrades Per Day")
print(trades_per_day.head())

plt.figure(figsize=(10, 5))
plt.bar(
    trades_per_day["date"],
    trades_per_day["Trades"],
    color="purple"
)
plt.title("Number of Trades Per Day")
plt.xlabel("Date")
plt.ylabel("Trades")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
# plt.savefig("Number_of_Trades_Per_Day.png", dpi=300)

# ==========================================================
# 5. Long / Short Ratio
# ==========================================================
long_short = (
    merged_data
    .groupby(["date", "Direction"])
    .size()
    .unstack(fill_value=0)
)

# Standardize column names
long_short.columns = long_short.columns.str.upper()

# Add BUY and SELL columns if missing
if "BUY" not in long_short.columns:
    long_short["BUY"] = 0

if "SELL" not in long_short.columns:
    long_short["SELL"] = 1

# Calculate Long / Short Ratio
long_short["LONG_SHORT_RATIO"] = (
    long_short["BUY"] / long_short["SELL"]
)

long_short = long_short.reset_index()

print("\nLong / Short Ratio")
print(long_short.head())

plt.figure(figsize=(10, 5))
plt.bar(
    long_short["date"],
    long_short["LONG_SHORT_RATIO"],
    color="red"
)
plt.title("Long / Short Ratio Over Time")
plt.xlabel("Date")
plt.ylabel("Long / Short Ratio")
plt.xticks(rotation=45)
plt.grid(axis="y", linestyle="--", alpha=0.5)
plt.tight_layout()
plt.show()
# plt.savefig("Long_or_Short_Ratio_Over_Time.png", dpi=300)

# ==========================================================
# 6. Frequent vs Infrequent Traders
# ==========================================================
trade_count = merged_data.groupby("Account").size()

threshold = trade_count.median()

segments = np.where(
    trade_count > threshold,
    "Frequent",
    "Infrequent"
)

trader_segment = pd.DataFrame({
    "Account": trade_count.index,
    "Trade Count": trade_count.values,
    "Segment": segments
})

print("\nTrader Segments")
print(trader_segment.head())

# ==========================================================
# 7. Consistent Winners
# ==========================================================
win_rate_account = (
    merged_data
    .assign(win=merged_data["Closed PnL"] > 0)
    .groupby("Account")["win"]
    .mean()
    .mul(100)
    .reset_index(name="Win Rate (%)")
)

win_rate_account["Segment"] = np.where(
    win_rate_account["Win Rate (%)"] >= 60,
    "Consistent Winner",
    "Others"
)

print("\nConsistent Winners")
print(win_rate_account.head())

# ==========================================================
# 8. Large vs Small Position Traders
# ==========================================================
size = (
    merged_data
    .groupby("Account")["Size Tokens"]
    .mean()
)

median_size = size.median()

position_segment = np.where(
    size > median_size,
    "Large Position",
    "Small Position"
)

position_df = pd.DataFrame({
    "Account": size.index,
    "Average Position Size": size.values,
    "Segment": position_segment
})

print("\nPosition Size Segments")
print(position_df.head())