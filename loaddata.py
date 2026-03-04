import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

tickers = [
    "AMGN"
]

start_date = "2010-01-01"
end_date = "2024-12-31"

price_data = yf.download(
    tickers,
    start=start_date,
    end=end_date,
    interval="1mo",
    auto_adjust=True,
    group_by="ticker"
)

all_data = []

for ticker in tickers:
    df = price_data[ticker].copy()

    # Features
    df["ret_1m"] = df["Close"].pct_change(1)
    df["ret_3m"] = df["Close"].pct_change(3)
    df["ret_6m"] = df["Close"].pct_change(6)

    df["vol_3m"] = df["Close"].pct_change().rolling(3).std()
    df["vol_6m"] = df["Close"].pct_change().rolling(6).std()

    df["volume_z"] = (
        df["Volume"] - df["Volume"].rolling(12).mean()
    ) / df["Volume"].rolling(12).std()

    df["ticker"] = ticker
    all_data.append(df)

dataset = pd.concat(all_data).reset_index()

# Target (next month return)
dataset["target"] = (
    dataset.groupby("ticker")["ret_1m"].shift(-1)
)

# dataset = dataset.dropna()

# Feature selection
features = [
    "ret_1m", "ret_3m", "ret_6m",
    "vol_3m", "vol_6m", "volume_z"
]

X = dataset[features].values
y = dataset["target"].values

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Optional: save
pd.DataFrame(X_scaled, columns=features).to_csv("X.csv", index=False)
pd.Series(y, name="target").to_csv("y.csv", index=False)


print(dataset.head())
