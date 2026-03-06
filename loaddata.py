import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# load tickers 
with open('tickers.txt', 'r') as f:
    tickers = []
    for line in f:
        tickers.append(line.strip())

valid_tickers = []
for t in tickers:
    info = yf.Ticker(t).history(period="max")
    if not info.empty:
        valid_tickers.append(t)

print(valid_tickers)
print(f"{len(valid_tickers)} / {len(tickers)} tickers are valid")


# download monthly data for IBB ETF
ibb = yf.download("IBB", start="2014-07-01", end="2025-12-31", interval="1mo", auto_adjust=True)

# Flatten multiindex columns
ibb.columns = ibb.columns.get_level_values(0)

# Move Date from index into column
ibb = ibb.reset_index()

# Now compute return
ibb["ibb_ret_1m"] = ibb["Close"].pct_change(1)

# Keep only what you need
ibb = ibb[["Date", "ibb_ret_1m"]]

print("ibb cols", ibb.columns)

start_date = "2014-02-01"
end_date = "2025-12-31"

price_data = yf.download(
    valid_tickers,
    start=start_date,
    end=end_date,
    interval="1mo",
    auto_adjust=True,
    group_by="ticker"
)

all_data = []

for ticker in valid_tickers:
    df = price_data[ticker].copy()
    df = df.reset_index()
    
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
    
dataset = pd.concat(all_data, ignore_index=True)
print("Premerge dataset cols", dataset.columns)


# merge the ETF data on date
dataset = dataset.merge(
    ibb[["Date", "ibb_ret_1m"]],
    how="left",
    on="Date"
)

print("Post merge dataset", dataset.columns)
# Target (next month return)
dataset["target"] = (
    dataset.groupby("ticker")["ret_1m"].shift(-1)
)

# OHE the ticker names
dataset = pd.get_dummies(dataset, columns=["ticker"])

ticker_cols = [col for col in dataset.columns if col.startswith("ticker_")]

dataset = dataset.dropna()
dataset = dataset[
    (dataset["Date"] >= "2015-01-01") &
    (dataset["Date"] <= "2025-12-31")
] 
# print(dataset.head())

# Feature selection
features = [
    "ret_1m", "ret_3m", "ret_6m",
    "vol_3m", "vol_6m", "volume_z", "ibb_ret_1m"
]

features = features + ticker_cols

# Split 
split_date = "2022-01-01"
train = dataset[dataset["Date"] < split_date]
test  = dataset[dataset["Date"] >= split_date]

X_train = train[features].values
y_train = train["target"].values

X_test  = test[features].values
y_test  = test["target"].values

print(train.shape)
print(test.shape)

# Scale - no need to scale y 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(dataset.head())

# Save
pd.DataFrame(X_train_scaled, columns=features).to_csv("X_train_scaled.csv", index=False)
pd.DataFrame(X_test_scaled, columns=features).to_csv("X_test_scaled.csv", index=False)
pd.Series(y_train, name="target").to_csv("y_train.csv", index=False)
pd.Series(y_test, name="target").to_csv("y_test.csv", index=False)

