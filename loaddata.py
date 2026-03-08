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
    try:
        ticker = yf.Ticker(t)
        if ticker.fast_info is not None:
            valid_tickers.append(t)
    except Exception:
        print(t)
        pass

print(f"{len(valid_tickers)} / {len(tickers)} tickers are valid")

# download monthly data for IBB ETF
ibb = yf.download("IBB", start="2009-01-01", end="2025-12-31", interval="1mo", auto_adjust=True)

# Flatten multiindex columns
ibb.columns = ibb.columns.get_level_values(0)

# Move Date from index into column
ibb = ibb.reset_index()

# Now compute return
ibb["ibb_ret_1m"] = ibb["Close"].pct_change(1)

# Keep only what you need
ibb = ibb[["Date", "ibb_ret_1m"]]

print("ibb cols", ibb.columns)

start_date = "2009-01-01"
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
    df["ret_12m"] = df["Close"].pct_change(12)

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
dataset = pd.get_dummies(dataset, columns=["ticker"], dtype=float)

ticker_cols = [col for col in dataset.columns if col.startswith("ticker_")]

# pd.DataFrame(dataset).to_csv("dataset.csv", index=False)

dataset = dataset.dropna()

dataset = dataset[
    (dataset["Date"] >= "2010-01-01") &
    (dataset["Date"] <= "2025-12-31")
] 

# Ensure 'Date' column is datetime
dataset['Date'] = pd.to_datetime(dataset['Date'])

# Optional: extract month for grouping later
# dataset['month'] = dataset['Date'].dt.to_period('M')

# Sort by date (and ticker if you want consistent order within month)
dataset = dataset.sort_values(['Date'] + ticker_cols).reset_index(drop=True)

print("Sorted dataset length:", dataset.shape)


# Top 20% flag per month
dataset['top20'] = dataset.groupby('Date')['target'].transform(
    lambda x: (x >= x.quantile(0.9)).astype(float)
)

pd.DataFrame(dataset).to_csv("dataset.csv", index=False)


print("Full dataset length", dataset.shape)



# Feature selection
num_features = [
    "ret_1m", "ret_3m", "ret_6m", "ret_12m",
    "vol_3m", "vol_6m", "volume_z", "ibb_ret_1m"
]
ohe_features = ticker_cols

features = num_features + ohe_features


# Split 
split_date = "2022-01-01"
train = dataset[dataset["Date"] < split_date]
test  = dataset[dataset["Date"] >= split_date]

train['month'] = train['Date'].dt.to_period('M')
test['month'] = test['Date'].dt.to_period('M')

print("train", train.shape)
print("test", test.shape)

# Scale - no need to scale y 
scaler = StandardScaler()

train_scaled = train.copy()
test_scaled = test.copy()

# only scale numeric
train_scaled[num_features] = scaler.fit_transform(train[num_features])
test_scaled[num_features] = scaler.transform(test[num_features])

# split into X and y
X_train = train_scaled[features].values
y_train = train["top20"].values

X_test = test_scaled[features].values
y_test = test["top20"].values

print(dataset.head())

# Save
pd.DataFrame(X_train, columns=features).to_csv("X_train.csv", index=False)
pd.DataFrame(X_test, columns=features).to_csv("X_test.csv", index=False)
pd.Series(y_train, name="target").to_csv("y_train.csv", index=False)
pd.Series(y_test, name="target").to_csv("y_test.csv", index=False)

