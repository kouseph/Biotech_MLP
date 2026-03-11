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
ibb = yf.download("IBB", start="2009-01-01", end="2026-01-01", interval="1mo", auto_adjust=True)
ibb.columns = ibb.columns.get_level_values(0)
ibb = ibb.reset_index()

# compute return
ibb["ibb_ret_1m"] = ibb["Close"].pct_change(1)

ibb = ibb[["Date", "ibb_ret_1m"]]

# download raw stock data
start_date = "2009-01-01"
end_date = "2026-01-01"

price_data = yf.download(
    valid_tickers,
    start=start_date,
    end=end_date,
    interval="1mo",
    auto_adjust=True,
    group_by="ticker"
)

all_data = []
print(price_data.shape)
print(price_data.head())
MAX_FUNDAMENTAL_AGE_DAYS = 180
FUNDAMENTAL_REPORT_LAG_DAYS = 45

def first_present_value(row, candidate_cols):
    for col in candidate_cols:
        if col in row.index and pd.notna(row[col]):
            return row[col]
    return np.nan

def compute_trend(series):
    clean = pd.Series(series).dropna()
    if len(clean) < 2:
        return np.nan
    x = np.arange(len(clean), dtype=float)
    y = clean.values.astype(float)
    return np.polyfit(x, y, 1)[0]

for ticker in valid_tickers:
    df = price_data[ticker].copy()
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).astype("datetime64[ns]")

    # if df["Close"].isna().all() or df["Volume"].isna().all():
    #     print(f"{ticker} has only NaNs, skipping.")
    #     continue
    # if df[["Close", "Volume"]].isna().any().any():
    #     print(f"{ticker} has some missing values.")
    
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

    df["vol_ratio"] = df["vol_3m"] / df["vol_6m"]
    # recent absolute vol
    df["recent_vol"] = abs(df["ret_1m"])

    # Fundamental features (quarterly), aligned to monthly rows
    tkr = yf.Ticker(ticker)
    cash_quarterly = pd.DataFrame(
        columns=[
            "cash_report_date",
            "cash_available_date",
            "total_cash",
            "total_cash_log",
            "total_cash_ge_1b",
            "total_cash_trend_4q",
            "total_cash_qoq_pct",
            "total_cash_yoy_pct",
        ]
    )
    lfcf_quarterly = pd.DataFrame(
        columns=[
            "lfcf_report_date",
            "lfcf_available_date",
            "lfcf",
            "lfcf_trend_4q",
            "lfcf_improving_4q",
            "lfcf_qoq_change",
            "lfcf_yoy_change",
        ]
    )

    try:
        qbs = tkr.quarterly_balance_sheet.T.reset_index().rename(columns={"index": "cash_report_date"})
        qbs["cash_report_date"] = pd.to_datetime(qbs["cash_report_date"]).astype("datetime64[ns]")
        qbs["total_cash"] = qbs.apply(
            lambda row: first_present_value(
                row,
                [
                    "Total Cash",
                    "Cash Cash Equivalents And Short Term Investments",
                    "Cash And Cash Equivalents",
                    "Cash Financial",
                ],
            ),
            axis=1,
        )
        qbs["total_cash_trend_4q"] = qbs["total_cash"].rolling(4).apply(compute_trend, raw=False)
        qbs["total_cash_log"] = np.log1p(qbs["total_cash"].clip(lower=0))
        qbs["total_cash_ge_1b"] = (qbs["total_cash"] >= 1_000_000_000).astype(float)
        qbs = qbs.sort_values("cash_report_date")
        qbs["cash_available_date"] = (
            qbs["cash_report_date"] + pd.Timedelta(days=FUNDAMENTAL_REPORT_LAG_DAYS)
        ).astype("datetime64[ns]")
        qbs["total_cash_qoq_pct"] = qbs["total_cash"].pct_change(1)
        qbs["total_cash_yoy_pct"] = qbs["total_cash"].pct_change(4)
        cash_quarterly = qbs[
            [
                "cash_report_date",
                "cash_available_date",
                "total_cash",
                "total_cash_log",
                "total_cash_ge_1b",
                "total_cash_trend_4q",
                "total_cash_qoq_pct",
                "total_cash_yoy_pct",
            ]
        ]
    except Exception:
        pass

    try:
        qcf = tkr.quarterly_cashflow.T.reset_index().rename(columns={"index": "lfcf_report_date"})
        qcf["lfcf_report_date"] = pd.to_datetime(qcf["lfcf_report_date"]).astype("datetime64[ns]")
        qcf["lfcf"] = qcf.apply(
            lambda row: first_present_value(
                row,
                [
                    "Levered Free Cash Flow",
                    "Free Cash Flow",
                ],
            ),
            axis=1,
        )
        qcf["lfcf_trend_4q"] = qcf["lfcf"].rolling(4).apply(compute_trend, raw=False)
        qcf["lfcf_improving_4q"] = qcf["lfcf"] - qcf["lfcf"].shift(3)
        qcf = qcf.sort_values("lfcf_report_date")
        qcf["lfcf_available_date"] = (
            qcf["lfcf_report_date"] + pd.Timedelta(days=FUNDAMENTAL_REPORT_LAG_DAYS)
        ).astype("datetime64[ns]")
        qcf["lfcf_qoq_change"] = qcf["lfcf"].diff(1)
        qcf["lfcf_yoy_change"] = qcf["lfcf"].diff(4)
        lfcf_quarterly = qcf[
            [
                "lfcf_report_date",
                "lfcf_available_date",
                "lfcf",
                "lfcf_trend_4q",
                "lfcf_improving_4q",
                "lfcf_qoq_change",
                "lfcf_yoy_change",
            ]
        ]
    except Exception:
        pass

    df = df.sort_values("Date")
    if not cash_quarterly.empty:
        df = pd.merge_asof(
            df,
            cash_quarterly.sort_values("cash_report_date"),
            left_on="Date",
            right_on="cash_available_date",
            direction="backward",
        )
    else:
        df["cash_report_date"] = pd.NaT
        df["cash_available_date"] = pd.NaT
        df["total_cash"] = np.nan
        df["total_cash_log"] = np.nan
        df["total_cash_ge_1b"] = np.nan
        df["total_cash_trend_4q"] = np.nan
        df["total_cash_qoq_pct"] = np.nan
        df["total_cash_yoy_pct"] = np.nan

    if not lfcf_quarterly.empty:
        df = pd.merge_asof(
            df,
            lfcf_quarterly.sort_values("lfcf_report_date"),
            left_on="Date",
            right_on="lfcf_available_date",
            direction="backward",
        )
    else:
        df["lfcf_report_date"] = pd.NaT
        df["lfcf_available_date"] = pd.NaT
        df["lfcf"] = np.nan
        df["lfcf_trend_4q"] = np.nan
        df["lfcf_improving_4q"] = np.nan
        df["lfcf_qoq_change"] = np.nan
        df["lfcf_yoy_change"] = np.nan

    df["days_since_total_cash_report"] = (df["Date"] - df["cash_available_date"]).dt.days
    df["days_since_lfcf_report"] = (df["Date"] - df["lfcf_available_date"]).dt.days

    stale_cash = df["days_since_total_cash_report"].isna() | (df["days_since_total_cash_report"] > MAX_FUNDAMENTAL_AGE_DAYS)
    stale_lfcf = df["days_since_lfcf_report"].isna() | (df["days_since_lfcf_report"] > MAX_FUNDAMENTAL_AGE_DAYS)

    df.loc[
        stale_cash,
        [
            "total_cash",
            "total_cash_log",
            "total_cash_ge_1b",
            "total_cash_trend_4q",
            "total_cash_qoq_pct",
            "total_cash_yoy_pct",
        ],
    ] = np.nan
    df.loc[
        stale_lfcf,
        [
            "lfcf",
            "lfcf_trend_4q",
            "lfcf_improving_4q",
            "lfcf_qoq_change",
            "lfcf_yoy_change",
        ],
    ] = np.nan

    df["ticker"] = ticker
    
    all_data.append(df)
    
dataset = pd.concat(all_data, ignore_index=True)

# Ensure 'Date' column is datetime
dataset['Date'] = pd.to_datetime(dataset['Date'])

# merge the ETF data on date
dataset = dataset.merge(
    ibb[["Date", "ibb_ret_1m"]],
    how="left",
    on="Date"
)



# Missing-value indicators for fundamentals (let model learn data availability)
dataset["total_cash_missing"] = dataset["total_cash"].isna().astype(float)
dataset["lfcf_missing"] = dataset["lfcf"].isna().astype(float)
dataset["fundamentals_available"] = ((dataset["total_cash_missing"] == 0) & (dataset["lfcf_missing"] == 0)).astype(float)
dataset["fundamental_recency_days"] = dataset[["days_since_total_cash_report", "days_since_lfcf_report"]].min(axis=1)
dataset["lfcf_to_total_cash"] = dataset["lfcf"] / dataset["total_cash"].replace(0, np.nan)

# calculate target (1 month in future)
dataset["target"] = (
    dataset.groupby("ticker")["ret_1m"].shift(-1).abs()
)


# # Top 20% flag per month
# dataset['top20'] = dataset.groupby('Date')['target'].transform(
#     lambda x: (x >= x.quantile(0.8)).astype(float)
# )

# OHE the ticker names
dataset = pd.get_dummies(dataset, columns=["ticker"], dtype=float)
ticker_cols = [col for col in dataset.columns if col.startswith("ticker_")]

# Drop only rows that are unusable for supervised learning target/price signals.
essential_monthly_features = [
    "ret_1m",
    "ret_3m",
    "ret_6m",
    "ret_12m",
    "vol_3m",
    "vol_6m",
    "volume_z",
    "ibb_ret_1m",
    "vol_ratio",
    "recent_vol",
]
dataset = dataset.dropna(subset=essential_monthly_features + ["target"])

# drop rows outside of time frame 
# start = "2010-01-01"
start = "2020-01-01"
end = "2026-01-01"
dataset = dataset[
    (dataset["Date"] >= start) &
    (dataset["Date"] <= end)
] 

pd.DataFrame(dataset).to_csv("yfinancedata.csv", index=False)

# Sort by date (and ticker if you want consistent order within month)
# dataset = dataset.sort_values(['Date'] + ticker_cols).reset_index(drop=True)

print("Full dataset length", dataset.shape)

# now, exporting into usable files for model
continuous_num_features = [
    "ret_1m", "ret_3m", "ret_6m", "ret_12m",
    "vol_3m", "vol_6m", "volume_z", "ibb_ret_1m",
    "total_cash", "total_cash_log", "total_cash_trend_4q",
    "total_cash_qoq_pct", "total_cash_yoy_pct",
    "lfcf", "lfcf_trend_4q", "lfcf_improving_4q", "lfcf_qoq_change", "lfcf_yoy_change",
    "lfcf_to_total_cash",
    "days_since_total_cash_report", "days_since_lfcf_report", "fundamental_recency_days",
    "vol_ratio", "recent_vol",
]
binary_num_features = [
    "total_cash_ge_1b",
    "total_cash_missing", "lfcf_missing", "fundamentals_available"
]
num_features = continuous_num_features + binary_num_features
ohe_features = ticker_cols

features = num_features + ohe_features 

# Split 
split_date = "2025-01-01"
train = dataset[dataset["Date"] < split_date]
test  = dataset[dataset["Date"] >= split_date]

# Robustify quarterly dynamic features against extreme outliers.
dynamic_feature_clip_cols = [
    "total_cash_qoq_pct",
    "total_cash_yoy_pct",
    "lfcf_qoq_change",
    "lfcf_yoy_change",
    "lfcf_improving_4q",
    "lfcf_to_total_cash",
]
for col in dynamic_feature_clip_cols:
    if col in train.columns:
        low = train[col].quantile(0.01)
        high = train[col].quantile(0.99)
        if pd.notna(low) and pd.notna(high):
            train.loc[:, col] = train[col].clip(low, high)
            test.loc[:, col] = test[col].clip(low, high)

# Impute fundamentals/age features from train medians (avoid zero-imputing continuous data)
continuous_fundamental_features = [
    "total_cash", "total_cash_log", "total_cash_trend_4q",
    "total_cash_qoq_pct", "total_cash_yoy_pct",
    "lfcf", "lfcf_trend_4q", "lfcf_improving_4q", "lfcf_qoq_change", "lfcf_yoy_change",
    "lfcf_to_total_cash",
    "days_since_total_cash_report", "days_since_lfcf_report", "fundamental_recency_days",
]
median_fill_values = train[continuous_fundamental_features].median()
median_fill_values = median_fill_values.fillna(0.0)
train.loc[:, continuous_fundamental_features] = train[continuous_fundamental_features].fillna(median_fill_values)
test.loc[:, continuous_fundamental_features] = test[continuous_fundamental_features].fillna(median_fill_values)

train.loc[:, ["total_cash_ge_1b", "total_cash_missing", "lfcf_missing", "fundamentals_available"]] = (
    train[["total_cash_ge_1b", "total_cash_missing", "lfcf_missing", "fundamentals_available"]].fillna(0.0)
)
test.loc[:, ["total_cash_ge_1b", "total_cash_missing", "lfcf_missing", "fundamentals_available"]] = (
    test[["total_cash_ge_1b", "total_cash_missing", "lfcf_missing", "fundamentals_available"]].fillna(0.0)
)

# Stabilize numeric matrix for modeling: remove inf and fill any remaining NaNs
train.loc[:, continuous_num_features] = train[continuous_num_features].replace([np.inf, -np.inf], np.nan)
test.loc[:, continuous_num_features] = test[continuous_num_features].replace([np.inf, -np.inf], np.nan)
continuous_fill_values = train[continuous_num_features].median().fillna(0.0)
train.loc[:, continuous_num_features] = train[continuous_num_features].fillna(continuous_fill_values)
test.loc[:, continuous_num_features] = test[continuous_num_features].fillna(continuous_fill_values)

test = test.copy()
test['month'] = pd.to_datetime(test['Date'])
test_months = test["month"].values
pd.Series(test_months, name="month").to_csv("test_months.csv", index=False)
pd.Series(test["fundamentals_available"].values, name="fundamentals_available").to_csv("test_fundamentals_available.csv", index=False)
pd.Series(test["fundamental_recency_days"].values, name="fundamental_recency_days").to_csv("test_fundamental_recency_days.csv", index=False)

print("train", train.shape)
print("test", test.shape)

# Scale - no need to scale y 
scaler = StandardScaler()

train_scaled = train.copy()
test_scaled = test.copy()

# only scale continuous numeric features (keep binary flags as 0/1)
train_scaled[continuous_num_features] = scaler.fit_transform(train[continuous_num_features])
test_scaled[continuous_num_features] = scaler.transform(test[continuous_num_features])

# split into X and y
X_train = train_scaled[features].values
y_train = train["target"].values

X_test = test_scaled[features].values
y_test = test["target"].values

print(dataset.head())


pd.DataFrame(dataset).to_csv("big_data.csv", index = False)

# Save
# pd.DataFrame(X_train, columns=features).to_csv("X_train.csv", index=False)
# pd.DataFrame(X_test, columns=features).to_csv("X_test.csv", index=False)
# pd.Series(y_train, name="target").to_csv("y_train.csv", index=False)
# pd.Series(y_test, name="target").to_csv("y_test.csv", index=False)
#
