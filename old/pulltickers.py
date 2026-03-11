import pandas as pd

url = "https://www.ishares.com/us/products/239699/ishares-biotechnology-etf/1467271812596.ajax?fileType=csv&fileName=IBB_holdings&dataType=fund"

df = pd.read_csv(url, skiprows=9)

tickers = df["Ticker"].dropna().unique().tolist()
names = df["Name"].dropna().unique().tolist()

tickers = tickers[:-2]

with open('tickers.txt', 'w') as f:
    for t in tickers:
        f.write(f"{t}\n")
print(tickers[:5])
print(names[:5])
print(len(tickers))