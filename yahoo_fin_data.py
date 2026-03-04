import yfinance as yf
import pandas as pd

microsoft = yf.Ticker('MSFT')
dict =  microsoft.info
df = pd.DataFrame.from_dict(dict,orient='index')
df = df.reset_index()
print(df)
