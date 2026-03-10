import yfinance as yf
import pandas as pd


ticker = yf.Ticker("PFE")
lfc = ticker.get_cashflow(freq = 'quarterly')
# print(ticker.balance_sheet)


print(lfc)
