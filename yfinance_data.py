import yfinance as yf
import pandas as pd

def get_stock_data(ticker_symbol):
    stock = yf.Ticker(ticker_symbol)
    
    # 1. Get Income Statement & Balance Sheet
    # This gives you the last 4-5 years of data
    income_stmt = stock.financials
    balance_sheet = stock.balance_sheet
    
    # 2. Extract specific rows for your features
    # Revenue Growth
    rev = income_stmt.loc['Total Revenue']
    rev_growth = rev.pct_change(periods=-1) # Growth year-over-year
    
    # Debt to Equity
    total_debt = balance_sheet.loc['Total Debt']
    equity = balance_sheet.loc['Stockholders Equity']
    debt_to_equity = total_debt / equity
    
    # 3. Market Cap and PE Ratio
    # yf.info provides current metrics, but for historical, 
    # you'd multiply historical Price by Shares Outstanding.
    hist_price = stock.history(start="2016-01-01", end="2022-12-31")
    
    hist_price.index = hist_price.index.tz_localize(None)
    return {
        "Rev Growth": rev_growth,
        "D/E Ratio": debt_to_equity,
        "Price History": hist_price['Close']
    }

# Example usage
data = get_stock_data("AMGN")

df = pd.DataFrame(data)

print(df)

df.to_csv("hey.csv")
