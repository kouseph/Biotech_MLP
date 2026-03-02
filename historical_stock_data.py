# import yahoo_fin.stock_info as si
import yfinance as yf
import pandas as pd

# Define your tickers and parameters
tickers = ["LLY", "NVS", "MRK", "NVO", "GSK", "BMY", "JNJ", "MCK", "PFE", "AZN", 
           "ABBV", "COR", "SNY", "ZTS", "HLN", "TAK", "TEVA", "VTRS", "JAZZ", 
           "AXSM", "ELAN", "CORT", "OGN", "BHC", "PRGO"]

start_date = "2016-01-01"
end_date = "2020-12-31"

all_data = []

for ticker in tickers:
    print(f"Fetching data for {ticker}...")
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date, interval="1mo")
    
    # Adding necessary columns
    df['Ticker'] = ticker
    
    # Standard yfinance doesn't provide VWAP/Transactions by default on monthly intervals
    # We calculate a proxy for VWAP: (Typical Price * Volume) / Total Volume
    df['VWAP_Proxy'] = ((df['High'] + df['Low'] + df['Close']) / 3 * df['Volume']).cumsum() / df['Volume'].cumsum()

    info = stock.info
    trailing_pe = info.get("trailingPE")
    forward_pe = info.get("forwardPE")
    df['trailing_pe'] = trailing_pe
    df['forwardPE'] = forward_pe


    try:
        financials = stock.quarterly_financials
        if 'Diluted EPS' in financials.index:
            eps_series = financials.loc['Diluted EPS']
        elif 'Basic EPS' in financials.index:
            eps_series = financials.loc['Basic EPS']
        else:
            eps_series = pd.Series()
            
        # Convert EPS index to datetime and sort
        eps_df = eps_series.to_frame(name='EPS')
        eps_df.index = pd.to_datetime(eps_df.index)
        eps_df = eps_df.sort_index()

        # 3. Merge Price and EPS
        # We use 'merge_asof' to match each month's price to the most recent EPS report
        df = df.sort_index()
        df['Date_Col'] = df.index
        df_combined = pd.merge_asof(df, eps_df, left_on='Date_Col', right_index=True, direction='backward')

        # 4. Calculate the P/E Ratio
        df_combined['Historical_PE'] = df_combined['Close'] / df_combined['EPS']
        
        # Cleanup
        df_combined.drop(columns=['Date_Col'], inplace=True)
        all_data.append(df_combined)
        
    except Exception as e:
        print(f"Could not calculate PE for {ticker}: {e}")
        all_data.append(df)








    
    all_data.append(df)

# Combine and export
final_df = pd.concat(all_data)
final_df.to_csv("Historical_Stock_Data_2016_2020.csv")
print("done!")








for ticker in tickers:
    print(f"Fetching and calculating historical P/E for {ticker}...")
    stock = yf.Ticker(ticker)
    
    # 1. Get Price Data
    df['Ticker'] = ticker
    
    # 2. Get Historical EPS (Quarterly)
    # .quarterly_financials returns a dataframe where rows are items like 'Diluted EPS'
    try:
        financials = stock.quarterly_financials
        if 'Diluted EPS' in financials.index:
            eps_series = financials.loc['Diluted EPS']
        elif 'Basic EPS' in financials.index:
            eps_series = financials.loc['Basic EPS']
        else:
            eps_series = pd.Series()
            
        # Convert EPS index to datetime and sort
        eps_df = eps_series.to_frame(name='EPS')
        eps_df.index = pd.to_datetime(eps_df.index)
        eps_df = eps_df.sort_index()

        df = df.sort_index()
        df['Date_Col'] = df.index
        df_combined = pd.merge_asof(df, eps_df, left_on='Date_Col', right_index=True, direction='backward')

        # 4. Calculate the P/E Ratio
        df_combined['Historical_PE'] = df_combined['Close'] / df_combined['EPS']
        
        # Cleanup
        df_combined.drop(columns=['Date_Col'], inplace=True)
        all_data.append(df_combined)
        
    except Exception as e:
        print(f"Could not calculate PE for {ticker}: {e}")
        all_data.append(df)

final_df = pd.concat(all_data)
print(final_df[['Ticker', 'Close', 'EPS', 'Historical_PE']].head())


