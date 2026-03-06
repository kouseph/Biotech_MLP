import yfinance as yf
import pandas as pd
import numpy as np

# Define your tickers and parameters
tickers = ["LLY", "NVS", "MRK", "NVO", "GSK", "BMY", "JNJ", "MCK", "PFE", "AZN", 
           "ABBV", "COR", "SNY", "ZTS", "HLN", "TAK", "TEVA", "VTRS", "JAZZ", 
           "AXSM", "ELAN", "CORT", "OGN", "BHC", "PRGO"]

start_date = "2016-01-01"
end_date = "2020-12-31"

all_data = []

for ticker in tickers:
    print(f"Fetching data and calculating PE for {ticker}...")
    try:
        stock = yf.Ticker(ticker)
        
        # 1. Fetch Monthly Price History
        df = stock.history(start=start_date, end=end_date, interval="1mo")
        if df.empty:
            print(f"No data found for {ticker} in this range.")
            continue
            
        # FIX: Remove timezone from the index to allow merging with EPS data
        df.index = df.index.tz_localize(None)
        df['Ticker'] = ticker
        df['Timestamp'] = df.index
        
        # 2. Calculate VWAP Proxy
        df['VWAP_Proxy'] = ((df['High'] + df['Low'] + df['Close']) / 3 * df['Volume']).cumsum() / df['Volume'].cumsum()

        # 3. Fetch Historical EPS (Quarterly)
        financials = stock.quarterly_financials
        eps_df = pd.DataFrame()
        
        if not financials.empty:
            if 'Diluted EPS' in financials.index:
                eps_series = financials.loc['Diluted EPS']
            elif 'Basic EPS' in financials.index:
                eps_series = financials.loc['Basic EPS']
            else:
                eps_series = pd.Series(dtype='float64')

            if not eps_series.empty:
                eps_df = eps_series.to_frame(name='EPS')
                eps_df.index = pd.to_datetime(eps_df.index).tz_localize(None)
                eps_df = eps_df.sort_index()

        # 4. Merge Price and EPS
        if not eps_df.empty:
            df = df.sort_index()
            # merge_asof matches the month to the most recent earnings report (backward)
            df_combined = pd.merge_asof(df, eps_df, left_index=True, right_index=True, direction='backward')
            # Calculate the P/E Ratio
            df_combined['Historical_PE'] = df_combined['Close'] / df_combined['EPS']
            all_data.append(df_combined)
        else:
            # If no EPS data is found, still keep the price data
            df['EPS'] = np.nan
            df['Historical_PE'] = np.nan
            all_data.append(df)
            
    except Exception as e:
        print(f"Error processing {ticker}: {e}")

# Combine all results
if all_data:
    final_df = pd.concat(all_data)
    
    # Fill in missing columns for your requested output
    final_df['transactions'] = np.nan # Not available for historical monthly yfinance
    final_df['otc'] = False          # Most of these are major exchange stocks
    
    # Save to CSV
    final_df.to_csv("Historical_Stock_Data_2016_2020_Final.csv", index=False)
    print("CSV Generated successfully!")
else:
    print("No data was collected. Please check your internet connection or ticker symbols.")
