import yfinance as yf
import matplotlib.pyplot as plt

tickers = ["LLY", "NVS", "MRK", "NVO", "GSK", "BMY", "JNJ", "MCK", "PFE", "AZN", 
           "ABBV", "COR", "SNY", "ZTS", "HLN", "TAK", "TEVA", "VTRS", "JAZZ", 
           "AXSM", "ELAN", "CORT", "OGN", "BHC", "PRGO"]

for ticker in tickers:
    print(f"Generating individual plot for {ticker}...")
    stock = yf.Ticker(ticker)
    df = stock.history(start="2016-01-01", end="2020-12-31", interval="1mo")
    
    if df.empty:
        print(f"Skipping {ticker}: No data found for this period.")
        continue

    # Create a figure with two subplots (Price on top, Volume on bottom)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [3, 1]})
    
    # 1. Plot the Price
    ax1.plot(df.index, df['Close'], color='navy', label='Monthly Close', linewidth=2)
    ax1.fill_between(df.index, df['Close'], color='skyblue', alpha=0.3)
    ax1.set_title(f"{ticker} Historical Performance (2016 - 2020)", fontsize=16, fontweight='bold')
    ax1.set_ylabel("Price (USD)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # 2. Plot the Volume
    ax2.bar(df.index, df['Volume'], color='gray', alpha=0.7, width=20) # width adjusted for monthly bars
    ax2.set_ylabel("Volume")
    ax2.set_xlabel("Date")

    # Final formatting
    plt.tight_layout()
    
    # This will open a new window for every ticker. 
    # Warning: Running all 25 at once might be overwhelming!
    plt.show()








# import yfinance as yf
# import pandas as pd
#
# tickers = ["LLY", "NVS", "MRK", "NVO", "GSK", "BMY", "JNJ", "MCK", "PFE", "AZN", 
#            "ABBV", "COR", "SNY", "ZTS", "HLN", "TAK", "TEVA", "VTRS", "JAZZ", 
#            "AXSM", "ELAN", "CORT", "OGN", "BHC", "PRGO"]
#
# start_date = "2016-01-01"
# end_date = "2020-12-31"
#
# all_data = []
#
# for ticker in tickers:
#     print(f"Fetching data for {ticker}...")
#     stock = yf.Ticker(ticker)
#     df = stock.history(start=start_date, end=end_date, interval="1mo")
#     
#     df['Ticker'] = ticker
#     df['Timestamp'] = df.index
#     
#     # Standard yfinance doesn't provide VWAP/Transactions by default on monthly intervals
#     # We calculate a proxy for VWAP: (Typical Price * Volume) / Total Volume
#     df['VWAP_Proxy'] = ((df['High'] + df['Low'] + df['Close']) / 3 * df['Volume']).cumsum() / df['Volume'].cumsum()
#     
#     all_data.append(df)
#
# # Combine and export
# final_df = pd.concat(all_data)
# print(final_df)
# # final_df.to_csv("Historical_Stock_Data_2016_2020.csv")
# # print("CSV Generated successfully!")
