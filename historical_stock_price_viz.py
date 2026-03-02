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
    
    plt.savefig(f"./plots/{ticker}_stock_data.png")
