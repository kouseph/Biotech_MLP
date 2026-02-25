import matplotlib.pyplot as plt
from datetime import datetime
import finnhub 

hello world, this is garbage

from dotenv import load_dotenv
import os

load_dotenv() 

finnhub_api_key = os.getenv("FINNHUB_API_KEY")


def plot_finnhub_library_data(symbol):
    # Initialize the Finnhub client
    finnhub_client = finnhub.Client(api_key=finnhub_api_key)
    
    financials = finnhub_client.company_basic_financials(symbol, 'all')
    
    series_data = financials.get('series', {}).get('annual', {}).get('currentRatio', [])
    
    if not series_data:
        print(f"No Current Ratio data available for {symbol}.")
        return[0,0,0]

    series_data.sort(key=lambda x: datetime.strptime(x['period'], '%Y-%m-%d'))

    dates = [item['period'] for item in series_data]
    values = [item['v'] for item in series_data]

    # Create the Plot
    return [dates, values, symbol]


VanEck = ["LLY", "NVS", "MRK", "NVO", "GSK", "BMY", "JNJ", "MCK", "PFE", "AZN", "ABBV", "COR", "SNY", "ZTS", "HLN", "TAK", "TEVA", "VTRS", "JAZZ", "AXSM", "ELAN", "CORT", "OGN", "BHC", "PRGO"]
tickers = [ "AAPL", "NVDA", "PFE", "ISRG", "JPM", "GS", "XOM", "SLB", "CAT", "LMT", "TSLA", "SBUX", "KO", "COST", "NEE", "DUK", "LIN", "NEM", "AMT", "PLD", "VZ", "TMUS", "FDX", "BA", "DIS" ]

ans = []
for i in tickers:
    ans.append(plot_finnhub_library_data(i))

plt.figure(figsize=(10, 5))
for i in ans:
    plt.plot(i[0], i[1], marker='s', label=i[2])
plt.xlabel('Fiscal Period')
plt.ylabel('Value (v)')
plt.grid(axis='y', linestyle='--', alpha=0.6)
plt.legend()


plt.show()
