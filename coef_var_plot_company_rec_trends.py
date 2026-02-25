import finnhub
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import time

from dotenv import load_dotenv
import os
import numpy as np

load_dotenv() 

finnhub_api_key = os.getenv("FINNHUB_API_KEY")



finnhub_client = finnhub.Client(api_key=finnhub_api_key)
# tickers = [ "AAPL", "NVDA", "PFE", "ISRG", "JPM", "GS", "XOM", "SLB", "CAT", "LMT", "TSLA", "SBUX", "KO", "COST", "NEE", "DUK", "LIN", "NEM", "AMT", "PLD", "VZ", "TMUS", "FDX", "BA", "DIS" ]
# tickers = ["LLY", "NVS", "MRK", "NVO", "GSK", "BMY", "JNJ", "MCK", "PFE", "AZN", "ABBV", "COR", "SNY", "ZTS", "HLN", "TAK", "TEVA", "VTRS", "JAZZ", "AXSM", "ELAN", "CORT", "OGN", "BHC", "PRGO"]


tickers = ["LLY", "NVS", "MRK", "NVO", "GSK", "BMY", "JNJ", "MCK", "PFE", "AZN", "ABBV", "COR", "SNY", "ZTS", "HLN", "TAK", "TEVA", "VTRS", "JAZZ", "AXSM", "ELAN", "CORT", "OGN", "BHC", "PRGO", "AAPL", "NVDA", "PFE", "ISRG", "JPM", "GS", "XOM", "SLB", "CAT", "LMT", "TSLA", "SBUX", "KO", "COST", "NEE", "DUK", "LIN", "NEM", "AMT", "PLD", "VZ", "TMUS", "FDX", "BA", "DIS"]
data_list = []

for symbol in tickers:
    try:
        res = finnhub_client.recommendation_trends(symbol)
        if res:
            latest = res[0]
            # Store only the numeric sentiment counts
            data_list.append({
                'symbol': symbol,
                'Strong Buy': latest['strongBuy'],
                'Buy': latest['buy'],
                'Hold': latest['hold'],
                'Sell': latest['sell'],
                'Strong Sell': latest['strongSell']
            })
        time.sleep(0.1) # Be kind to the API rate limit
    except Exception as e:
        print(f"Error fetching {symbol}: {e}")

# 1. Create DataFrame and transpose it
# We want Tickers as columns and Recommendation types as rows
df = pd.DataFrame(data_list).set_index('symbol').T

# 2. Calculate the Correlation Matrix
# This compares how similar the distribution of 'Buy' vs 'Hold' is between stocks
corr_matrix = df.corr()

# 3. Plotting the Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', center=0, linewidths=0.5)

plt.title('Stock Sentiment Correlation Matrix\n(Based on Analyst Recommendation Distribution)', fontsize=15)



print("\n" + "="*30)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
average_correlation = corr_matrix.where(mask).stack().mean()
print(f"Group Average Correlation: {average_correlation:.2f}")

if average_correlation > 0.7:
    print("Verdict: These stocks are moving in a tight pack (High Certainty).")
elif average_correlation > 0.4:
    print("Verdict: These stocks have moderate similarity.")
else:
    print("Verdict: These stocks are independent/diversified.")
print("="*30 + "\n")

# 5. Visualizing the Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, cmap='RdYlGn', center=0, fmt=".2f")
plt.title(f'Analyst Sentiment Correlation (Avg: {average_correlation:.2f})')
plt.show()


plt.show()
