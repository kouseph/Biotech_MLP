import finnhub
import pandas as pd
import matplotlib.pyplot as plt

from dotenv import load_dotenv
import os

load_dotenv() 

finnhub_api_key = os.getenv("FINNHUB_API_KEY")


VanEck = ["LLY", "NVS", "MRK", "NVO", "GSK", "BMY", "JNJ", "MCK", "PFE", "AZN", "ABBV", "COR", "SNY", "ZTS", "HLN", "TAK", "TEVA", "VTRS", "JAZZ", "AXSM", "ELAN", "CORT", "OGN", "BHC", "PRGO"]
tickers = [ "AAPL", "NVDA", "PFE", "ISRG", "JPM", "GS", "XOM", "SLB", "CAT", "LMT", "TSLA", "SBUX", "KO", "COST", "NEE", "DUK", "LIN", "NEM", "AMT", "PLD", "VZ", "TMUS", "FDX", "BA", "DIS" ]


all_trends = []

# 2. Fetch and Extract Data
for symbol in VanEck:
    # returns a list of dictionaries (monthly trends)
    finnhub_client = finnhub.Client(api_key=finnhub_api_key)
    res = finnhub_client.recommendation_trends(symbol)
    
    if res:
        # We take the most recent month (index 0)
        latest = res[0]
        all_trends.append(latest)

# 3. Create DataFrame
df = pd.DataFrame(all_trends)

# 4. Calculate Percentages (Normalization)
# This makes it easier to compare "Buy Sentiment" regardless of analyst count
cols = ['strongBuy', 'buy', 'hold', 'sell', 'strongSell']
df['total'] = df[cols].sum(axis=1)

for col in cols:
    df[f'{col}_%'] = (df[col] / df['total']) * 100

# 5. Sorting for Clarity
# Sort by highest "Strong Buy" + "Buy" percentage
df['bullish_score'] = df['strongBuy_%'] + df['buy_%']
df = df.sort_values('bullish_score', ascending=False)

# 6. Visualization: Stacked Bar Chart
df.set_index('symbol')[[f'{c}_%' for c in cols]].plot(
    kind='barh', 
    stacked=True, 
    figsize=(12, 10),
    color=['darkgreen', 'limegreen', 'gold', 'orange', 'red']
)

plt.title('Analyst Recommendation Distribution (Normalized %)')
plt.xlabel('Percentage of Analysts')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
