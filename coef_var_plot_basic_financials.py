import finnhub
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import os
import numpy as np

load_dotenv() 

finnhub_api_key = os.getenv("FINNHUB_API_KEY")


tickers = ["LLY", "NVS", "MRK", "NVO", "GSK", "BMY", "JNJ", "MCK", "PFE", "AZN", "ABBV", "COR", "SNY", "ZTS", "HLN", "TAK", "TEVA", "VTRS", "JAZZ", "AXSM", "ELAN", "CORT", "OGN", "BHC", "PRGO"]
# tickers = [ "AAPL", "NVDA", "PFE", "ISRG", "JPM", "GS", "XOM", "SLB", "CAT", "LMT", "TSLA", "SBUX", "KO", "COST", "NEE", "DUK", "LIN", "NEM", "AMT", "PLD", "VZ", "TMUS", "FDX", "BA", "DIS" ]


finnhub_client = finnhub.Client(api_key=finnhub_api_key)
all_data = {}

# 2. Fetch and Process
for t in tickers:
    res = finnhub_client.company_basic_financials(t, 'all')
    ratios = res.get('series', {}).get('annual', {}).get('currentRatio', [])
    if ratios:
        # Create a series of values for each stock
        all_data[t] = [item['v'] for item in ratios]

df = pd.DataFrame(dict([(k, pd.Series(v)) for k, v in all_data.items()]))

stats = pd.DataFrame({
    'Mean': df.mean(),
    'Std Dev': df.std(),
    'CV': df.std() / df.mean()
}).sort_values('CV')

print("--- Biotech Liquidity Stability (Lowest CV is most stable) ---")
print(stats)

# 4. Correlation Heatmap (To find the 'Patterns')
plt.figure(figsize=(12, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='RdYlGn', center=0)
plt.title('Correlation of Current Ratios Across Biotech Basket')
plt.show()


corr_matrix = df.corr()

# Get the average of the upper triangle of the matrix (to avoid self-correlation)
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
average_correlation = corr_matrix.where(mask).stack().mean()

print(f"Basket Similarity Score: {average_correlation:.4f}")

if average_correlation > 0.7:
    print("Verdict: These stocks are moving in a tight pack (High Certainty).")
elif average_correlation > 0.4:
    print("Verdict: These stocks have moderate similarity.")
else:
    print("Verdict: These stocks are independent/diversified.")

