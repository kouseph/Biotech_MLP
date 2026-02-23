from dotenv import load_dotenv
import os
import finnhub

load_dotenv() 

finnhub_api_key = os.getenv("FINNHUB_API_KEY")

finnhub_client = finnhub.Client(api_key=finnhub_api_key)

print(finnhub_client.company_basic_financials('JNJ', 'all'))



