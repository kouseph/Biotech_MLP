import requests
from dotenv import load_dotenv
import os

load_dotenv() 

guardian_api_key = os.getenv("GUARDIAN_API_KEY")

def get_historical_pharma_news(query, start_date, end_date):
    url = f"https://content.guardianapis.com/search?q={query}&from-date={start_date}&to-date={end_date}&section=business|science&api-key={guardian_api_key}"
    
    response = requests.get(url).json()
    # This gives you headlines and "trailText" (summaries) from 2020!
    print(url)
    return response['response']['results']

# Example: Get news from the height of the first wave
news_april_2020 = get_historical_pharma_news("", "2020-04-01", "2022-04-30")

print(news_april_2020)
