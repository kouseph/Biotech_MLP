from dotenv import load_dotenv
import os
import finnhub
import pandas as pd

load_dotenv() 

finnhub_api_key = os.getenv("FINNHUB_API_KEY")

finnhub_client = finnhub.Client(api_key=finnhub_api_key)

VanEck = ["LLY", "NVS", "MRK", "NVO", "GSK", "BMY", "JNJ", "MCK", "PFE", "AZN", "ABBV", "COR", "SNY", "ZTS", "HLN", "TAK", "TEVA", "VTRS", "JAZZ", "AXSM", "ELAN", "CORT", "OGN", "BHC", "PRGO"]

# df = finnhub_client.company_basic_financials('JNJ', 'all')
# print(pd.read_json('hey.json'))


data = finnhub_client.company_basic_financials('IHE', 'all')

data.get(series_data = data.get('series', {}).get('annual', {}).get('currentRatio', []))


# pd.read_json(finnhub_client.company_basic_financials('IHE', 'all'))


# for i in VanEck:
#     data.append(finnhub_client.company_basic_financials(i, 'metrics'))



"""
PPH	VanEck Pharmaceutical ETF
IHE	iShares U.S. Pharmaceuticals ETF
PJP	Invesco Pharmaceuticals ETF	
XPH	State Street SPDR S&P Pharmaceuticals ETF
KURE	KraneShares MSCI All China Health Care Index ETF	
FTXH	First Trust Nasdaq Pharmaceuticals ETF
PILL	Direxion Daily Pharmaceutical & Medical Bull 3X Shares
"""


"""
{
        'metric':
        {'10DayAverageTradingVolume': 0.07472, 
         '13WeekPriceReturnDaily': 11.2509,
         '26WeekPriceReturnDaily': 31.036,
         '3MonthADReturnStd': 16.401266,
         '3MonthAverageTradingVolume': 0.09736,
         '52WeekHigh': 91.49,
         '52WeekHighDate': '2026-02-13',
         '52WeekLow': 58.9737,
         '52WeekLowDate': '2025-04-09',
         '52WeekPriceReturnDaily': 27.3083,
         '5DayPriceReturnDaily': 0.6966,
         'beta': 0.5207113,
         'monthToDatePriceReturnDaily': 5.0646,
         'priceRelativeToS&P50013Week': 7.871,
         'priceRelativeToS&P50026Week': 24.9595,
         'priceRelativeToS&P5004Week': 6.4361,
         'priceRelativeToS&P50052Week': 13.5653,
         'priceRelativeToS&P500Ytd': 7.287,
         'yearToDatePriceReturnDaily': 7.3559},

        'metricType': 'all', 
        'series': {}, 
        'symbol': 'IHE'}


"""

