import requests

url = "https://data.alpaca.markets/v1beta1/news?start=2016-01-04&end=2024-01-04&sort=desc"

headers = {
    "accept": "application/json",
    "APCA-API-KEY-ID": "PKUQOEWGXW635SMBNRYCPNAOGT",
    "APCA-API-SECRET-KEY": "7BnJtRDTatAQHXoWvoj5Amj6rFq9KeRRDSZkBtdxwVkq"
}

response = requests.get(url, headers=headers)

print(response.text)
