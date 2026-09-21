import os
import requests
import pandas as pd
import time

API_KEY = os.environ["OPENAQ_API_KEY"]
SENSOR_ID = 23534

url = f"https://api.openaq.org/v3/sensors/{SENSOR_ID}/days"

headers = {"X-API-Key": API_KEY}

all_data = []
page = 1

while True:
    params = {
        "limit": 1000,
        "page": page
    }

    response = requests.get(url, headers=headers, params=params)
    response.raise_for_status()

    results = response.json()["results"]

    if not results:
        break

    for x in results:
        all_data.append({
            "date": x["period"]["datetimeFrom"]["local"],
            "pm25": x["value"]
        })

    print(f"Page {page}: {len(results)} records")
    page += 1
    time.sleep(0.2)

df = pd.DataFrame(all_data)

df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date")
df = df.drop_duplicates("date")

df.to_csv("data/delhi_pm25_daily.csv", index=False)

print("\nDONE")
print("Shape:", df.shape)
print("Date range:", df["date"].min(), "to", df["date"].max())
print(df.head())
print(df.tail())