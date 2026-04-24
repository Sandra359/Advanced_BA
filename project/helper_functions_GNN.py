import requests
import pandas as pd
from dateutil.relativedelta import relativedelta
from datetime import datetime
import torch
import time


BASE = "https://dashboard.elering.ee/api"

def get_nps_prices(start: str, end: str, BASE=BASE) -> pd.DataFrame:

    r = requests.get(BASE + "/nps/price", params={"start": start, "end": end})
    r.raise_for_status()

    data = r.json()["data"]  # {"ee": [...], "lv": [...], "lt": [...], "fi": [...]}

    dfs = []
    for country, records in data.items():
        df = pd.DataFrame(records)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df = df.set_index("timestamp")
        df = df.rename(columns={"price": country})
        dfs.append(df)

    df_prices = pd.concat(dfs, axis=1).sort_index()
    return df_prices


def get_cross_border_flows(start: str, end: str, BASE=BASE) -> pd.DataFrame:

    r = requests.get(BASE + "/transmission/cross-border/hourly", params={"start": start, "end": end})
    r.raise_for_status()

    data = r.json()["data"]

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.set_index("timestamp").sort_index()

    # Keep only relevant cross-border flows
    available_cols = df.columns.tolist()
    col_mapping = {
        "finland": ("ee", "fi"),
        "latvia": ("ee", "lv"),
        "russia_narva": ("ee", "ru_narva"),
        "russia_pihkva": ("ee", "ru_pihkva"),
    }
    selected_cols = [c for c in col_mapping.keys() if c in available_cols]
    df = df[selected_cols]
    df.columns = [col_mapping[c] for c in selected_cols] 

    return df

def get_system_production(start: str, end: str, BASE=BASE) -> pd.DataFrame:
    
    r = requests.get(BASE + "/system", params={"start": start, "end": end})
    r.raise_for_status()

    data = r.json()["data"]

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.set_index("timestamp").sort_index()

    return df

def get_balance_total(start: str, end: str, BASE=BASE) -> pd.DataFrame:
    """Monthly energy balance — too coarse for hourly model, kept for reference only."""
    r = requests.get(BASE + "/balance/total", params={"start": start, "end": end})
    r.raise_for_status()
    df = pd.DataFrame(r.json()["data"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    return df.set_index("timestamp").sort_index()





# --- 1. helper functions ---

def generate_monthly_ranges(start_iso: str, end_iso: str) -> list:
    '''Creates chunks of 1 month to avoid errors from too large requests. Officially the API should support up to 1 year.'''
    start = datetime.fromisoformat(start_iso.replace("Z", "+00:00"))
    end   = datetime.fromisoformat(end_iso.replace("Z", "+00:00"))
    ranges = []
    current = start
    while current < end:
        next_month = min(current + relativedelta(months=1), end)
        ranges.append((
            current.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
            next_month.strftime("%Y-%m-%dT%H:%M:%S.000Z")
        ))
        current = next_month
    return ranges

def fetch_all(fetch_fn, start: str, end: str) -> pd.DataFrame:
    '''Fetches data using the provided fetch functions in monthly chunks and combiines it into one dataframe.'''
    chunks = []
    for chunk_start, chunk_end in generate_monthly_ranges(start, end):
        #print(f"  Fetching {chunk_start[:10]} → {chunk_end[:10]}...")
        df = fetch_fn(chunk_start, chunk_end)
        chunks.append(df)
    chunks = [c for c in chunks if not c.empty]
    combined = pd.concat(chunks).sort_index()
    return combined

def quantile_loss(preds, targets, quantiles=[0.1, 0.5, 0.9]):
    targets = targets.squeeze()
    losses  = []
    for i, q in enumerate(quantiles):
        e = targets - preds[:, i]
        losses.append(torch.max(q * e, (q - 1) * e))
    return torch.stack(losses, dim=1).mean()
    


def get_weather_openmeteo(lat: float, lon: float,
                           start: str, end: str) -> pd.DataFrame:
    """
    Free historical weather — no API key, no subscription needed.
    Uses ERA5 reanalysis — covers 1979 to present at hourly resolution.
    """
    r = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude":        lat,
            "longitude":       lon,
            "start_date":      start[:10],  # "2019-01-01"
            "end_date":        end[:10],    # "2026-02-01"
            "hourly":          "temperature_2m,wind_speed_10m,wind_speed_100m",
            "wind_speed_unit": "ms",
            "timezone":        "UTC",
        },
        timeout=60
    )

    if r.status_code != 200:
        print(f"  ERROR {r.status_code}: {r.text[:200]}")
        return pd.DataFrame()

    data = r.json()
    df = pd.DataFrame({
        "timestamp":       pd.to_datetime(data["hourly"]["time"], utc=True),
        "temperature":     data["hourly"]["temperature_2m"],
        "wind_speed_10m":  data["hourly"]["wind_speed_10m"],
        "wind_speed_100m": data["hourly"]["wind_speed_100m"],
    })

    df = df.set_index("timestamp").sort_index()
    print(f"  Fetched {len(df)} hours: {df.index.min()} → {df.index.max()}")
    return df

