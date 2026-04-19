import requests
import pandas as pd
from dateutil.relativedelta import relativedelta
from datetime import datetime


BASE = "https://dashboard.elering.ee/api"

def get_nps_prices(start: str, end: str) -> pd.DataFrame:

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


def get_cross_border_flows(start: str, end: str) -> pd.DataFrame:

    r = requests.get(BASE + "/transmission/cross-border/hourly", params={"start": start, "end": end})
    r.raise_for_status()

    data = r.json()["data"]

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.set_index("timestamp").sort_index()

    # Keep only relevant cross-border flows (drop cable-level detail and Russia)
    df = df[["finland", "latvia"]]
    df.columns = [("ee", "fi"), ("ee", "lv")]

    return df

def get_system_production(start: str, end: str) -> pd.DataFrame:
    
    r = requests.get(BASE + "/system", params={"start": start, "end": end})
    r.raise_for_status()

    data = r.json()["data"]

    df = pd.DataFrame(data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
    df = df.set_index("timestamp").sort_index()

    return df

def get_balance_total(start: str, end: str) -> pd.DataFrame:
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
        print(f"  Fetching {chunk_start[:10]} → {chunk_end[:10]}...")
        df = fetch_fn(chunk_start, chunk_end)
        chunks.append(df)
    combined = pd.concat(chunks).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    
