import requests
import pandas as pd
from dateutil.relativedelta import relativedelta
from datetime import datetime
import torch


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
    
import datetime

OPENWEATHER_KEY = "822ed994e581c3d5ec9ce7d445f486e3"
HIST_URL = "http://history.openweathermap.org/data/2.5/history/city"

def get_weather_history(lat: float, lon: float, 
                         start: str, end: str,
                         api_key: str = OPENWEATHER_KEY) -> pd.DataFrame:
    """
    Fetch hourly weather history from OpenWeatherMap for a single coordinate.
    
    Parameters:
        lat, lon:  Coordinates of the location
        start, end: ISO format strings e.g. "2019-01-01T00:00:00.000Z"
        api_key:   OpenWeatherMap API key
    
    Returns:
        DataFrame with hourly temperature and wind speed, UTC indexed
    """
    # Convert ISO strings to unix timestamps
    start_ts = int(datetime.datetime.fromisoformat(
        start.replace("Z", "+00:00")).timestamp())
    end_ts   = int(datetime.datetime.fromisoformat(
        end.replace("Z", "+00:00")).timestamp())
    
    CHUNK_SEC = 168 * 3600  # 168 hours = max per API call
    
    all_records = []
    current_ts  = start_ts
    
    while current_ts < end_ts:
        chunk_end = min(current_ts + CHUNK_SEC, end_ts)
        
        params = {
            "lat":   lat,
            "lon":   lon,
            "type":  "hour",
            "start": current_ts,
            "end":   chunk_end,
            "appid": api_key,
            "units": "metric",
        }
        
        try:
            r = requests.get(HIST_URL, params=params, timeout=30)
            if r.status_code != 200:
                print(f"  WARNING {r.status_code} for chunk "
                      f"{datetime.datetime.fromtimestamp(current_ts, tz=datetime.timezone.utc).date()}: "
                      f"{r.text[:100]}")
                current_ts = chunk_end
                time.sleep(1)
                continue
            
            records = r.json().get("list", [])
            all_records.extend(records)
            print(f"  Fetched {len(records)} hours from "
                  f"{datetime.datetime.fromtimestamp(current_ts, tz=datetime.timezone.utc).date()}: "
                  f"→ {datetime.datetime.fromtimestamp(chunk_end,   tz=datetime.timezone.utc).date()}")
        
        except requests.exceptions.Timeout:
            print(f"  Timeout, retrying...")
            time.sleep(5)
            continue
        
        current_ts = chunk_end
        time.sleep(0.5)  # be polite to the API
    
    if not all_records:
        print("  No records returned — check API key and subscription")
        return pd.DataFrame()
    
    # Parse into clean DataFrame
    rows = []
    for rec in all_records:
        rows.append({
            "timestamp":      pd.Timestamp(rec["dt"], unit="s", tz="UTC"),
            "temperature":    rec.get("main", {}).get("temp"),
            "wind_speed_10m": rec.get("wind", {}).get("speed"),
            "wind_gust":      rec.get("wind", {}).get("gust"),
            "wind_dir":       rec.get("wind", {}).get("deg"),
            "pressure":       rec.get("main", {}).get("pressure"),
        })
    
    df = pd.DataFrame(rows).set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df


def get_weather_for_locations(locations_df: pd.DataFrame,
                               start: str, end: str,
                               lat_col: str = "centroid_lat",
                               lon_col: str = "centroid_lon",
                               name_col: str = "area_name") -> pd.DataFrame:
    """
    Fetch weather for multiple wind farm locations and average them.
    
    Parameters:
        locations_df: DataFrame with lat/lon columns for each wind farm site
        start, end:   ISO date strings
        
    Returns:
        Single DataFrame with averaged weather across all locations, hourly UTC indexed
    """
    all_dfs = []
    
    for _, row in locations_df.iterrows():
        print(f"\nFetching weather for: {row[name_col]} "
              f"({row[lat_col]:.2f}°N, {row[lon_col]:.2f}°E)")
        
        df = get_weather_history(
            lat=row[lat_col],
            lon=row[lon_col],
            start=start,
            end=end
        )
        
        if not df.empty:
            all_dfs.append(df)
    
    if not all_dfs:
        print("No weather data fetched")
        return pd.DataFrame()
    
    # Average across all locations
    combined = pd.concat(all_dfs).groupby(level=0).mean()
    print(f"\nCombined weather shape: {combined.shape}")
    print(f"Date range: {combined.index.min()} → {combined.index.max()}")
    return combined