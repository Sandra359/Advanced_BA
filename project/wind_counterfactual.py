"""
Counterfactual Wind Production Scenarios — January 2026
=========================================================
Estimates hourly wind production for January 2026 using ERA5 reanalysis
weather data from Open-Meteo and a Vestas V150-4.5 MW power curve model.

Output: data/wind_production_scenarios.csv  (744 rows × 3 columns)
        wind_mwh_baseline  — existing 694 MW fleet
        wind_mwh_scenA     — baseline + 3 established-permit farms (+323 MW)
        wind_mwh_scenB     — Scenario A + 5 pipeline farms (+564 MW)
"""

import os
import numpy as np
import pandas as pd
import requests

_DIR  = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_DIR, "..", "data")
OUT_CSV = os.path.join(_DATA, "wind_production_scenarios.csv")

JAN2026_START = "2026-01-01"
JAN2026_END   = "2026-01-31"


# ── Power curve: Vestas V150-4.5 MW ──────────────────────────────────────────
# Cut-in ~3.5 m/s | rated ~13 m/s | cut-out 25 m/s
_PC_WS = np.array([0,1,2,3,3.5,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,25.1,50])
_PC_KW = np.array([0,0,0,0,  0,66,220,480,820,1250,1850,2550,3200,3750,4150,4350,4450,4480,4490,4495,4498,4500,4500,4500,4500,4500,4500,0,0])


def _air_density(temp_c, pressure_hpa):
    return (pressure_hpa * 100) / (287.05 * (temp_c + 273.15))


def turbine_cf(wind_speed_100m, temp_c, pressure_hpa):
    """Capacity factor in [0, 1] — 100 m wind speed used as V150 hub-height proxy (hub = 105 m, 0.7% difference)."""
    rho    = _air_density(temp_c, pressure_hpa)
    p_kw   = np.interp(wind_speed_100m, _PC_WS, _PC_KW) * (rho / 1.225)
    return np.clip(p_kw / 4500.0, 0, 1)


# ── ERA5 weather fetcher ──────────────────────────────────────────────────────
def get_weather_openmeteo(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """Fetch hourly ERA5 reanalysis from Open-Meteo archive (free, no API key)."""
    r = requests.get(
        "https://archive-api.open-meteo.com/v1/archive",
        params={
            "latitude":        lat,
            "longitude":       lon,
            "start_date":      start[:10],
            "end_date":        end[:10],
            "hourly":          "temperature_2m,wind_speed_10m,wind_speed_100m,surface_pressure",
            "wind_speed_unit": "ms",
            "timezone":        "UTC",
        },
        timeout=60,
    )
    if r.status_code != 200:
        raise RuntimeError(f"Open-Meteo error {r.status_code}: {r.text[:200]}")

    data = r.json()
    df = pd.DataFrame({
        "timestamp":       pd.to_datetime(data["hourly"]["time"], utc=True),
        "temperature":     data["hourly"]["temperature_2m"],
        "wind_speed_10m":  data["hourly"]["wind_speed_10m"],
        "wind_speed_100m": data["hourly"]["wind_speed_100m"],
        "pressure":        data["hourly"]["surface_pressure"],
    }).set_index("timestamp").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    print(f"  Fetched {len(df)} hours  |  wind_100m mean: {df['wind_speed_100m'].mean():.1f} m/s")
    return df


# ── Farm definitions ──────────────────────────────────────────────────────────
# Scenario A — valid special/detailed plans (+323 MW)
# Coordinates: geographic centroids from official Estonian Land Board GIS layers
#   tuul_ep_kehtestatud.geojson (Lääneranna, Pärnu+Tori)
#   tuulealad_DP.geojson (Aidu)
NEW_FARMS_A = [
    {"name": "Lääneranna (area 2)",       "lat": 58.4473, "lon": 23.7673, "capacity_mw": 137},
    {"name": "Pärnu+Tori (Põlendmaa)",    "lat": 58.3775, "lon": 24.7786, "capacity_mw":  86},
    {"name": "Aidu renewable energy park","lat": 59.3195, "lon": 27.0613, "capacity_mw": 100},
]

# Scenario B extra — plans moving towards implementation: SEA published (+564 MW)
# Source: tuuleenergeetika_KOV_EP_alad.geojson
NEW_FARMS_B_EXTRA = [
    {"name": "Lääneranna (pipeline)",   "lat": 58.5658, "lon": 23.7781, "capacity_mw": 150},
    {"name": "Tori rural municipality", "lat": 58.5496, "lon": 24.7787, "capacity_mw": 100},
    {"name": "Lääne-Nigula",            "lat": 59.0159, "lon": 23.9870, "capacity_mw": 100},
    {"name": "Põhja-Pärnumaa",          "lat": 58.6254, "lon": 24.8906, "capacity_mw": 100},
    {"name": "Lüganuse (Evecon+Enery)", "lat": 59.3959, "lon": 27.0476, "capacity_mw": 114},
]

BASELINE_LAT, BASELINE_LON = 58.90, 24.75   # central Estonia centroid for existing 694 MW fleet
BASELINE_MW = 694


# ── Fetch weather ─────────────────────────────────────────────────────────────
def fetch_weather_for_farms(farms, label):
    results = {}
    for farm in farms:
        print(f"\n  [{farm['name']}] {farm['capacity_mw']} MW  lat={farm['lat']}, lon={farm['lon']}")
        wx = get_weather_openmeteo(farm["lat"], farm["lon"], JAN2026_START, JAN2026_END)
        results[farm["name"]] = (farm, wx)
    print(f"\n  Done — {label}: {len(results)} locations")
    return results


# ── MW production per scenario ────────────────────────────────────────────────
def farms_to_mw(weather_dict):
    total = None
    for name, (farm, wx) in weather_dict.items():
        cf      = turbine_cf(wx["wind_speed_100m"].values, wx["temperature"].values, wx["pressure"].values)
        farm_mw = pd.Series(cf * farm["capacity_mw"], index=wx.index, name=name)
        total   = farm_mw if total is None else total.add(farm_mw, fill_value=0)
    return total


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Fetching Scenario A farm weather ===")
    weather_A = fetch_weather_for_farms(NEW_FARMS_A, "Scenario A")

    print("\n=== Fetching Scenario B extra farm weather ===")
    weather_B_extra = fetch_weather_for_farms(NEW_FARMS_B_EXTRA, "Scenario B extra")

    print("\n=== Fetching baseline (existing fleet centroid) ===")
    baseline_wx = get_weather_openmeteo(BASELINE_LAT, BASELINE_LON, JAN2026_START, JAN2026_END)

    # Compute production series
    new_A_mw       = farms_to_mw(weather_A)
    new_B_extra_mw = farms_to_mw(weather_B_extra)

    baseline_cf = turbine_cf(
        baseline_wx["wind_speed_100m"].values,
        baseline_wx["temperature"].values,
        baseline_wx["pressure"].values,
    )
    baseline_mw = pd.Series(baseline_cf * BASELINE_MW, index=baseline_wx.index)

    idx = baseline_mw.index
    scenarios = pd.DataFrame({
        "wind_mwh_baseline": baseline_mw,
        "wind_mwh_scenA":    baseline_mw + new_A_mw.reindex(idx).fillna(0),
        "wind_mwh_scenB":    baseline_mw + new_A_mw.reindex(idx).fillna(0) + new_B_extra_mw.reindex(idx).fillna(0),
    }, index=idx)

    # Validate
    assert len(scenarios) == 744,                                         f"Expected 744 hours, got {len(scenarios)}"
    assert scenarios.isna().sum().sum() == 0,                             "NaN values found"
    assert (scenarios["wind_mwh_scenA"] >= scenarios["wind_mwh_baseline"]).all(), "Scenario A below baseline"
    assert (scenarios["wind_mwh_scenB"] >= scenarios["wind_mwh_scenA"]).all(),    "Scenario B below Scenario A"
    print("\nAll validation checks passed.")

    scenarios.to_csv(OUT_CSV)
    print(f"Saved → {OUT_CSV}")
    print(f"Shape: {scenarios.shape}")
    print(f"\nMean production:")
    print(f"  Baseline (694 MW):  {scenarios['wind_mwh_baseline'].mean():.1f} MW  (CF {baseline_cf.mean():.3f})")
    print(f"  Scenario A (+323):  {scenarios['wind_mwh_scenA'].mean():.1f} MW")
    print(f"  Scenario B (+887):  {scenarios['wind_mwh_scenB'].mean():.1f} MW")
