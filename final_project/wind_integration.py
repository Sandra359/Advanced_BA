"""
ENTSOE Wind Production Integration
====================================
Loads and processes ENTSOE Transparency Platform CSV exports for Estonian
electricity generation (2019–2025), producing a clean hourly UTC time series.

Output: data/entsoe_production_hourly.csv  (61 368 rows × 13 columns)
        Columns: biomass, fossil_*, hydro_*, other*, solar, waste, wind_onshore,
                 capacity_factor, installed_mw
"""

import glob
import os
import numpy as np
import pandas as pd

_DIR  = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_DIR, "..", "data")

DATA_DIR = os.path.join(_DATA, "historical_production_data")
OUT_CSV  = os.path.join(_DATA, "entsoe_production_hourly.csv")

# ── 1. Glob CSVs — exclude 2026 (held-out test period) ──────────────────────
files = sorted([
    f for f in glob.glob(os.path.join(DATA_DIR, "*.csv"))
    if not os.path.basename(f).startswith("2026")
])
print(f"Found {len(files)} ENTSOE CSV files (2019–2025 only):")
for f in files:
    print(" ", os.path.basename(f))


# ── 2. Load all CSVs with DST-safe timestamp parsing ────────────────────────
#
# ENTSOE encodes each hour as a "start - end" interval string.
# DST spring-forward: "(CET) - (CEST)" suffix appears; non-existent hour skipped.
# DST fall-back: ambiguous hour appears twice, once "(CEST)" once "(EET)".
#
# Fix: build an explicit boolean `ambiguous` array by reading the CET/EET suffix
# directly from the raw MTU string — rows with (CET) or (EET) → DST=False,
# everything else → DST=True.  This correctly routes the two fall-back rows.

frames = []
for f in files:
    df = pd.read_csv(f, quotechar='"')
    df.columns = ["mtu", "area", "ptype", "mw"]

    mtu_start = df["mtu"].str.split(" - ").str[0]

    is_standard  = mtu_start.str.contains(r"\(CET\)|\(EET\)", regex=True, na=False)
    ambiguous_arr = np.where(is_standard, False, True)

    ts_clean = (
        mtu_start
        .str.replace(r"\s*\(\w+\)", "", regex=True)
        .str.strip()
    )
    df["ts"] = pd.to_datetime(ts_clean, format="%d/%m/%Y %H:%M:%S")
    df["ts"] = (
        df["ts"]
        .dt.tz_localize("Europe/Tallinn",
                         ambiguous=ambiguous_arr,
                         nonexistent="shift_forward")
        .dt.tz_convert("UTC")
    )

    frames.append(df[["ts", "ptype", "mw"]])

raw = pd.concat(frames, ignore_index=True)
print(f"\nTimestamp range: {raw['ts'].min()} → {raw['ts'].max()}")
print(f"Total rows: {len(raw):,}")


# ── 3. Coerce "n/e" to NaN, drop Wind Offshore (all NaN for Estonia) ────────
raw["mw"] = pd.to_numeric(raw["mw"], errors="coerce")
raw = raw[raw["ptype"] != "Wind Offshore"]


# ── 4. Pivot to wide format (one row per hour, one column per type) ──────────
production = (
    raw
    .pivot_table(index="ts", columns="ptype", values="mw", aggfunc="mean")
    .sort_index()
)
production.columns = (
    production.columns
    .str.strip()
    .str.lower()
    .str.replace(" ", "_", regex=False)
    .str.replace("/", "_", regex=False)
)
print(f"\nPivoted shape (before dedup): {production.shape}")


# ── 5. Deduplicate year-boundary overlaps + resample to clean hourly UTC ─────
production = production[~production.index.duplicated(keep="first")]
production = production.resample("h").mean()
print(f"Shape after dedup + resample: {production.shape}")
print(f"NaN count — wind_onshore: {production['wind_onshore'].isna().sum()}")


# ── 6. Capacity factor vs piecewise installed capacity ───────────────────────
# Piecewise approximation from Elering annual reports.
capacity_history = pd.Series({
    pd.Timestamp("2019-01-01", tz="UTC"): 310,
    pd.Timestamp("2022-01-01", tz="UTC"): 403,
    pd.Timestamp("2023-01-01", tz="UTC"): 500,
    pd.Timestamp("2024-01-01", tz="UTC"): 640,
    pd.Timestamp("2025-01-01", tz="UTC"): 694,
})
installed = capacity_history.reindex(production.index, method="ffill").fillna(310)
production["capacity_factor"] = (production["wind_onshore"] / installed).clip(0, 1)
production["installed_mw"]    = installed

jan_cf = production.loc[
    (production.index.month == 1) & (production.index.year < 2026),
    "capacity_factor"
].mean()
print(f"\nHistorical January mean CF (2019–2025): {jan_cf:.3f}")


# ── 7. Save ───────────────────────────────────────────────────────────────────
production.to_csv(OUT_CSV)
print(f"\nSaved → {OUT_CSV}")
print(f"Shape: {production.shape}")
print(f"Columns: {list(production.columns)}")
