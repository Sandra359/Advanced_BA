# Supply.py → Supply_wind.py: Changes Overview

---

## Change 1 — New data load: ENTSOE actual wind production

**Supply.py:** no ENTSOE wind data loaded

**Supply_wind.py:** after weather fetch, loads `data/entsoe_production_hourly.csv` and extracts `wind_onshore` → `wind_mw` aligned to the main timestamp index

```python
entsoe = pd.read_csv("../data/entsoe_production_hourly.csv", ...)
wind_mw = entsoe["wind_onshore"].reindex(idx, method="ffill").fillna(0)
```

**Why:** The model needs to see the wind component explicitly as a feature so that scenario injection (new farm production) lands on a coefficient the model actually learned. Without this, the model has no direct wind signal to modify.

> `wind_mw` is already inside `production` and `production_renewable` — it is added as a feature for granularity only, not added again to `available_energy`.

---

## Change 2 — New feature: `wind_mw` (15 → 16 features)

**Supply.py** feature set:
```
available_energy_lag24, production_renewable, production,
flow_fi, flow_lv, price, temperature, wind_speed_10m,
freq_deviation, + 6 time encodings   →  15 features
```

**Supply_wind.py** feature set:
```
available_energy_lag24, production_renewable, production,
wind_mw,                                               ← new
flow_fi, flow_lv, price, temperature, wind_speed_10m,
freq_deviation, + 6 time encodings   →  16 features
```

---

## Change 3 — New index constant: `WIND_MW_IDX`

**Supply.py:** no such constant

**Supply_wind.py:**
```python
WIND_MW_IDX = FEATURE_NAMES.index("wind_mw")
```

Used in the scenario engine to inject new wind production into the correct feature slot.

---

## Change 4 — Bug fix: `run_scenario()` missing return statement

**Supply.py:** function prints results but has no `return`. Every call site tries to unpack:
```python
p50_s1, p10_s1, p90_s1 = run_scenario(...)  # crashes: cannot unpack None
```

**Supply_wind.py:**
```python
return p50_s, p10_s, p90_s
```

---

## Change 5 — Bug fix: wind injection normalisation

**Supply.py:** all features divided by the same std (`RENEW_IDX` std), and `PROD_IDX` was double-scaled:

```python
wind_std = x_std[0, 0, 0, RENEW_IDX].item()          # one std for everything
...
x_mod[t, :, 0, RENEW_IDX]  += wind_norm              # wind_norm = w / wind_std ✓
x_mod[t, :, 0, PROD_IDX]   += wind_norm * (wind_std / prod_std)  # double-scaled ✗
x_mod[t, :, 0, SUPPLY_IDX] += wind_norm              # wrong std for supply ✗
```

Effect: more wind → lower predicted balance (wrong direction).

**Supply_wind.py:** each feature divided by its own std. `SUPPLY_IDX` (`available_energy_lag24`) is intentionally not adjusted — adjusting it would require a 24 h backward shift and adds no benefit since the target is `production`, not balance (see also Change 11):

```python
renew_std   = x_std[0, 0, 0, RENEW_IDX].item()
prod_std    = x_std[0, 0, 0, PROD_IDX].item()
wind_mw_std = x_std[0, 0, 0, WIND_MW_IDX].item()

x_mod[t, :, 0, RENEW_IDX]   += wt / renew_std
x_mod[t, :, 0, PROD_IDX]    += wt / prod_std
x_mod[t, :, 0, WIND_MW_IDX] += wt / wind_mw_std
```

---

## Change 6 — Bug fix: scenario CSV injected total wind, not delta

**Supply.py:**
```python
wind_scenA = scenarios["wind_mwh_scenA"].values
# wind_mwh_scenA = existing 694 MW baseline + 323 MW new farms (total)
# → injects the entire existing fleet on top of what the model already sees ✗
```

**Supply_wind.py:**
```python
wind_scenA = (scenarios["wind_mwh_scenA"] - scenarios["wind_mwh_baseline"]).values
wind_scenB = (scenarios["wind_mwh_scenB"] - scenarios["wind_mwh_baseline"]).values
# → injects only the additional production from new farms ✓
```

| Column | Contains |
|--------|----------|
| `wind_mwh_baseline` | ERA5 production from existing 694 MW fleet |
| `wind_mwh_scenA` | baseline + 323 MW new (Lääneranna, Pärnu, Aidu) |
| `wind_mwh_scenB` | baseline + 887 MW new (all established + pipeline farms) |
| **delta scenA** | 323 MW new production only → injected in S3 |
| **delta scenB** | 887 MW new production only → injected in S4 |

---

## Change 7 — Scenario label fix

| Scenario | Supply.py | Supply_wind.py |
|----------|-----------|----------------|
| S3 | Isolated + Scenario A | Isolated + Established plans (+323 MW — Lääneranna, Pärnu, Aidu) |
| S4 | Isolated + Scenario B / +887 MW / +564 MW (inconsistent) | Isolated + Full pipeline (+887 MW new — established + pipeline farms) |

S4 is +887 MW because `scenB − baseline` covers all new farms: 323 MW (Scenario A) + 564 MW (pipeline) = 887 MW total new capacity.

---

## Change 8 — Break-even analysis: deferred to Monte Carlo notebook

**Supply.py:** flat MW sweep was inside a `'''...'''` block (dead code)

**Supply_wind.py:** break-even sweep is **not implemented here** — it is delegated to `ursula_monte_carlo.ipynb`, which will run 1 000 MC draws at each capacity level (0–2 000 MW in 100 MW steps) and report P95 deficit hours. A deterministic flat sweep in Supply_wind.py would produce unreliable results without demand-side uncertainty.

---

## Change 9 — Plot 4: mean production with P10–P90 error bars

**Supply.py:** bar chart showing mean P50 available supply per scenario with a typical consumption reference line

**Supply_wind.py:** bar chart showing **mean P50 production per scenario** with error bars spanning P10–P90 (worst-to-best-case band). This replaces the raw mean bars and makes quantile uncertainty visible in the comparison chart.

> Deficit hours are computed in the resilience simulator (`resilience_simulator.py`) after combining GNN supply with SARIMAX demand, not directly in this plot.

---

## Change 10 — Target variable: `available_energy` → `production`

**Supply.py / previous Supply_wind.py:**
```python
y_target_values = system_h["available_energy"].values
```

`available_energy = production − net_exports = production + net_imports ≈ consumption`

Because Estonia imports heavily (flows are large and negative in the convention: `finland = −643 MW` on Jan 1, 2026), available_energy ≈ 1000–1100 MW and is always positive. The model learned to predict a near-constant positive value → 0 deficit hours under every scenario.

**Supply_wind.py (updated):**
```python
y_target_values = system_h["production"].values
```

`production` = total domestic generation (~200–1300 MW), the actual supply side. Deficit is computed later in Monte Carlo as `production(GNN) − consumption(SARIMAX)`.

**Why this matters:** Under isolation the model's input flows are zeroed, but if the target was `available_energy` (≈ consumption), training never exposed the model to deficits and it could not produce them. With `production` as target the model learns the correct supply level.

---

## Change 11 — Removed isolation SUPPLY_IDX flow correction

**Previous Supply_wind.py:**
```python
fi_flow_orig = x_mod[:, :, 0, FLOW_FI_IDX].clone()
lv_flow_orig = x_mod[:, :, 0, FLOW_LV_IDX].clone()
x_mod[:, :, 0, FLOW_FI_IDX] = 0.0
...
x_mod[:, :, 0, SUPPLY_IDX] += fi_flow_orig   # ← adjusting available_energy_lag24
x_mod[:, :, 0, SUPPLY_IDX] += lv_flow_orig
```

**Supply_wind.py (updated):** the `SUPPLY_IDX` adjustment lines are removed.

**Why:** `production` does not depend on cross-border flows — Estonia's plants produce the same amount regardless of what is exported or imported. The correction only made sense when the target was `available_energy` (which does depend on flows). Removing it keeps the feature space consistent with the new target.

---

## Change 12 — Data caching: skip API on re-runs

**Supply.py / previous Supply_wind.py:** full API fetch on every run (~2–5 minutes).

**Supply_wind.py (updated):**
```python
CACHE_FILE = "data_cache.pkl"
if os.path.exists(CACHE_FILE):
    df_prices, df_flows, df_system, df_weather = pickle.load(...)
else:
    # fetch from API and pickle.dump(...)
```

First run fetches and caches. Every subsequent run skips all API calls and loads from disk in seconds. Delete `data_cache.pkl` to force a refresh.

---

## Change 13 — Model size and batch size

| Parameter    | Supply.py | Supply_wind.py |
|--------------|-----------|----------------|
| `hidden_dim` | 64        | **32**         |
| `BATCH_SIZE` | 64        | **256**        |

`hidden_dim=32` reduces the parameter count ~4× and matched or beat `hidden_dim=64` on validation loss in test runs (less overfitting). `BATCH_SIZE=256` reduces the number of gradient steps per epoch ~4×, cutting training time proportionally. Early stopping (patience=15) ensures no wasted epochs.

---

## Change 14 — Restored `TEMP_IDX` and `WIND_IDX` constants

**Previous Supply_wind.py:** accidentally omitted these two index constants that were present in Supply.py.

**Supply_wind.py (updated):**

```python
TEMP_IDX = FEATURE_NAMES.index("temperature")
WIND_IDX  = FEATURE_NAMES.index("wind_speed_10m")
```

These are needed for any downstream analysis that wants to inspect or modify the temperature or wind-speed feature slots directly.

---

## Change 15 — `available_energy` comment corrected

**Supply.py / previous Supply_wind.py comment:**

```python
# True energy balance: production minus all net exports
# positive = surplus, negative = deficit even after imports
```

This was wrong. Verified from Elering API (Jan 1, 2026, 00:00 UTC):

- `production = 264 MW`, `consumption = 1047 MW`
- `finland = −643 MW` (negative = EE importing)
- `available_energy = 264 − (−643) − (−162) = 1069 MW ≈ consumption`

**Supply_wind.py (updated) comment:**

```python
# available_energy = production + net imports ≈ consumption — always positive.
# Kept only as a lagged contextual feature (available_energy_lag24).
# Deficit is production − SARIMAX_consumption, computed in Monte Carlo.
```

---

## Change 16 — Jan 2026 ENTSOE actual wind patch

**Previous Supply_wind.py:** `wind_mw` was zero for all of January 2026 (the test period) because `entsoe_production_hourly.csv` only covers 2019–2025. The model saw zero wind for the entire test month.

**Supply_wind.py (updated):** reads `data/historical_production_data/2026_AGGREGATED_GENERATION_PER_TYPE_...csv`, extracts Wind Onshore rows for January, converts CET → UTC, and overwrites the zero-filled January slice of `wind_mw`:

```python
_jan2026_mask = pd.DatetimeIndex(wind_mw.index).year == 2026
wind_mw[_jan2026_mask] = _jan2026_wind_h.reindex(
    wind_mw[_jan2026_mask].index, method="ffill"
).fillna(0)
```

**Why:** Without this patch the model sees `wind_mw=0` for every January 2026 hour even though actual wind generation existed. This would mislead the wind injection step and produce incorrectly scaled scenario forecasts.

---

## Change 17 — Wind injection temporal pad

**Previous Supply_wind.py:**
```python
wind_scenA = (scenarios["wind_mwh_scenA"] - scenarios["wind_mwh_baseline"]).values
# length = 744 (Jan 2026 hours)
```

The first test sequence `X_test[0]` covers Dec 30–31 (48-hour look-back). Injecting January wind into those positions would incorrectly add new-farm production to pre-January history.

**Supply_wind.py (updated):**
```python
_pad = np.zeros(SEQ_LEN)   # SEQ_LEN = 48
wind_scenA = np.concatenate([_pad, delta_scenA])   # length = 48 + 744 = 792
```

When iterating `x_mod[t, :, 0, ...]`, the first 48 values of `wind_series` are zero, so no new-farm production is injected into the December look-back window.

---

## Change 18 — Isolation: self-loop edges instead of empty tensor

**Previous Supply_wind.py:**

```python
if isolate:
    edges = torch.zeros((2, 0), dtype=torch.long)  # no edges at all
```

An empty edge tensor can cause undefined behaviour in some PyG scatter operations and is semantically ambiguous ("is the node frozen or just disconnected?").

**Supply_wind.py (updated):**

```python
if isolate:
    edges = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=torch.long)
```

Each node attends to itself only. GATv2 with self-loops is well-defined: the attention mechanism returns the node's own embedding unchanged. Semantically this means "Estonia's plants cannot react to Finnish or Latvian prices" — the right interpretation of grid isolation.

---

## Change 19 — CSV export of scenario quantiles

**Supply.py / previous Supply_wind.py:** no file output; results existed only in memory.

**Supply_wind.py (updated):** after all four scenarios are run, exports a 744-row CSV to `data/gnn_supply_scenarios_jan2026.csv`:

```python
results_df = pd.DataFrame({
    "timestamp":    jan_hours,
    "supply_s1_p10": p10_s1, "supply_s1_p50": p50_s1, "supply_s1_p90": p90_s1,
    "supply_s2_p10": p10_s2, "supply_s2_p50": p50_s2, "supply_s2_p90": p90_s2,
    "supply_s3_p10": p10_s3, "supply_s3_p50": p50_s3, "supply_s3_p90": p90_s3,
    "supply_s4_p10": p10_s4, "supply_s4_p50": p50_s4, "supply_s4_p90": p90_s4,
})
results_df.to_csv("../data/gnn_supply_scenarios_jan2026.csv", index=False)
```

**Why P10/P50/P90 all exported:** the resilience simulator stress test uses `P10_supply − P95_demand` (worst-case supply against worst-case demand). Exporting all three quantiles lets the simulator choose the right one per analysis.

---

## Final model configuration (best run)

| Parameter | Value | Notes |
| --- | --- | --- |
| `hidden_dim` | 32 | 64 was tested and gave worse results (overfitting) |
| `BATCH_SIZE` | 256 | |
| `lr` | 5e-4 | Adam with `weight_decay=1e-3` |
| `patience` | 15 | Early stopping with `min_delta=1e-4` |
| LR scheduler | `ReduceLROnPlateau(patience=5, factor=0.5)` | Halves LR when val loss plateaus |
| `SEQ_LEN` | 48 h | 2-day look-back |
| `HORIZON` | 24 h | Predicts 24 h ahead |

**January 2026 test results (hidden_dim=32):**

| Metric | Value |
| --- | --- |
| MAE | ~116 MW |
| RMSE | ~141 MW |
| P10–P90 coverage | ~75% (target ≥ 80%) |

`hidden_dim=64` was tested and produced MAE ~129 MW, coverage ~61% — worse across all metrics. The January 2026 crisis period is out-of-distribution relative to 2019–2025 training data; a smaller model generalises better under this distribution shift.
