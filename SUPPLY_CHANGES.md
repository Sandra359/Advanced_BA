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

**Supply_wind.py:** each feature divided by its own std:

```python
renew_std   = x_std[0, 0, 0, RENEW_IDX].item()
prod_std    = x_std[0, 0, 0, PROD_IDX].item()
supply_std  = x_std[0, 0, 0, SUPPLY_IDX].item()
wind_mw_std = x_std[0, 0, 0, WIND_MW_IDX].item()

x_mod[t, :, 0, RENEW_IDX]   += w / renew_std
x_mod[t, :, 0, PROD_IDX]    += w / prod_std
x_mod[t, :, 0, SUPPLY_IDX]  += w / supply_std
x_mod[t, :, 0, WIND_MW_IDX] += w / wind_mw_std       # new: consistent injection
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

## Change 8 — Break-even analysis: activated

**Supply.py:** flat MW sweep was inside a `'''...'''` block (dead code)

**Supply_wind.py:** active, runs flat MW additions from 250 → 2,000 MW under isolation and reports the percentage of hours in deficit at each level

---

## Change 9 — Plot 4: deficit hours instead of mean supply bars

**Supply.py:** bar chart showing mean P50 available supply per scenario with a typical consumption reference line

**Supply_wind.py:** grouped bar chart showing **deficit hours** (balance < 0 MW) and **severe deficit hours** (balance < −100 MW) per scenario — directly answers the resilience question rather than showing an always-positive average
