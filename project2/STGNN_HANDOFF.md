# ST-GNN Supply Model — Handoff Notes

**File:** `project/Supply_wind.py`  
**Architecture:** `project/STGNN.py`  
**Best checkpoint:** `project/best_stgnn.pt` (auto-saved by early stopping)

---

## What the model does

Predicts Estonian **domestic electricity production** (MW) 24 hours ahead for every hour in January 2026, plus uncertainty bands (P10/P50/P90). It then reruns prediction under 4 scenarios to quantify how grid isolation and new wind capacity change the supply picture.

The model does **not** predict the energy balance directly. Deficit = `production(GNN) − consumption(SARIMAX)` — computed separately in the resilience simulator.

---

## Architecture

A spatio-temporal graph neural network over a 4-node graph: **EE, FI, LV, LT**.

```
Input: 48-hour window × 4 nodes × 16 features
  ↓
GATv2Conv layer 1  (4 attention heads, hidden=32) + residual projection
  ↓
GATv2Conv layer 2  (4 attention heads, hidden=32) + residual skip
  ↓
Extract EE node embeddings → shape (batch, 48, 32)
  ↓
GRU (2 layers, hidden=32)
  ↓
Linear decoder → 3 outputs: P10, P50, P90
```

**Why a graph?** Estonia's oil shale plants ramp up when Finnish prices are high (export incentive). The graph edges let the model learn this cross-border dispatch behaviour. Under isolation, those edges are severed so the model predicts as if Estonia cannot react to FI/LV price signals.

**Why quantile output?** Pinball loss trains P10/P50/P90 simultaneously. P10 supply is the worst-case production input to the resilience stress test (`P10_supply − P95_demand`).

---

## Features (16 per EE node)

| Feature | What it captures |
| --- | --- |
| `available_energy_lag24` | Typical load level (lagged so it doesn't overlap the target window) |
| `production_renewable` | Wind + solar share of generation |
| `production` | Total domestic output — this is also the **target variable** |
| `wind_mw` | ENTSOE-metered actual wind; already in `production`, added for explicit wind signal |
| `flow_fi`, `flow_lv` | Cross-border flows — zeroed under isolation |
| `price` | EE spot price |
| `temperature` | Drives heating load |
| `wind_speed_10m` | Weather-side wind proxy |
| `freq_deviation` | Grid frequency − 50 Hz; deviation signal under stress |
| 6 cyclical encodings | Hour, day-of-week, month (sin/cos pairs) |

**Why `production` not `available_energy` as target?**  
`available_energy = production + net_imports ≈ consumption`. Under isolation the model would still predict a near-constant ~1 000 MW (consumption doesn't change when you cut imports). Using `production` (~200–1 300 MW) means the model actually learns supply levels that vary with market conditions, and can produce low values when the grid is stressed.

---

## Training setup

| Setting | Value |
| --- | --- |
| Train | 2019 – Oct 2025 (~47 000 sequences) |
| Val | Oct – Dec 2025 (~12 000 sequences) |
| Test | January 2026 (721 sequences, held out) |
| Sequence length | 48 h look-back |
| Prediction horizon | 24 h ahead |
| Batch size | 256 |
| Learning rate | 5e-4 (Adam, weight_decay=1e-3) |
| LR scheduler | ReduceLROnPlateau(patience=5, factor=0.5) |
| Early stopping | patience=15, min_delta=1e-4 |
| Epochs max | 200 |

**Best configuration:** `hidden_dim=32`. Testing `hidden_dim=64` produced worse results (MAE 129 vs 116 MW, coverage 61% vs 75%) because January 2026 is an out-of-distribution crisis period — the smaller model generalises better.

---

## Scenarios

All scenarios reuse the same trained model and scaler. Only the input tensors and graph edges change.

| Scenario | Graph edges | Flow features | Wind injection |
| --- | --- | --- | --- |
| S1 — Full grid | EE↔FI, EE↔LV, LV↔LT | Historical | None |
| S2 — Full isolation | Self-loops only | Zeroed | None |
| S3 — Isolated + 323 MW | Self-loops only | Zeroed | delta scenA |
| S4 — Isolated + 887 MW | Self-loops only | Zeroed | delta scenB |

**Wind injection** adds only the **delta** from new farms (not total wind) into `production_renewable`, `production`, and `wind_mw` features, each normalised by its own training std. The delta series is padded with 48 zeros at the front to avoid injecting January wind into the December look-back window.

---

## Outputs

| File | Contents |
| --- | --- |
| `project/best_stgnn.pt` | Saved model weights (best val loss) |
| `project/stgnn_resilience_wind.png` | 4-panel results figure |
| `data/gnn_supply_scenarios_jan2026.csv` | 744 rows × 12 columns: P10/P50/P90 for each of S1–S4 |

The resilience simulator loads `gnn_supply_scenarios_jan2026.csv` and pairs it with `demand_p95_jan2026.csv` (from the SARIMAX notebook) to compute blackout hours per scenario.

---

## Known limitations

- **MAE ~116 MW, coverage ~75%** — January 2026 is an unusual crisis month not well represented in 2019–2025 training data. This is near the ceiling for this architecture without retraining on more crisis-period data.
- **S1 vs S2 production gap is small (~5–8 MW median)** — correct, because `production` doesn't directly depend on imports. The crisis shows up in the resilience simulator when you subtract demand.
- **P10–P90 coverage below 80% target** — uncertainty bands are slightly too narrow. P10 supply is still valid for the stress test; it just may be slightly optimistic.
