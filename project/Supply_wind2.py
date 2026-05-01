import os
import pickle
import pandas as pd
import helper_functions_GNN as helper
from STGNN import STGNN
import numpy as np
import torch
import matplotlib.pyplot as plt
import power_scaler as ps
import GNN_optimizer as opt

current_dir = os.path.dirname(os.path.abspath(__file__))

BASE = "https://dashboard.elering.ee/api"
START = "2019-01-01T00:00:00.000Z"
END = "2026-02-01T00:00:00.000Z"

# we use cache for the API calls, so we don't have to wait for them every time we run the notebook. The cache is stored in a pickle file called "data_cache.pkl". If the file exists, we load the data from it. If it doesn't exist, we fetch the data from the API and save it to the cache file for future use. This way, we only pay the API call cost once, and subsequent runs are much faster. You can delete the cache file if you want to refresh the data from the API.
CACHE_FILE = os.path.join(current_dir, "data_cache.pkl")

if os.path.exists(CACHE_FILE):
    print("[CACHE] Loading cached data (skip API calls)...")
    with open(CACHE_FILE, "rb") as _f:
        df_prices, df_flows, df_system, df_weather = pickle.load(_f)
    print("[CACHE] Done.")
else:
    print("[API] Fetching data — first run, will cache to disk...")
    df_prices = helper.fetch_all(helper.get_nps_prices, START, END)
    df_flows = helper.fetch_all(helper.get_cross_border_flows, START, END)
    df_system = helper.fetch_all(helper.get_system_production, START, END)
    df_weather = helper.get_weather_openmeteo(
        lat=58.90, lon=24.75, start=START, end=END
    )
    with open(CACHE_FILE, "wb") as _f:
        pickle.dump((df_prices, df_flows, df_system, df_weather), _f)
    print("[API] Data fetched and cached.")

df_prices_hourly = df_prices.resample("h").mean()
df_flows_hourly = df_flows.resample("h").mean().fillna(0)
df_system_hourly = df_system.resample("h").mean()
df_weather_hourly = df_weather.resample("h").mean()

prices_h = df_prices_hourly.copy()
flows_h = df_flows_hourly.copy()
system_h = df_system_hourly.copy()
weather_h = df_weather_hourly.copy()

idx = prices_h.index
flows_h = flows_h.reindex(idx, method="ffill")
system_h = system_h.reindex(idx, method="ffill")
weather_h = df_weather_hourly.reindex(idx, method="ffill")

# available_energy = production + net imports (negative flows = EE is importing).
# This is approximately equal to consumption — it is always positive and is NOT
# a deficit signal. It is kept only as a lagged contextual feature (24 h lag)
# so the model can see typical load levels. The GNN target is production alone;
# deficit is computed later in Monte Carlo as production − SARIMAX_consumption.
system_h["available_energy"] = (
    system_h["production"]
    - flows_h[("ee", "fi")]
    - flows_h[("ee", "lv")]
    - flows_h[("ee", "ru_narva")]
    - flows_h[("ee", "ru_pihkva")]
)
print(
    f"    available_energy range: {system_h['available_energy'].min():.0f} – {system_h['available_energy'].max():.0f} MW  (≈ consumption, always positive)"
)

# Cyclic calendar features
hour_sin = np.sin(2 * np.pi * idx.hour / 24)
hour_cos = np.cos(2 * np.pi * idx.hour / 24)
dow_sin = np.sin(2 * np.pi * idx.dayofweek / 7)
dow_cos = np.cos(2 * np.pi * idx.dayofweek / 7)
month_sin = np.sin(2 * np.pi * (idx.month - 1) / 12)
month_cos = np.cos(2 * np.pi * (idx.month - 1) / 12)

essential_cols = ["available_energy", "production", "production_renewable", "frequency"]
for col in essential_cols:
    if col in system_h.columns:
        system_h[col] = system_h[col].interpolate(method="linear")
        system_h[col] = system_h[col].ffill().bfill()

freq_deviation = system_h["frequency"] - 50.0



# Flow sign diagnostic
print(
    "[DIAGNOSTIC] Mean flow EE→FI (should be negative if EE is net importer from FI):",
    flows_h[("ee", "fi")].mean().round(1),
)
print("[DIAGNOSTIC] Mean flow EE→LV:", flows_h[("ee", "lv")].mean().round(1))

# --- Load actual ENTSOE wind production (2019–2025) as an additional feature ---
# wind_mw is the ENTSOE-metered actual wind generation. It is already included
# inside `production` and `production_renewable` — this feature just gives the
# model an explicit view of the wind component, which is necessary so that
# scenario injection (new farms) lands on the right coefficient.
# We do NOT add it again to available_energy here; available_energy already
# reflects it via production.

# Change here!!
entsoe = pd.read_csv(
    os.path.join(current_dir, "..", "data", "entsoe_production_hourly.csv"), index_col=0, parse_dates=True
)
wind_mw = entsoe["wind_onshore"].reindex(idx, method="ffill").fillna(0)
print(f"    wind_mw NaN count after reindex: {wind_mw.isna().sum()}")

# ENTSOE data ends at 2025. For January 2026 (test period), load actual metered
# wind from the 2026 ENTSOE file. Only January is used — later months are unreliable.
_raw_2026 = pd.read_csv(
    os.path.join(current_dir, "..", "data", "historical_production_data",
    "2026_AGGREGATED_GENERATION_PER_TYPE_GENERATION_202512312300-202612312300.csv")
)
_jan_wind = _raw_2026[
    (_raw_2026["Production Type"] == "Wind Onshore")
    & (_raw_2026["MTU (CET/CEST)"].str.startswith("01/"))
].copy()
_jan_wind["time"] = pd.to_datetime(
    _jan_wind["MTU (CET/CEST)"].str.split(" - ").str[0],
    format="%m/%d/%Y %H:%M:%S",
)
_jan_wind["time"] = _jan_wind["time"].dt.tz_localize("CET").dt.tz_convert("UTC")
_jan_wind["Generation (MW)"] = pd.to_numeric(
    _jan_wind["Generation (MW)"], errors="coerce"
)
_jan2026_wind_h = _jan_wind.set_index("time")["Generation (MW)"].resample("h").mean()

_jan2026_mask = pd.DatetimeIndex(wind_mw.index).year == 2026
wind_mw[_jan2026_mask] = _jan2026_wind_h.reindex(
    wind_mw[_jan2026_mask].index, method="ffill"
).fillna(0)
print(
    f"    wind_mw: ENTSOE actual for 2019-2025 + Jan 2026 ({_jan2026_mask.sum()} hours); Jan 2026 mean = {wind_mw[_jan2026_mask].mean():.1f} MW"
)
# sandra change
# EE node: full feature set
# available_energy is lagged 24 h so the model can't trivially copy the most
# recent balance value (which would overlap with the prediction window).

# Defined early because it is used for the available_energy lag below
HORIZON = 24
ee_feats = pd.DataFrame({
    
    "production_lag1": system_h["production"].shift(1),
    "production_lag24": system_h["production"].shift(HORIZON),
    "available_energy_lag24": system_h["available_energy"].shift(HORIZON),
    "production_renewable": system_h["production_renewable"].shift(HORIZON),
    
    "flow_fi": flows_h[("ee", "fi")].shift(1), 
    "flow_lv": flows_h[("ee", "lv")].shift(1),
    "price": prices_h["ee"].shift(1),
    

    "temperature": weather_h["temperature"],
    "wind_speed_10m": weather_h["wind_speed_10m"],
    "wind_mw": wind_mw, 
    
    # Calendar (Static, always safe)
    "hour_sin": hour_sin, "hour_cos": hour_cos,
    "dow_sin": dow_sin, "dow_cos": dow_cos,
    "month_sin": month_sin, "month_cos": month_cos,
}, index=idx).fillna(0)


fi_feats = (
    pd.DataFrame({"price": prices_h["fi"], "flow_fi": flows_h[("ee", "fi")]}, index=idx)
    .reindex(columns=ee_feats.columns, fill_value=0)
    .fillna(0)
)
lv_feats = (
    pd.DataFrame({"price": prices_h["lv"], "flow_lv": flows_h[("ee", "lv")]}, index=idx)
    .reindex(columns=ee_feats.columns, fill_value=0)
    .fillna(0)
)
lt_feats = (
    pd.DataFrame({"price": prices_h["lt"]}, index=idx)
    .reindex(columns=ee_feats.columns, fill_value=0)
    .fillna(0)
)

node_data = np.stack(
    [
        ee_feats.values,
        fi_feats.values,
        lv_feats.values,
        lt_feats.values,
    ],
    axis=1,
)

# sandra change
NUM_NODES = 4
NUM_FEATURES = node_data.shape[2]
FEATURE_NAMES = list(ee_feats.columns)

# Autoregressive features
PROD_LAG1_IDX = FEATURE_NAMES.index("production_lag1")
PROD_LAG24_IDX = FEATURE_NAMES.index("production_lag24")
AVAIL_LAG_IDX = FEATURE_NAMES.index("available_energy_lag24")

# Renewable / production features used in wind injection
RENEW_IDX = FEATURE_NAMES.index("production_renewable")
WIND_MW_IDX = FEATURE_NAMES.index("wind_mw")

# Flow features used in isolation scenario
FLOW_FI_IDX = FEATURE_NAMES.index("flow_fi")
FLOW_LV_IDX = FEATURE_NAMES.index("flow_lv")

# Weather features
TEMP_IDX = FEATURE_NAMES.index("temperature")
WIND_IDX = FEATURE_NAMES.index("wind_speed_10m")

print(f"  Features ({NUM_FEATURES}): {FEATURE_NAMES}")

# ==================================================
# STEP 2: SEQUENCES + SPLITS
# ==================================================
print("\n[2/6] Creating sequences...")

SEQ_LEN = 48  # 2 days

y_target_values = system_h["production"].values  # production as target

X_list, y_list = [], []
for t in range(SEQ_LEN, len(node_data) - HORIZON):
    X_list.append(node_data[t - SEQ_LEN : t])
    y_list.append(y_target_values[t + HORIZON - 1])

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.float32)

DIFF_LAG = 24
y_levels   = y.copy()                       # keep originals for inversion
y_diff     = y[DIFF_LAG:] - y[:-DIFF_LAG]  # differenced target
X          = X[DIFF_LAG:]                   # align X with differenced y
y_last     = y[:-DIFF_LAG]                  # level at t-DIFF_LAG (for inversion)
y          = y_diff                         # model trains on differences

jan2026_start = np.searchsorted(
    idx[SEQ_LEN + DIFF_LAG:-HORIZON], pd.Timestamp("2026-01-01", tz="UTC")
)

X_train_full,  y_train_full  = X[:jan2026_start],  y[:jan2026_start]
X_test,        y_test        = X[jan2026_start:],  y[jan2026_start:]
y_last_test                  = y_last[jan2026_start:]       # for inversion
y_levels_test                = y_levels[DIFF_LAG + jan2026_start:]  # true absolute levels

val_split = int(len(X_train_full) * 0.8)
X_train, y_train = X_train_full[:val_split], y_train_full[:val_split]
X_val, y_val = X_train_full[val_split:], y_train_full[val_split:]

print(
    f"Train: {len(X_train):,} ({X_train.shape}) | Val: {len(X_val):,} | Test (Jan 2026): {len(X_test):,}"
)

scaler = ps.PowerScaler(X_train, y_train, diff_lag=DIFF_LAG)

X_train_t = scaler.scale_x(X_train)
X_val_t = scaler.scale_x(X_val)
X_test_t = scaler.scale_x(X_test)

y_train_t = scaler.scale_y(y_train).view(-1, 1)
y_val_t = scaler.scale_y(y_val).view(-1, 1)
y_test_t = scaler.scale_y(y_test).view(-1, 1)

edge_index = torch.tensor(
    [
        [0, 1, 0, 2, 2, 3],
        [1, 0, 2, 0, 3, 2],
    ],
    dtype=torch.long,
)

edge_weight = torch.tensor([
    abs(flows_h[("ee", "fi")].mean()),   # EE→FI
    abs(flows_h[("ee", "fi")].mean()),   # FI→EE
    abs(flows_h[("ee", "lv")].mean()),   # EE→LV
    abs(flows_h[("ee", "lv")].mean()),   # LV→EE
    abs(flows_h[("ee", "lv")].mean()),   # LV↔LT: proxy using EE→LV (no direct data)
    abs(flows_h[("ee", "lv")].mean()),   # LT↔LV: same proxy
], dtype=torch.float)

edge_weight = edge_weight / edge_weight.max()

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
model = STGNN(NUM_FEATURES, hidden_dim=32, seq_len=SEQ_LEN, dropout=0.3).to(device)
edge_index = edge_index.to(device)
edge_weight = edge_weight.to(device)  # move to same device as model
x_std = scaler.x_std.to(device)

print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,} on {device}")

# ==================================================
# STEP 4: TRAINING LOOP
# ==================================================
print("\n[4/6] Training...")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5, weight_decay=1e-2)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=8, factor=0.5
)

BATCH_SIZE = 256
EPOCHS = 50

early_stopping = opt.EarlyStopping(patience=10, min_delta=1e-4, path="best_stgnn.pt")

train_losses, val_losses = [], []

for epoch in range(EPOCHS):
    model.train()
    perm = torch.randperm(len(X_train_t))
    epoch_loss = 0.0
    n_batches = 0

    for i in range(0, len(X_train_t), BATCH_SIZE):
        idx_b = perm[i : i + BATCH_SIZE]
        xb = X_train_t[idx_b].to(device)
        yb = y_train_t[idx_b].to(device)

        optimizer.zero_grad()
        preds = model(xb, edge_index, edge_weight)
        loss = helper.quantile_loss(preds, yb)

        if torch.isnan(loss):
            print(f"  [!] NaN loss at epoch {epoch + 1}, batch {i}. Skipping.")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1

    model.eval()
    val_losses_b = []
    with torch.no_grad():
        for i in range(0, len(X_val_t), BATCH_SIZE):
            xb_v = X_val_t[i : i + BATCH_SIZE].to(device)
            yb_v = y_val_t[i : i + BATCH_SIZE].to(device)
            v_loss = helper.quantile_loss(model(xb_v, edge_index, edge_weight), yb_v)
            val_losses_b.append(v_loss.item())

    avg_train_loss = epoch_loss / max(n_batches, 1)
    avg_val_loss = np.mean(val_losses_b)

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    scheduler.step(avg_val_loss)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(
            f"  Epoch {epoch + 1:3d}: train={avg_train_loss:.4f} | "
            f"val={avg_val_loss:.4f} | lr={optimizer.param_groups[0]['lr']:.6f}"
        )

    early_stopping.step(avg_val_loss, model, epoch + 1)
    if early_stopping.stop:
        print(f"\n  Early stopping triggered after epoch {epoch + 1}.")
        break

model = early_stopping.load_best(model)
print(f"  Training complete. Best val loss: {early_stopping.best_loss:.4f}")

# ==================================================
# STEP 5: EVALUATE + SCENARIOS
# ==================================================
print("\n[5/6] Evaluating on January 2026...")
 
model.eval()
with torch.no_grad():
    test_preds = model(X_test_t.to(device), edge_index, edge_weight).cpu().numpy()
 
print(f"  Predictions generated for {len(test_preds)} samples.")
 
# Baseline model output (S1 = full grid, ingen manipulation)
p10 = scaler.inverse_y(test_preds[:, 0], last_level=y_last_test)
p50 = scaler.inverse_y(test_preds[:, 1], last_level=y_last_test)
p90 = scaler.inverse_y(test_preds[:, 2], last_level=y_last_test)
actual = scaler.inverse_y(y_test_t.squeeze().numpy(), last_level=y_last_test)
 
mae      = np.mean(np.abs(p50 - actual))
rmse     = np.sqrt(np.mean((p50 - actual) ** 2))
coverage = np.mean((actual >= p10) & (actual <= p90))
 
print(f"  MAE:              {mae:.1f} MW")
print(f"  RMSE:             {rmse:.1f} MW")
print(f"  P10-P90 coverage: {coverage * 100:.1f}%  (target >= 80%)")
 
# --------------------------------------------------
# Timestamps for januar 2026 test-periode
# --------------------------------------------------
_ts_start = SEQ_LEN + jan2026_start + HORIZON - 1
jan_hours = idx[_ts_start : _ts_start + len(y_test)]   # DatetimeIndex
 
# --------------------------------------------------
# Import delta: hvad mister vi ved isolation?
# Elering konvention: negative flow = EE importerer
# Vi fratrækker KUN import (ikke eksport)
# --------------------------------------------------
fi_flow = flows_h[("ee", "fi")].reindex(jan_hours, method="ffill").fillna(0).values
lv_flow = flows_h[("ee", "lv")].reindex(jan_hours, method="ffill").fillna(0).values
 
"""
import_delta = -(fi_flow + lv_flow)   # both export and import
 
print(f"\n  Import delta (isolation):")
print(f"    Mean: {import_delta.mean():.1f} MW/h tabt ved isolation")
print(f"    Max:  {import_delta.max():.1f} MW/h")
"""
 
# --------------------------------------------------
# Vind-scenarier: kun delta fra NYE vindfarme
# --------------------------------------------------
data_path     = os.path.join(current_dir, "..", "data", "wind_production_scenarios.csv")
wind_scenarios = pd.read_csv(data_path, index_col=0, parse_dates=True)
 
# Reindex til januar 2026 timestamps
wind_scenA_delta = (
    (wind_scenarios["wind_mwh_scenA"] - wind_scenarios["wind_mwh_baseline"])
    .reindex(jan_hours, method="ffill")
    .fillna(0)
    .values
)
wind_scenB_delta = (
    (wind_scenarios["wind_mwh_scenB"] - wind_scenarios["wind_mwh_baseline"])
    .reindex(jan_hours, method="ffill")
    .fillna(0)
    .values
)
 
print(f"\n  Wind deltas:")
print(f"    Scenario A mean: {wind_scenA_delta.mean():.1f} MW")
print(f"    Scenario B mean: {wind_scenB_delta.mean():.1f} MW")
 
# --------------------------------------------------
# Byg scenarier via post-hoc justering
# --------------------------------------------------
# Usikkerhedsspænd fra modellen bevares i alle scenarier
#spread_lo = p50 - p10   # nedre usikkerhed
#spread_hi = p90 - p50   # øvre usikkerhed
 
# Netto import til EE (positiv = EE modtager energi fra udlandet)
# fi_flow negativ = EE importerer fra FI → vi vil have positiv værdi
netto_import = -(fi_flow + lv_flow)   # mean ≈ +587 MW

# S1: Fuld grid = produktion + hvad vi importerer netto
p50_s1 = p50 + netto_import
p10_s1 = p10 + netto_import           # bevar spread
p90_s1 = p90 + netto_import

# S2: Isolation = kun indenlandsk produktion (model output uændret)
p50_s2, p10_s2, p90_s2 = p50.copy(), p10.copy(), p90.copy()

# S3: Isolation + Scenario A vind
p50_s3 = p50_s2 + wind_scenA_delta
p10_s3 = p10_s2 + wind_scenA_delta
p90_s3 = p90_s2 + wind_scenA_delta

# S4: Isolation + Scenario B vind
p50_s4 = p50_s2 + wind_scenB_delta
p10_s4 = p10_s2 + wind_scenB_delta
p90_s4 = p90_s2 + wind_scenB_delta

# Clip til 0 — produktion kan ikke være negativ
p10_s1 = np.maximum(p10_s1, 0)
p10_s2 = np.maximum(p10_s2, 0)
p10_s3 = np.maximum(p10_s3, 0)
p10_s4 = np.maximum(p10_s4, 0)

print("\n  --- Scenario comparison (January 2026) ---")
for name, s_p50, s_p10, s_p90 in [
    ("S1: Full grid (baseline)",              p50_s1, p10_s1, p90_s1),
    ("S2: Isolated",                          p50_s2, p10_s2, p90_s2),
    ("S3: Isolated + Scenario A (est. wind)", p50_s3, p10_s3, p90_s3),
    ("S4: Isolated + Scenario B (pipeline)",  p50_s4, p10_s4, p90_s4),
]:
    print(f"\n  {name}")
    print(f"    Mean P50: {s_p50.mean():.0f} MW  |  P10: {s_p10.mean():.0f}  |  P90: {s_p90.mean():.0f}")
    print(f"    Min P50:  {s_p50.min():.0f} MW")
 
jan_hours = jan_hours.to_numpy()   # konverter til numpy for plotting

# combine plots from both codes:
# ==================================================
# STEP 6: VISUALIZE
# ==================================================
print("\n[6/6] Visualizing...")

# Derive timestamps correctly from index (Supply_wind.py approach — more robust)
_ts_start = SEQ_LEN + jan2026_start + HORIZON - 1
jan_hours = idx[_ts_start : _ts_start + len(y_test)].to_numpy()

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(
    "ST-GNN: Estonian Energy Production — January 2026", fontsize=14, fontweight="bold"
)

# Plot 1: training curves
axes[0, 0].plot(train_losses, label="Train", lw=2, color="steelblue")
axes[0, 0].plot(val_losses, label="Val", lw=2, color="orange")
axes[0, 0].set_title("Training & Validation Loss")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Quantile loss")
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: sammenlign model produktion (p50) med faktisk produktion
# actual = hvad modellen blev trænet på (kun produktion, ikke import)
# så p50 vs actual er stadig korrekt her
axes[0, 1].fill_between(
    jan_hours, p10, p90, alpha=0.15, color="steelblue", label="P10–P90 band"
)
axes[0, 1].plot(jan_hours, p50, lw=2, color="steelblue", label="P50 forecast (production)")
axes[0, 1].plot(
    jan_hours, actual, lw=1.5, color="red", linestyle="--", label="Actual production"
)
axes[0, 1].set_title(f"Model Production Forecast vs Actual  (MAE = {mae:.0f} MW)")
axes[0, 1].set_ylabel("Production (MW)")
axes[0, 1].set_xlim(jan_hours.min(), jan_hours.max())
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(alpha=0.3)
axes[0, 1].tick_params(axis="x", rotation=30)

# Plot 3: scenario comparison over time — show isolation cost and wind benefit
axes[1, 0].plot(jan_hours, p50_s1, lw=2, color="green", label="S1: Full grid")
axes[1, 0].plot(jan_hours, p50_s2, lw=2, color="red", label="S2: Isolated")
axes[1, 0].plot(
    jan_hours, p50_s3, lw=1.5, color="orange", label="S3: Isolated + Scenario A (established wind plans)"
)
axes[1, 0].plot(jan_hours, p50_s4, lw=1.5, color="gold", label="S4: Isolated + Scenario B (pipeline wind farms)")

# Fill between S1 and S2 — cost of isolation
axes[1, 0].fill_between(
    jan_hours,
    p50_s1,
    p50_s2,
    alpha=0.15,
    color="blue",
    label="Cost of isolation (S1–S2)",
)

# Plot 3 titel
axes[1, 0].set_title("EE Available Energy by Scenario — January 2026")
axes[1, 0].set_ylabel("Available energy (MW)")
axes[1, 0].set_xlim(jan_hours.min(), jan_hours.max())
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(alpha=0.3)
axes[1, 0].tick_params(axis="x", rotation=30)

# Plot 4: mean production per scenario with P10-P90 error bars (Supply_wind approach)
# + consumption reference line (Supply_prod approach)
s_labels = ["S1\nFull grid", "S2\nIsolated", "S3\n Scenario A", "S4\nScenario B"]
s_colors = ["green", "red", "orange", "gold"]
s_p50 = [p50_s1.mean(), p50_s2.mean(), p50_s3.mean(), p50_s4.mean()]
# clip to 0 — quantile crossing (p10 > p50) can produce negative deltas with a poorly
# calibrated model, and errorbar() rejects negative yerr values.
s_err_lo = [
    max(0.0, p50_s1.mean() - p10_s1.mean()),
    max(0.0, p50_s2.mean() - p10_s2.mean()),
    max(0.0, p50_s3.mean() - p10_s3.mean()),
    max(0.0, p50_s4.mean() - p10_s4.mean()),
]
s_err_hi = [
    max(0.0, p90_s1.mean() - p50_s1.mean()),
    max(0.0, p90_s2.mean() - p50_s2.mean()),
    max(0.0, p90_s3.mean() - p50_s3.mean()),
    max(0.0, p90_s4.mean() - p50_s4.mean()),
]

x = np.arange(4)
bars = axes[1, 1].bar(x, s_p50, 0.5, color=s_colors, alpha=0.75)
axes[1, 1].errorbar(
    x,
    s_p50,
    yerr=[s_err_lo, s_err_hi],
    fmt="none",
    color="black",
    capsize=6,
    lw=1.5,
    label="P10–P90 range",
)

# Value labels on bars
for bar, val in zip(bars, s_p50):
    axes[1, 1].text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 10,
        f"{val:.0f}",
        ha="center",
        va="bottom",
        fontsize=8,
    )

# Consumption reference line — key for interpreting deficit risk
axes[1, 1].axhline(
    900, color="black", lw=1.5, linestyle="--", label="Typical consumption (~900 MW)"
)

axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(s_labels, fontsize=9)

# Plot 4 titel  
axes[1, 1].set_ylabel("Mean available energy (MW)")
axes[1, 1].set_title("Mean Available Energy by Scenario  (error bars = P10–P90)")
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("stgnn_resilience.png", dpi=150, bbox_inches="tight")
plt.close()

print("\n✓ Complete! Saved to stgnn_resilience.png")

# Export scenario quantiles for Monte Carlo resilience simulation
results_df = pd.DataFrame({
    "timestamp":     jan_hours,
    "supply_s1_p10": p10_s1, "supply_s1_p50": p50_s1, "supply_s1_p90": p90_s1,
    "supply_s2_p10": p10_s2, "supply_s2_p50": p50_s2, "supply_s2_p90": p90_s2,
    "supply_s3_p10": p10_s3, "supply_s3_p50": p50_s3, "supply_s3_p90": p90_s3,
    "supply_s4_p10": p10_s4, "supply_s4_p50": p50_s4, "supply_s4_p90": p90_s4,
})
_out_csv = os.path.join(current_dir, "..", "data", "gnn_supply_scenarios_jan2026.csv")
results_df.to_csv(_out_csv, index=False)
print(f"\n  Exported scenario quantiles to {_out_csv}")