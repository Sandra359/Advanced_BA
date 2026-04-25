from datetime import datetime
import pandas as pd
import helper_functions_GNN as helper
from STGNN import STGNN
import numpy as np
import torch
import torch.nn as nn
import torch_geometric.nn as gnn
import matplotlib.pyplot as plt
import sys
import power_scaler as ps
import  GNN_optimizer as opt
import os

BASE = "https://dashboard.elering.ee/api"
# Fetching data for 2019-2026
START = "2019-01-01T00:00:00.000Z"
END   = "2026-02-01T00:00:00.000Z"


df_prices = helper.fetch_all(helper.get_nps_prices, START, END)
df_flows = helper.fetch_all(helper.get_cross_border_flows, START, END)
df_system = helper.fetch_all(helper.get_system_production, START, END)

# Simple — just fetch weather for one central Estonian location - we assume that it is representative for the whole country for now, but we can add more locations later if needed
df_weather = helper.get_weather_openmeteo(
    lat=58.90,  # central Estonia / Pärnu area
    lon=24.75,
    start=START,
    end=END
)


# --- 3. standardize to hourly, then daily ---

df_prices_hourly  = df_prices.resample("h").mean() # The data is mostly per 15 minutes and some are hourly
df_flows_hourly   = df_flows.resample("h").mean().fillna(0) # The data is only hourly but resampling to be sure for alignment
df_system_hourly  = df_system.resample("h").mean() # Mostly per 5 minutes, some entries are 0:00 (maybe missing?) and 5 minutes
df_weather_hourly = df_weather.resample("h").mean()

# --- 4. merge into one wide dataframe ---

df_daily = pd.concat([
    df_prices_hourly.add_prefix("price_"),
    df_flows_hourly.add_prefix("flow_"),
    df_system_hourly.add_prefix("system_"),
    df_weather_hourly.add_prefix("weather_"),
], axis=1).sort_index()



flow_cols = [col for col in df_daily.columns if col.startswith("flow_")]
df_daily[flow_cols] = df_daily[flow_cols].fillna(0)
df_daily = df_daily.dropna(how="all") # Thinking about imputing them


#_h Target: True available energy in Estonia = production + imports
# Scenarios: S1 = full grid, S2 = full isolation, S3 = isolated + wind (TBD)

prices_h = df_prices_hourly.copy()
flows_h  = df_flows_hourly.copy()
system_h = df_system_hourly.copy()
weather_h = df_weather_hourly.copy()

idx       = prices_h.index
flows_h   = flows_h.reindex(idx, method="ffill")
system_h  = system_h.reindex(idx, method="ffill")
weather_h = df_weather_hourly.reindex(idx, method="ffill")  # reindex here — idx now defined
# True energy balance: production + all imports - 
system_h["available_energy"] = (
    system_h["production"]
    - flows_h[("ee", "fi")]
    - flows_h[("ee", "lv")] 
    - flows_h[("ee", "ru_narva")]
    - flows_h[("ee", "ru_pihkva")]
) # available energy = energy produced in 


print(f"    Hours without enough energy: {(system_h['available_energy'] < 0).sum()}")



# Calendar features (sine/cosine encoding — avoids treating Mon=1, Sun=7 as numeric)
hour_sin  = np.sin(2 * np.pi * idx.hour / 24)
hour_cos  = np.cos(2 * np.pi * idx.hour / 24)
dow_sin   = np.sin(2 * np.pi * idx.dayofweek / 7)
dow_cos   = np.cos(2 * np.pi * idx.dayofweek / 7)
month_sin = np.sin(2 * np.pi * (idx.month - 1) / 12) # Range is originally from 1 to 12, so subtract 1 to get 0-11 for encoding
month_cos = np.cos(2 * np.pi * (idx.month - 1) / 12)

# Frequency deviation from 50 Hz — real-time grid stress signal -- research laterhttps://clouglobal.com/power-grid-frequency-why-is-it-important/

essential_cols = [ "available_energy", "production", "production_renewable", "frequency" ]

for col in essential_cols:
    if col in system_h.columns:
        system_h[col] = system_h[col].interpolate(method='linear')
        system_h[col] = system_h[col].ffill().bfill()

freq_deviation = (system_h["frequency"] - 50.0)


# Define horizon early so we can use it for the lagged feature below
HORIZON = 24   # predict 24h ahead (also defined again in STEP 2 — consistent)

# ── Flow sign diagnostic ─────────────────────────────────────────────────────
# The Elering API convention: positive = EE exports, negative = EE imports.
# available_energy = production - exports + imports
#                  = production - flow_ee_fi - flow_ee_lv - flow_ee_ru_*
# works ONLY when flows are positive for exports and negative for imports.
# Quick sanity check: on average EE imports from FI, so mean(flow_fi) should
# be negative. If it is positive, flip the sign in the formula above.
print("[DIAGNOSTIC] Mean flow EE→FI (should be negative if EE is net importer from FI):",
      flows_h[("ee", "fi")].mean().round(1))
print("[DIAGNOSTIC] Mean flow EE→LV (should be negative if EE imports from LV):",
      flows_h[("ee", "lv")].mean().round(1))
# If both means are positive, EE is a net exporter to those countries, which is
# plausible. If unexpected, swap the formula sign.
# ─────────────────────────────────────────────────────────────────────────────
#include wind production from OpenWeather in the features — we will also use it in the scenario engine to create a "renewable boom" scenario with higher wind production than historically observed. We can also experiment with using it as a feature only for the EE node, or for all nodes (FI/LV also have some wind, but we don't have good data for it — might add later if needed). For now we add it just for EE and set it to 0 for other nodes, but we can easily change this later if we want to experiment with adding some proxy wind features for the neighbouring countries as well.
entsoe = pd.read_csv(
    "../data/entsoe_production_hourly.csv", index_col=0, parse_dates=True
)
wind_mw = entsoe["wind_onshore"].reindex(idx, method="ffill").fillna(0)
print(f"    wind_mw NaN count after reindex: {wind_mw.isna().sum()}")

# EE node: full feature set
# FIX: available_energy kept but lagged by HORIZON (24h).
# Removing it entirely hurt badly — lagged balance is a genuinely informative
# signal. But using it unlagged caused the model to partially "copy" recent
# values that overlap with the target window. Shifting by 24h ensures the
# model only sees energy balance from before the prediction horizon,
# preserving the predictive signal without any target leakage.
ee_feats = pd.DataFrame({
    "available_energy_lag24": system_h["available_energy"].shift(HORIZON),
    "production_renewable": system_h["production_renewable"],
    "production":           system_h["production"],
    "wind_mw":              wind_mw,  # ENTSOE actual wind; already in production
    "flow_fi":              flows_h[("ee", "fi")],
    "flow_lv":              flows_h[("ee", "lv")],
    "price":                prices_h["ee"],
    "temperature":          weather_h["temperature"],    # ← now active
    "wind_speed_10m":       weather_h["wind_speed_10m"], # ← now active
    "freq_deviation":       freq_deviation,
    "hour_sin":             hour_sin,
    "hour_cos":             hour_cos,
    "dow_sin":              dow_sin,
    "dow_cos":              dow_cos,
    "month_sin":            month_sin,
    "month_cos":            month_cos,
}, index=idx).fillna(0)# Might change it into imputing values later instead of filling with 0, but for now it is fine since we have no missing values in these features

# Neighbouring nodes — only features we actually have for them
fi_feats = pd.DataFrame({
    "price":   prices_h["fi"],
    "flow_fi": flows_h[("ee", "fi")],
}, index=idx).reindex(columns=ee_feats.columns, fill_value=0).fillna(0)

lv_feats = pd.DataFrame({
    "price":   prices_h["lv"],
    "flow_lv": flows_h[("ee", "lv")],
}, index=idx).reindex(columns=ee_feats.columns, fill_value=0).fillna(0)

lt_feats = pd.DataFrame({
    "price": prices_h["lt"],
}, index=idx).reindex(columns=ee_feats.columns, fill_value=0).fillna(0)

# Stack into (T, N, F)
node_data = np.stack([
    ee_feats.values,   # node 0: EE
    fi_feats.values,   # node 1: FI
    lv_feats.values,   # node 2: LV
    lt_feats.values,   # node 3: LT
], axis=1)

# We didn't include Russia since we don't have good features for it

NUM_NODES     = 4
NUM_FEATURES  = node_data.shape[2]
FEATURE_NAMES = list(ee_feats.columns)
SUPPLY_IDX  = FEATURE_NAMES.index("available_energy_lag24")
FLOW_FI_IDX = FEATURE_NAMES.index("flow_fi")
FLOW_LV_IDX = FEATURE_NAMES.index("flow_lv")
RENEW_IDX   = FEATURE_NAMES.index("production_renewable")
PROD_IDX    = FEATURE_NAMES.index("production")
TEMP_IDX    = FEATURE_NAMES.index("temperature")
WIND_IDX    = FEATURE_NAMES.index("wind_speed_10m")
WIND_MW_IDX = FEATURE_NAMES.index("wind_mw")  # ENTSOE wind MW feature index


print(f"    Features ({NUM_FEATURES}): {FEATURE_NAMES}")


# ==================================================
# STEP 2: SEQUENCES + SPLITS
# ==================================================
print("\n[2/6] Creating sequences...")

SEQ_LEN = 48   # 2 days of hourly history

y_target_values = system_h["available_energy"].values

X_list, y_list = [], []
for t in range(SEQ_LEN, len(node_data) - HORIZON):
    X_list.append(node_data[t - SEQ_LEN:t]) # Each sample is a (SEQ_LEN, N, F) tensor covering hours t-SEQ_LEN to t-1
    y_list.append(y_target_values[t + HORIZON - 1])

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.float32)

# January 2026 = held-out test (never seen during training)
jan2026_start = np.searchsorted(
    idx[SEQ_LEN:-HORIZON],
    pd.Timestamp("2026-01-01", tz="UTC")
)
X_train_full, y_train_full = X[:jan2026_start], y[:jan2026_start]
X_test,        y_test       = X[jan2026_start:], y[jan2026_start:]

#we don't use cross validation since we have a clear temporal split and want to simulate real-world forecasting where we train on past data and predict future unseen data. We will use a validation split from the training set to tune hyperparameters and prevent overfitting, but the final evaluation will be on the held-out January 2026 data to see how well our model generalizes to truly unseen conditions.
val_split = int(len(X_train_full) * 0.8)
X_train, y_train = X_train_full[:val_split],  y_train_full[:val_split]
X_val,   y_val   = X_train_full[val_split:],  y_train_full[val_split:]

print(f"Train: {len(X_train):,} ({X_train.shape}) | Val: {len(X_val):,} | Test (Jan 2026): {len(X_test):,}")


# Normalize — fit ONLY on training data to prevent leakage
scaler = ps.PowerScaler(X_train, y_train) #scalar object that we can use to scale and inverse scale our data

# Scale all sets (train/val/test) using the same scaler fitted on training data
#normalize for same scale 
X_train_t = scaler.scale_x(X_train)
X_val_t   = scaler.scale_x(X_val)
X_test_t  = scaler.scale_x(X_test)

y_train_t = scaler.scale_y(y_train).view(-1, 1)
y_val_t   = scaler.scale_y(y_val).view(-1, 1)
y_test_t  = scaler.scale_y(y_test).view(-1, 1) # Implement differencing


# Graph: EE(0) <-> FI(1), EE(0) <-> LV(2), LV(2) <-> LT(3)
edge_index = torch.tensor([
    [0, 1, 0, 2, 2, 3], # source nodes
    [1, 0, 2, 0, 3, 2], # target nodes (bidirectional edges)
], dtype=torch.long)


device = torch.device(
    "cuda" if torch.cuda.is_available() else
    ("mps"  if torch.backends.mps.is_available() else "cpu")
)
model      = STGNN(NUM_FEATURES, hidden_dim=64).to(device)  # FIX: was 32, increased for better capacity
edge_index = edge_index.to(device)
x_mean       = scaler.x_mean.to(device)
x_std        = scaler.x_std.to(device)

print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,} on {device}")



# ==================================================
# STEP 4: TRAINING LOOP
# ==================================================
print("\n[4/6] Training...")


# ── Optimizer / scheduler / hyperparameters ──────────────────────────────────
optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-3)  # FIX: lr lowered 1e-3→5e-4 for more stable val loss
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

BATCH_SIZE = 64
EPOCHS     = 200   # upper bound — early stopping will terminate well before this

"""
Diagnostics after training:
  Val loss still falling at last epoch  → raise EPOCHS (shouldn't happen with patience=15)
  Val loss flat / rising after epoch X  → early stopping working as intended
  Large train/val gap throughout        → more dropout or smaller hidden_dim
"""

early_stopping = opt.EarlyStopping(patience=15, min_delta=1e-4, path="best_stgnn.pt")

# Move edge_index to device ONCE before the loop
edge_index = edge_index.to(device)

train_losses, val_losses = [], []

for epoch in range(EPOCHS):
    # ── Train ─────────────────────────────────────────────────────────────────
    model.train()
    perm = torch.randperm(len(X_train_t))
    epoch_loss = 0.0
    n_batches  = 0

    for i in range(0, len(X_train_t), BATCH_SIZE):
        idx_b = perm[i:i + BATCH_SIZE]
        xb = X_train_t[idx_b].to(device)
        yb = y_train_t[idx_b].to(device)

        optimizer.zero_grad()
        preds = model(xb, edge_index)
        loss  = helper.quantile_loss(preds, yb)

        if torch.isnan(loss):
            print(f"  [!] NaN loss at epoch {epoch+1}, batch {i}. Skipping batch.")
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        epoch_loss += loss.item()
        n_batches  += 1

    # ── Validate ──────────────────────────────────────────────────────────────
    model.eval()
    val_losses_b = []
    with torch.no_grad():
        for i in range(0, len(X_val_t), BATCH_SIZE):
            xb_v = X_val_t[i:i + BATCH_SIZE].to(device)
            yb_v = y_val_t[i:i + BATCH_SIZE].to(device)
            v_loss = helper.quantile_loss(model(xb_v, edge_index), yb_v)
            val_losses_b.append(v_loss.item())

    avg_train_loss = epoch_loss / max(n_batches, 1)
    avg_val_loss   = np.mean(val_losses_b)

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)

    scheduler.step(avg_val_loss)

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1:3d}: train={avg_train_loss:.4f} | "
              f"val={avg_val_loss:.4f} | lr={optimizer.param_groups[0]['lr']:.6f}")

    early_stopping.step(avg_val_loss, model, epoch + 1)
    if early_stopping.stop:
        print(f"\n  Early stopping triggered after epoch {epoch+1}.")
        break

# Restore the best checkpoint before evaluation
model = early_stopping.load_best(model)
print(f"  Training complete. Best val loss: {early_stopping.best_loss:.4f}")

# ==================================================
# STEP 5: EVALUATE + SCENARIOS
# ==================================================
print("\n[5/6] Evaluating on January 2026...")

model.eval()
with torch.no_grad():
    # Ensure test data is on the correct device
    X_test_device = X_test_t.to(device)
    # Get predictions and move back to CPU for numpy/plotting
    test_preds = model(X_test_device, edge_index).cpu().numpy()

print(f"  Predictions generated for {len(test_preds)} samples.")

"""
# Load wind production scenarios from counterfactual analysis
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "..", "data", "wind_production_scenarios.csv")
wind_scenarios = pd.read_csv(data_path, index_col=0, parse_dates=True)
wind_scenA = wind_scenarios["wind_mwh_scenA"].values
wind_scenB = wind_scenarios["wind_mwh_scenB"].values
"""
p10    = scaler.inverse_y(test_preds[:, 0])
p50    = scaler.inverse_y(test_preds[:, 1])
p90    = scaler.inverse_y(test_preds[:, 2])
actual = scaler.inverse_y(y_test_t.squeeze().numpy())

mae      = np.mean(np.abs(p50 - actual))
rmse     = np.sqrt(np.mean((p50 - actual) ** 2))
coverage = np.mean((actual >= p10) & (actual <= p90))

print(f"  MAE:              {mae:.1f} MW")
print(f"  RMSE:             {rmse:.1f} MW")
print(f"  P10-P90 coverage: {coverage*100:.1f}%  (target ≥ 80%)") # We want our 80% prediction interval to actually cover the true value at least 80% of the time
print(f"  Mean predicted supply (P50): {p50.mean():.1f} MW")
print(f"  Min predicted supply (P10):  {p10.min():.1f} MW")
print(f"  Max predicted supply (P90):  {p90.max():.1f} MW")

# --------------------------------------------------
# Scenario engine
# --------------------------------------------------
def run_scenario(name, isolate=False, wind_series=None, extra_wind_mw=0):
    """
    isolate:        remove ALL cross-border edges and zero import flows
    wind_series:    np.array (T,) of simulated wind production in MW
                    → plug in real OpenWeather-derived series
    """
    edges = edge_index.clone()
    x_mod = X_test_t.clone().cpu()   # work on CPU for modification

    if isolate:
        # Remove all edges — Estonia fully island-mode
        edges = torch.zeros((2, 0), dtype=torch.long)

       
    # S2 scenario: Estonia fully isolated — no cross-border connections
    # Step 1: save original flow values before zeroing (needed to update supply)
        fi_flow_orig = x_mod[:, :, 0, FLOW_FI_IDX].clone()
        lv_flow_orig = x_mod[:, :, 0, FLOW_LV_IDX].clone()

        x_mod[:, :, 0, FLOW_FI_IDX] = 0.0
        x_mod[:, :, 1, FLOW_FI_IDX] = 0.0
        x_mod[:, :, 0, FLOW_LV_IDX] = 0.0
        x_mod[:, :, 2, FLOW_LV_IDX] = 0.0

        # Brug de gemte værdier til at opdatere available_energy
        x_mod[:, :, 0, SUPPLY_IDX] += fi_flow_orig
        x_mod[:, :, 0, SUPPLY_IDX] += lv_flow_orig

        # --- Wind production ---
    # Option A: real simulated series from OpenWeather (use when ready)
    if wind_series is not None:
        # wind_series shape: (n_test_hours,) in MW
        # Normalize using training std for renewable feature
        renew_std = x_std[0, 0, 0, RENEW_IDX].item()
        prod_std = x_std[0, 0, 0, PROD_IDX].item()
        supply_std = x_std[0, 0, 0, SUPPLY_IDX].item()
        wind_mw_std = x_std[0, 0, 0, WIND_MW_IDX].item()

        for t in range(len(x_mod)):
            w = wind_series[t : t + SEQ_LEN]
            if len(w) == SEQ_LEN:
                wt = torch.from_numpy(w.astype(np.float32))
                x_mod[t, :, 0, RENEW_IDX] += wt / renew_std
                x_mod[t, :, 0, PROD_IDX] += wt / prod_std
                x_mod[t, :, 0, SUPPLY_IDX] += wt / supply_std
                x_mod[t, :, 0, WIND_MW_IDX] += wt / wind_mw_std

    elif extra_wind_mw > 0:
        renew_std = x_std[0, 0, 0, RENEW_IDX].item()
        prod_std = x_std[0, 0, 0, PROD_IDX].item()
        supply_std = x_std[0, 0, 0, SUPPLY_IDX].item()
        wind_mw_std = x_std[0, 0, 0, WIND_MW_IDX].item()
        w = float(extra_wind_mw)
        x_mod[:, :, 0, RENEW_IDX] += w / renew_std
        x_mod[:, :, 0, PROD_IDX] += w / prod_std
        x_mod[:, :, 0, SUPPLY_IDX] += w / supply_std
        x_mod[:, :, 0, WIND_MW_IDX] += w / wind_mw_std
    with torch.no_grad():
        preds = model(x_mod.to(device), edges.to(device)).cpu().numpy()

    p10_s = scaler.inverse_y(preds[:, 0])
    p50_s = scaler.inverse_y(preds[:, 1])
    p90_s = scaler.inverse_y(preds[:, 2])


       
    print(f"\n  {name}")
    print(f"    Mean available supply (P50): {p50_s.mean():+.0f} MW")
    print(f"    Min available supply (P10):  {p10_s.min():+.0f} MW")
    print(f"    Max available supply (P90):  {p90_s.max():+.0f} MW")
    print(f"    Supply reduction vs S1:      {p50_s1.mean() - p50_s.mean():+.0f} MW"
        if name != "S1: Full grid — all connections intact" else "")
    
current_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(current_dir, "..", "data", "wind_production_scenarios.csv")
wind_scenarios = pd.read_csv(data_path, index_col=0, parse_dates=True)
"""
scenarios = pd.read_csv(
    "../data/wind_production_scenarios.csv", index_col=0, parse_dates=True
)
"""
baseline = wind_scenarios["wind_mwh_baseline"].values
wind_scenA = (
    wind_scenarios["wind_mwh_scenA"] - wind_scenarios["wind_mwh_baseline"]
).values  # +323 MW new
wind_scenB = (
    wind_scenarios["wind_mwh_scenB"] - wind_scenarios["wind_mwh_baseline"]
).values  # +887 MW new (323+564)

# --- Run scenarios ---
print("\n  --- Scenario comparison (January 2026) ---")

p50_s1, p10_s1, p90_s1 = run_scenario(
    "S1: Full grid — all connections intact")

p50_s2, p10_s2, p90_s2 = run_scenario(
    "S2: Full isolation — no cross-border connections",
    isolate=True)

p50_s3, p10_s3, p90_s3 = run_scenario(
    "S3: Isolated + Scenario A (established plans)",
    isolate=True,
    wind_series=wind_scenA)

p50_s4, p10_s4, p90_s4 = run_scenario(
    "S4: Isolated + Scenario B (pipeline farms)",
    isolate=True,
    wind_series=wind_scenB)

'''
# Additional wind capacity sensitivity analysis (isolated scenario) — varies with flat MW for reference
print("\n  --- Supply level by flat wind addition (isolated) — for reference ---")
for mw in [250, 500, 750, 1000]:
    p50_w, p10_w, _ = run_scenario(
        f"Isolated + {mw} MW wind (flat)",
        isolate=True,
        extra_wind_mw=mw
    )
    print(f"    {mw:5d} MW wind → mean supply: {p50_w.mean():+.0f} MW  "
          f"| min supply (P10): {p10_w.min():+.0f} MW")
          '''
# ==================================================
# STEP 6: VISUALIZE
# ==================================================
print("\n[6/6] Visualizing...")

jan_hours = pd.date_range("2026-01-01", periods=len(p50_s1), freq="h", tz="UTC").values

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle("ST-GNN: Estonian Energy Resilience — January 2026",
             fontsize=14, fontweight="bold")

# Plot 1: training curves
axes[0, 0].plot(train_losses, label="Train", lw=2, color="steelblue")
axes[0, 0].plot(val_losses,   label="Val",   lw=2, color="orange")
axes[0, 0].set_title("Training & Validation Loss")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Quantile loss")
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: forecast vs actual
axes[0, 1].axhline(0, color="black", lw=1, linestyle="--", label="0 MW (break-even)")
axes[0, 1].fill_between(jan_hours, p10, p90, alpha=0.15,
                         color="steelblue", label="P10–P90 band")
axes[0, 1].plot(jan_hours, p50,    lw=2,   color="steelblue", label="P50 forecast")
axes[0, 1].plot(jan_hours, actual, lw=1.5, color="red",
                linestyle="--", label="Actual balance")
axes[0, 1].fill_between(jan_hours, p50, 0,
    where=(p50 < 0), alpha=0.25, color="red", label="Predicted supply")
axes[0, 1].set_title(f"S1 Forecast vs Actual  (MAE = {mae:.0f} MW)")
axes[0, 1].set_ylabel("Energy balance (MW)")
axes[0, 1].set_xlim(jan_hours.min(), jan_hours.max())
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(alpha=0.3)
axes[0, 1].tick_params(axis="x", rotation=30)

# Plot 3: scenario comparison over time
# Plot 3: scenario supply comparison over time
axes[1, 0].plot(jan_hours, p50_s1, lw=2,   color="green",  label="Baseline")
axes[1, 0].plot(jan_hours, p50_s2, lw=2,   color="red",    label="Isolated (no wind)")
axes[1, 0].plot(jan_hours, p50_s3, lw=1.5, color="orange", label="Scenario A (established plans)")
axes[1, 0].plot(jan_hours, p50_s4, lw=1.5, color="gold",   label="Scenario B (pipeline farms)")

# Show the gap between S1 and S2 — this is the import dependency
axes[1, 0].fill_between(jan_hours, p50_s1, p50_s2,
    alpha=0.15, color="red", label="Import dependency gap (S1-S2)")

axes[1, 0].set_title("Available Energy Supply by Scenario — January 2026")
axes[1, 0].set_ylabel("Available Supply (MW)")
axes[1, 0].set_xlim(jan_hours.min(), jan_hours.max())
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(alpha=0.3)
axes[1, 0].tick_params(axis="x", rotation=30)
# Plot 4: mean available supply by scenario
scenario_names = ["Baseline", "Isolated\n(no wind)",
                  "Scenario A\n(established)", "Scenario B\n(pipeline)"]
mean_supply = [p50_s1.mean(), p50_s2.mean(),
               p50_s3.mean(), p50_s4.mean()]
min_supply  = [p10_s1.min(), p10_s2.min(),
               p10_s3.min(), p10_s4.min()]
colors = ["green", "red", "orange", "gold"]

x = np.arange(4)
bars = axes[1, 1].bar(x, mean_supply, 0.5, color=colors, alpha=0.8,
                      label="Mean available supply (P50)")
axes[1, 1].scatter(x, min_supply, color="black", zorder=5,
                   marker="v", s=80, label="Worst hour (P10)")

# Add value labels on top of bars
for bar, val in zip(bars, mean_supply):
    axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
                    f"{val:.0f}", ha="center", va="bottom", fontsize=8)

# Add typical Estonian consumption line for reference
axes[1, 1].axhline(900, color="black", lw=1.5, linestyle="--",
                   label="Typical consumption (~900 MW)")

axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(scenario_names, fontsize=9)
axes[1, 1].set_ylabel("Available Supply (MW)")
axes[1, 1].set_title("Mean Available Supply by Scenario\n")
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("stgnn_resilience.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n✓ Complete! Saved to stgnn_resilience.png")
print("\n✓ S3-S4 now use realistic wind production scenarios from ursula_wind_counterfactual.ipynb")
print("  S3: Established plans (Lääneranna, Pärnu+Tori, Aidu) = +323 MW")
print("  S4: Pipeline farms (Lääneranna, Tori, Lääne-Nigula, Põhja-Pärnumaa, Lüganuse) = +887 MW")
