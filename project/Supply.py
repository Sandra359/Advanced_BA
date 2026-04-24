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

BASE = "https://dashboard.elering.ee/api"
# Fetching data for 2019-2026
START = "2019-01-01T00:00:00.000Z"
END   = "2026-02-01T00:00:00.000Z"

df_prices = helper.fetch_all(helper.get_nps_prices, START, END)
df_flows = helper.fetch_all(helper.get_cross_border_flows, START, END)
df_system = helper.fetch_all(helper.get_system_production, START, END)

# --- 3. standardize to hourly, then daily ---

df_prices_hourly  = df_prices.resample("h").mean() # The data is mostly per 15 minutes and some are hourly
df_flows_hourly   = df_flows.resample("h").mean().fillna(0) # The data is only hourly but resampling to be sure for alignment
df_system_hourly  = df_system.resample("h").mean() # Mostly per 5 minutes, some entries are 0:00 (maybe missing?) and 5 minutes

# --- 4. merge into one wide dataframe ---

df_daily = pd.concat([
    df_prices_hourly.add_prefix("price_"),
    df_flows_hourly.add_prefix("flow_"),
    df_system_hourly.add_prefix("system_"),
], axis=1).sort_index()



flow_cols = [col for col in df_daily.columns if col.startswith("flow_")]
df_daily[flow_cols] = df_daily[flow_cols].fillna(0)
df_daily = df_daily.dropna(how="all") # Thinking about imputing them


# Target: True available energy in Estonia = production + imports
# Scenarios: S1 = full grid, S2 = full isolation, S3 = isolated + wind (TBD)

prices_h = df_prices_hourly.copy()
flows_h  = df_flows_hourly.copy()
system_h = df_system_hourly.copy()

idx      = prices_h.index
flows_h  = flows_h.reindex(idx, method="ffill")
system_h = system_h.reindex(idx, method="ffill")

# True energy balance: production + all imports - consumption
# Positive = surplus, Negative = real deficit even after imports
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


# EE node: full feature set
ee_feats = pd.DataFrame({
    "available_energy":   system_h["available_energy"],
    "production_renewable": system_h["production_renewable"],
    "production":           system_h["production"],
    "flow_fi":              flows_h[("ee", "fi")],
    "flow_lv":              flows_h[("ee", "lv")],
    "price":                prices_h["ee"],
    "freq_deviation":       freq_deviation,
    "hour_sin":             hour_sin,
    "hour_cos":             hour_cos,
    "dow_sin":              dow_sin,
    "dow_cos":              dow_cos,
    "month_sin":            month_sin,
    "month_cos":            month_cos,
}, index=idx).fillna(0) # Might change it into imputing values later instead of filling with 0, but for now it is fine since we have no missing values in these features

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
SUPPLY_IDX = FEATURE_NAMES.index("available_energy")  # target is the true energy balance in Estonia (production + imports)
FLOW_FI_IDX   = FEATURE_NAMES.index("flow_fi")
FLOW_LV_IDX   = FEATURE_NAMES.index("flow_lv")
RENEW_IDX     = FEATURE_NAMES.index("production_renewable")
PROD_IDX      = FEATURE_NAMES.index("production")



# ==================================================
# STEP 2: SEQUENCES + SPLITS
# ==================================================
print("\n[2/6] Creating sequences...")

SEQ_LEN = 48   # 2 days of hourly history
HORIZON = 24   # predict 24h ahead

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

val_split = int(len(X_train_full) * 0.8)
X_train, y_train = X_train_full[:val_split],  y_train_full[:val_split]
X_val,   y_val   = X_train_full[val_split:],  y_train_full[val_split:]

print(f"Train: {len(X_train):,} ({X_train.shape}) | Val: {len(X_val):,} | Test (Jan 2026): {len(X_test):,}")


# Normalize — fit ONLY on training data to prevent leakage
scaler = ps.PowerScaler(X_train, y_train)

X_train_t = scaler.scale_x(X_train)
X_val_t   = scaler.scale_x(X_val)
X_test_t  = scaler.scale_x(X_test)

y_train_t = scaler.scale_y(y_train).view(-1, 1)
y_val_t   = scaler.scale_y(y_val).view(-1, 1)
y_test_t  = scaler.scale_y(y_test).view(-1, 1) # Implement differencing


# Graph: EE(0) <-> FI(1), EE(0) <-> LV(2), LV(2) <-> LT(3)
edge_index = torch.tensor([
    [0, 1, 0, 2, 2, 3],
    [1, 0, 2, 0, 3, 2],
], dtype=torch.long)


device = torch.device(
    "cuda" if torch.cuda.is_available() else
    ("mps"  if torch.backends.mps.is_available() else "cpu")
)
model      = STGNN(NUM_FEATURES, hidden_dim=32).to(device)
edge_index = edge_index.to(device)
x_mean       = scaler.x_mean.to(device)
x_std        = scaler.x_std.to(device)

print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,} on {device}")



# ==================================================
# STEP 4: TRAINING LOOP (IMPROVED)
# ==================================================
print("\n[4/6] Training...")

# Setup optimizer and scheduler
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

BATCH_SIZE = 64
EPOCHS     = 50

# Move edge_index to device ONCE before training
edge_index = edge_index.to(device)

train_losses, val_losses = [], []
best_val_loss = float("inf")
# Initialize best_state with current weights as a fallback
best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

for epoch in range(EPOCHS):
    model.train()
    perm = torch.randperm(len(X_train_t))
    epoch_loss = 0.0
    n_batches = 0

    for i in range(0, len(X_train_t), BATCH_SIZE):
        idx_b = perm[i:i + BATCH_SIZE]
        xb = X_train_t[idx_b].to(device)
        yb = y_train_t[idx_b].to(device)

        optimizer.zero_grad()
        
        # Forward pass
        preds = model(xb, edge_index)
        loss = helper.quantile_loss(preds, yb)
        
        # Check for NaN loss (prevents model corruption)
        if torch.isnan(loss):
            print(f"  [!] NaN loss detected at epoch {epoch+1}, batch {i}. Check your inputs!")
            continue

        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        epoch_loss += loss.item()
        n_batches += 1

    # Validation Phase
    model.eval()
    val_losses_b = []
    with torch.no_grad():
        for i in range(0, len(X_val_t), BATCH_SIZE):
            xb_v = X_val_t[i:i + BATCH_SIZE].to(device)
            yb_v = y_val_t[i:i + BATCH_SIZE].to(device)
            
            v_preds = model(xb_v, edge_index)
            v_loss = helper.quantile_loss(v_preds, yb_v)
            val_losses_b.append(v_loss.item())
    
    avg_train_loss = epoch_loss / max(n_batches, 1)
    avg_val_loss = np.mean(val_losses_b)

    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    
    # Adjust learning rate based on validation performance
    scheduler.step(avg_val_loss)

    # Save best model
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"  Epoch {epoch+1:3d}: train_loss={avg_train_loss:.4f} | "
              f"val_loss={avg_val_loss:.4f} | lr={optimizer.param_groups[0]['lr']:.6f}")

# Load the best performing weights back into the model
model.load_state_dict(best_state)
print(f"  Training complete. Best Val Loss: {best_val_loss:.4f}")

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

p10    = scaler.inverse_y(test_preds[:, 0])
p50    = scaler.inverse_y(test_preds[:, 1])
p90    = scaler.inverse_y(test_preds[:, 2])
actual = scaler.inverse_y(y_test_t.squeeze().numpy())

mae      = np.mean(np.abs(p50 - actual))
rmse     = np.sqrt(np.mean((p50 - actual) ** 2))
coverage = np.mean((actual >= p10) & (actual <= p90))

print(f"  MAE:              {mae:.1f} MW")
print(f"  RMSE:             {rmse:.1f} MW")
print(f"  P10-P90 coverage: {coverage*100:.1f}%  (target ≥ 80%)")
print(f"  Actual deficit hours (Jan 2026): {(actual < 0).sum()}")

# --------------------------------------------------
# Scenario engine
# --------------------------------------------------
def run_scenario(name, isolate=False, wind_series=None, extra_wind_mw=0):
    """
    isolate:        remove ALL cross-border edges and zero import flows
    wind_series:    np.array (T,) of simulated wind production in MW
                    → plug in real OpenWeather-derived series when ready
    extra_wind_mw:  flat MW addition (used until wind_series is available)
    """
    edges = edge_index.clone()
    x_mod = X_test_t.clone().cpu()   # work on CPU for modification

    if isolate:
        # Remove all edges — Estonia fully island-mode
        edges = torch.zeros((2, 0), dtype=torch.long)

        # Zero out import flows in features (Estonia can't import)
        fi_flow_orig = x_mod[:, :, 0, FLOW_FI_IDX].clone()
        lv_flow_orig = x_mod[:, :, 0, FLOW_LV_IDX].clone()

        x_mod[:, :, 0, FLOW_FI_IDX] = 0.0
        x_mod[:, :, 1, FLOW_FI_IDX] = 0.0
        x_mod[:, :, 0, FLOW_LV_IDX] = 0.0
        x_mod[:, :, 2, FLOW_LV_IDX] = 0.0


    # --- Wind production ---
    # Option A: real simulated series from OpenWeather (use when ready)
    if wind_series is not None:
        # wind_series shape: (n_test_hours,) in MW
        # Normalize using training std for renewable feature
        wind_std = x_std [0, 0, 0, RENEW_IDX].item()
        prod_std = x_std[0, 0, 0, PROD_IDX].item()

        for t in range(len(x_mod)):
            # The sequence window for sample t covers hours t to t+SEQ_LEN
            # Add simulated wind into each hour of the window
            wind_window = wind_series[t:t + SEQ_LEN]
            if len(wind_window) == SEQ_LEN:
                wind_norm = torch.from_numpy(
                    wind_window.astype(np.float32) / wind_std
                )
                x_mod[t, :, 0, RENEW_IDX]    += wind_norm
                x_mod[t, :, 0, PROD_IDX]     += wind_norm * (wind_std / prod_std)
                x_mod[t, :, 0, SUPPLY_IDX]  += wind_norm

    # Option B: flat MW addition (placeholder until wind_series is ready)
    elif extra_wind_mw > 0:
        wind_std = x_std [0, 0, 0, RENEW_IDX].item()
        prod_std = x_std[0, 0, 0, PROD_IDX].item()
        wind_norm = extra_wind_mw / wind_std
        prod_norm = extra_wind_mw / prod_std
        x_mod[:, :, 0, RENEW_IDX]   += wind_norm
        x_mod[:, :, 0, PROD_IDX]    += prod_norm
        x_mod[:, :, 0, SUPPLY_IDX] += wind_norm

    with torch.no_grad():
        preds = model(x_mod.to(device), edges.to(device)).cpu().numpy()

    p10_s = scaler.inverse_y(preds[:, 0])
    p50_s = scaler.inverse_y(preds[:, 1])
    p90_s = scaler.inverse_y(preds[:, 2])

    deficit_h = (p50_s < 0).sum()
    severe_h  = (p50_s < -100).sum()

    print(f"\n  {name}")
    print(f"    Median balance:      {p50_s.mean():+.0f} MW")
    print(f"    Hours in deficit:    {deficit_h} / {len(p50_s)}")
    print(f"    Severe deficit hrs:  {severe_h}  (< -100 MW)")
    print(f"    Worst hour (P10):    {p10_s.min():+.0f} MW")
    return p50_s, p10_s, p90_s

# --- Run scenarios ---
print("\n  --- Scenario comparison (January 2026) ---")

p50_s1, p10_s1, p90_s1 = run_scenario(
    "S1: Full grid — all connections intact")

p50_s2, p10_s2, p90_s2 = run_scenario(
    "S2: Full isolation — no cross-border connections",
    isolate=True)

# S3: placeholder — replace extra_wind_mw with wind_series when ready
# -----------------------------------------------------------------------
# When OpenWeather wind simulation is ready, call like this instead:
#   p50_s3, p10_s3, p90_s3 = run_scenario(
#       "S3: Isolated + simulated wind farms",
#       isolate=True,
#       wind_series=your_simulated_wind_array)  # shape (n_test_hours,)
# -----------------------------------------------------------------------
p50_s3, p10_s3, p90_s3 = run_scenario(
    "S3: Isolated + 500 MW wind [PLACEHOLDER — replace with simulated series]",
    isolate=True,
    extra_wind_mw=500)

p50_s4, p10_s4, p90_s4 = run_scenario(
    "S4: Isolated + 1000 MW wind [PLACEHOLDER]",
    isolate=True,
    extra_wind_mw=1000)

p50_s5, p10_s5, p90_s5 = run_scenario(
    "S5: Isolated + 2000 MW wind [PLACEHOLDER]",
    isolate=True,
    extra_wind_mw=2000)

# Break-even analysis: how much wind does Estonia need?
print("\n  --- Break-even: MW of wind needed for full isolation survival ---")
for mw in [250, 500, 750, 1000, 1500, 2000]:
    p50_w, _, _ = run_scenario(f"", isolate=True, extra_wind_mw=mw)
    pct = (p50_w < 0).sum() / len(p50_w) * 100
    print(f"    {mw:5d} MW wind → {pct:5.1f}% hours in deficit")

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
    where=(p50 < 0), alpha=0.25, color="red", label="Predicted deficit")
axes[0, 1].set_title(f"S1 Forecast vs Actual  (MAE = {mae:.0f} MW)")
axes[0, 1].set_ylabel("Energy balance (MW)")
axes[0, 1].set_xlim(jan_hours.min(), jan_hours.max())
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(alpha=0.3)
axes[0, 1].tick_params(axis="x", rotation=30)

# Plot 3: scenario comparison over time
axes[1, 0].axhline(0, color="black", lw=1.2, linestyle="--", label="0 MW")
axes[1, 0].plot(jan_hours, p50_s1, lw=2,   color="green",  label="S1: Full grid")
axes[1, 0].plot(jan_hours, p50_s2, lw=2,   color="red",    label="S2: Full isolation")
axes[1, 0].plot(jan_hours, p50_s3, lw=1.5, color="orange", label="S3: Isolated + 500 MW wind")
axes[1, 0].plot(jan_hours, p50_s4, lw=1.5, color="gold",   label="S4: Isolated + 1000 MW wind")
axes[1, 0].fill_between(jan_hours, p50_s2, 0,
    where=(p50_s2 < 0), alpha=0.12, color="red")
axes[1, 0].fill_between(jan_hours, p50_s3, 0,
    where=(p50_s3 < 0), alpha=0.12, color="orange")
axes[1, 0].set_title("Energy Balance by Scenario")
axes[1, 0].set_ylabel("Balance (MW)  [+ surplus, − deficit]")
axes[1, 0].set_xlim(jan_hours.min(), jan_hours.max())
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(alpha=0.3)
axes[1, 0].tick_params(axis="x", rotation=30)

# Plot 4: deficit hours bar chart
labels  = ["S1\nFull grid", "S2\nIsolated",
           "S3\n+500 MW", "S4\n+1000 MW", "S5\n+2000 MW"]
d_hours = [(p50_s1 < 0).sum(), (p50_s2 < 0).sum(),
           (p50_s3 < 0).sum(), (p50_s4 < 0).sum(), (p50_s5 < 0).sum()]
s_hours = [(p50_s1 < -100).sum(), (p50_s2 < -100).sum(),
           (p50_s3 < -100).sum(), (p50_s4 < -100).sum(), (p50_s5 < -100).sum()]
colors  = ["green", "red", "orange", "gold", "limegreen"]

x = np.arange(5)
axes[1, 1].bar(x - 0.2, d_hours, 0.35, color=colors, alpha=0.8,
               label="Deficit hours (< 0 MW)")
axes[1, 1].bar(x + 0.2, s_hours, 0.35, color=colors, alpha=0.4,
               hatch="//", label="Severe deficit (< -100 MW)")
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(labels, fontsize=9)
axes[1, 1].set_ylabel("Hours in January 2026")
axes[1, 1].set_title("Resilience: Deficit Hours by Scenario\n"
                      "[S3–S5 are placeholders until wind simulation is ready]")
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("stgnn_resilience.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n✓ Complete! Saved to stgnn_resilience.png")
print("\nNOTE: S3-S5 use flat MW placeholders.")
print("Replace extra_wind_mw with wind_series= when OpenWeather")
print("simulation is ready.")