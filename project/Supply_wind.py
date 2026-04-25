from datetime import datetime
import os
import pickle
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
import GNN_optimizer as opt

BASE = "https://dashboard.elering.ee/api"
START = "2019-01-01T00:00:00.000Z"
END = "2026-02-01T00:00:00.000Z"

CACHE_FILE = "data_cache.pkl"

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

df_daily = pd.concat(
    [
        df_prices_hourly.add_prefix("price_"),
        df_flows_hourly.add_prefix("flow_"),
        df_system_hourly.add_prefix("system_"),
        df_weather_hourly.add_prefix("weather_"),
    ],
    axis=1,
).sort_index()

flow_cols = [col for col in df_daily.columns if col.startswith("flow_")]
df_daily[flow_cols] = df_daily[flow_cols].fillna(0)
df_daily = df_daily.dropna(how="all")

prices_h = df_prices_hourly.copy()
flows_h = df_flows_hourly.copy()
system_h = df_system_hourly.copy()
weather_h = df_weather_hourly.copy()

idx = prices_h.index
flows_h = flows_h.reindex(idx, method="ffill")
system_h = system_h.reindex(idx, method="ffill")
weather_h = df_weather_hourly.reindex(idx, method="ffill")

# True energy balance: production minus all net exports
# positive = surplus, negative = deficit even after imports
system_h["available_energy"] = (
    system_h["production"]
    - flows_h[("ee", "fi")]
    - flows_h[("ee", "lv")]
    - flows_h[("ee", "ru_narva")]
    - flows_h[("ee", "ru_pihkva")]
)
print(f"    Hours without enough energy: {(system_h['available_energy'] < 0).sum()}")

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

# Defined early because it is used for the available_energy lag below
HORIZON = 24

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
entsoe = pd.read_csv(
    "../data/entsoe_production_hourly.csv", index_col=0, parse_dates=True
)
wind_mw = entsoe["wind_onshore"].reindex(idx, method="ffill").fillna(0)
print(f"    wind_mw NaN count after reindex: {wind_mw.isna().sum()}")

# EE node: full feature set
# available_energy is lagged 24 h so the model can't trivially copy the most
# recent balance value (which would overlap with the prediction window).
ee_feats = pd.DataFrame(
    {
        "available_energy_lag24": system_h["available_energy"].shift(HORIZON),
        "production_renewable": system_h["production_renewable"],
        "production": system_h["production"],
        "wind_mw": wind_mw,  # ENTSOE actual wind; already in production
        "flow_fi": flows_h[("ee", "fi")],
        "flow_lv": flows_h[("ee", "lv")],
        "price": prices_h["ee"],
        "temperature": weather_h["temperature"],
        "wind_speed_10m": weather_h["wind_speed_10m"],
        "freq_deviation": freq_deviation,
        "hour_sin": hour_sin,
        "hour_cos": hour_cos,
        "dow_sin": dow_sin,
        "dow_cos": dow_cos,
        "month_sin": month_sin,
        "month_cos": month_cos,
    },
    index=idx,
).fillna(0)

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

NUM_NODES = 4
NUM_FEATURES = node_data.shape[2]
FEATURE_NAMES = list(ee_feats.columns)

SUPPLY_IDX = FEATURE_NAMES.index("available_energy_lag24")
FLOW_FI_IDX = FEATURE_NAMES.index("flow_fi")
FLOW_LV_IDX = FEATURE_NAMES.index("flow_lv")
RENEW_IDX = FEATURE_NAMES.index("production_renewable")
PROD_IDX = FEATURE_NAMES.index("production")
WIND_MW_IDX = FEATURE_NAMES.index("wind_mw")
TEMP_IDX = FEATURE_NAMES.index("temperature")
WIND_IDX = FEATURE_NAMES.index("wind_speed_10m")

print(f"    Features ({NUM_FEATURES}): {FEATURE_NAMES}")

# ==================================================
# STEP 2: SEQUENCES + SPLITS
# ==================================================
print("\n[2/6] Creating sequences...")

SEQ_LEN = 48  # 2 days of hourly history

y_target_values = system_h["production"].values

X_list, y_list = [], []
for t in range(SEQ_LEN, len(node_data) - HORIZON):
    X_list.append(node_data[t - SEQ_LEN : t])
    y_list.append(y_target_values[t + HORIZON - 1])

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.float32)

jan2026_start = np.searchsorted(
    idx[SEQ_LEN:-HORIZON], pd.Timestamp("2026-01-01", tz="UTC")
)
X_train_full, y_train_full = X[:jan2026_start], y[:jan2026_start]
X_test, y_test = X[jan2026_start:], y[jan2026_start:]

val_split = int(len(X_train_full) * 0.8)
X_train, y_train = X_train_full[:val_split], y_train_full[:val_split]
X_val, y_val = X_train_full[val_split:], y_train_full[val_split:]

print(
    f"Train: {len(X_train):,} ({X_train.shape}) | Val: {len(X_val):,} | Test (Jan 2026): {len(X_test):,}"
)

scaler = ps.PowerScaler(X_train, y_train)

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

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else ("mps" if torch.backends.mps.is_available() else "cpu")
)
model = STGNN(NUM_FEATURES, hidden_dim=32).to(device)
edge_index = edge_index.to(device)
x_mean = scaler.x_mean.to(device)
x_std = scaler.x_std.to(device)

print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,} on {device}")

# ==================================================
# STEP 4: TRAINING LOOP
# ==================================================
print("\n[4/6] Training...")

optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=5, factor=0.5
)

BATCH_SIZE = 256
EPOCHS = 200

early_stopping = opt.EarlyStopping(patience=15, min_delta=1e-4, path="best_stgnn.pt")
edge_index = edge_index.to(device)

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
        preds = model(xb, edge_index)
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
            v_loss = helper.quantile_loss(model(xb_v, edge_index), yb_v)
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
    test_preds = model(X_test_t.to(device), edge_index).cpu().numpy()

print(f"  Predictions generated for {len(test_preds)} samples.")

p10 = scaler.inverse_y(test_preds[:, 0])
p50 = scaler.inverse_y(test_preds[:, 1])
p90 = scaler.inverse_y(test_preds[:, 2])
actual = scaler.inverse_y(y_test_t.squeeze().numpy())

mae = np.mean(np.abs(p50 - actual))
rmse = np.sqrt(np.mean((p50 - actual) ** 2))
coverage = np.mean((actual >= p10) & (actual <= p90))

print(f"  MAE:              {mae:.1f} MW")
print(f"  RMSE:             {rmse:.1f} MW")
print(f"  P10-P90 coverage: {coverage * 100:.1f}%  (target ≥ 80%)")
print(f"  Actual production range: {actual.min():.0f} – {actual.max():.0f} MW")


# --------------------------------------------------
# Scenario engine
# --------------------------------------------------
def run_scenario(name, isolate=False, wind_series=None, extra_wind_mw=0):
    """
    isolate:       remove ALL cross-border edges and zero import flows
    wind_series:   np.array (T,) of ADDITIONAL hourly wind production in MW
                   from NEW farms — not yet included in historical production.
                   Injected into production_renewable, production,
                   available_energy_lag24, and wind_mw (all consistently).
    extra_wind_mw: flat MW for break-even sweep only
    """
    edges = edge_index.clone()
    x_mod = X_test_t.clone().cpu()

    if isolate:
        edges = torch.zeros((2, 0), dtype=torch.long)

        x_mod[:, :, 0, FLOW_FI_IDX] = 0.0
        x_mod[:, :, 1, FLOW_FI_IDX] = 0.0
        x_mod[:, :, 0, FLOW_LV_IDX] = 0.0
        x_mod[:, :, 2, FLOW_LV_IDX] = 0.0

    if wind_series is not None:
        # wind_series is ADDITIONAL production from new farms (not in historical data).
        # Each of the 4 affected features gets its own std for correct scaling.
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

    deficit_h = (p50_s < 0).sum()
    severe_h = (p50_s < -100).sum()

    if name:
        print(f"\n  {name}")
        print(f"    Median production:   {p50_s.mean():+.0f} MW")
        print(f"    Min production (P10):{p10_s.min():+.0f} MW")
        print(f"    Hours prod < 0:      {deficit_h} / {len(p50_s)}")
        print(f"    Hours prod < -100:   {severe_h}")

    return p50_s, p10_s, p90_s


# Load counterfactual wind production series for January 2026.
# The CSV columns are TOTAL wind production (baseline 694 MW + new farms).
# We subtract the baseline to get only the ADDITIONAL production from new capacity,
# so we don't double-count wind that is already in the model's wind_mw feature.
scenarios = pd.read_csv(
    "../data/wind_production_scenarios.csv", index_col=0, parse_dates=True
)
baseline = scenarios["wind_mwh_baseline"].values
wind_scenA = (
    scenarios["wind_mwh_scenA"] - scenarios["wind_mwh_baseline"]
).values  # +323 MW new
wind_scenB = (
    scenarios["wind_mwh_scenB"] - scenarios["wind_mwh_baseline"]
).values  # +887 MW new (323+564)

print("\n  --- Scenario comparison (January 2026) ---")

p50_s1, p10_s1, p90_s1 = run_scenario("S1: Full grid — all connections intact")

p50_s2, p10_s2, p90_s2 = run_scenario(
    "S2: Full isolation — no cross-border connections", isolate=True
)

p50_s3, p10_s3, p90_s3 = run_scenario(
    "S3: Isolated + Established plans (+323 MW — Lääneranna, Pärnu, Aidu)",
    isolate=True,
    wind_series=wind_scenA,
)

p50_s4, p10_s4, p90_s4 = run_scenario(
    "S4: Isolated + Full pipeline (+887 MW new — established + pipeline farms)",
    isolate=True,
    wind_series=wind_scenB,
)

print("\n  --- Break-even: MW of flat wind needed for full isolation survival ---")
for mw in [250, 500, 750, 1000, 1500, 2000]:
    p50_w, _, _ = run_scenario("", isolate=True, extra_wind_mw=mw)
    pct = (p50_w < 0).sum() / len(p50_w) * 100
    print(f"    {mw:5d} MW wind → {pct:5.1f}% hours in deficit")

# ==================================================
# STEP 6: VISUALIZE
# ==================================================
print("\n[6/6] Visualizing...")

jan_hours = pd.date_range("2026-01-01", periods=len(p50_s1), freq="h", tz="UTC").values

fig, axes = plt.subplots(2, 2, figsize=(16, 10))
fig.suptitle(
    "ST-GNN: Estonian Energy Resilience — January 2026", fontsize=14, fontweight="bold"
)

# Plot 1: training curves
axes[0, 0].plot(train_losses, label="Train", lw=2, color="steelblue")
axes[0, 0].plot(val_losses, label="Val", lw=2, color="orange")
axes[0, 0].set_title("Training & Validation Loss")
axes[0, 0].set_xlabel("Epoch")
axes[0, 0].set_ylabel("Quantile loss")
axes[0, 0].legend()
axes[0, 0].grid(alpha=0.3)

# Plot 2: S1 forecast vs actual
axes[0, 1].axhline(0, color="black", lw=1, linestyle="--", label="0 MW (break-even)")
axes[0, 1].fill_between(
    jan_hours, p10, p90, alpha=0.15, color="steelblue", label="P10–P90 band"
)
axes[0, 1].plot(jan_hours, p50, lw=2, color="steelblue", label="P50 forecast")
axes[0, 1].plot(
    jan_hours, actual, lw=1.5, color="red", linestyle="--", label="Actual production"
)
axes[0, 1].set_title(f"S1 Forecast vs Actual  (MAE = {mae:.0f} MW)")
axes[0, 1].set_ylabel("Production (MW)")
axes[0, 1].set_xlim(jan_hours.min(), jan_hours.max())
axes[0, 1].legend(fontsize=8)
axes[0, 1].grid(alpha=0.3)
axes[0, 1].tick_params(axis="x", rotation=30)

# Plot 3: scenario balance over time
axes[1, 0].axhline(0, color="black", lw=1.2, linestyle="--", label="0 MW")
axes[1, 0].plot(jan_hours, p50_s1, lw=2, color="green", label="S1: Full grid")
axes[1, 0].plot(jan_hours, p50_s2, lw=2, color="red", label="S2: Full isolation")
axes[1, 0].plot(
    jan_hours, p50_s3, lw=1.5, color="orange", label="S3: Isolated + 323 MW"
)
axes[1, 0].plot(
    jan_hours, p50_s4, lw=1.5, color="gold", label="S4: Isolated + 887 MW new"
)
axes[1, 0].fill_between(
    jan_hours, p50_s2, 0, where=(p50_s2 < 0), alpha=0.12, color="red"
)
axes[1, 0].fill_between(
    jan_hours, p50_s3, 0, where=(p50_s3 < 0), alpha=0.12, color="orange"
)
axes[1, 0].set_title("Production by Scenario")
axes[1, 0].set_ylabel("Production (MW)")
axes[1, 0].set_xlim(jan_hours.min(), jan_hours.max())
axes[1, 0].legend(fontsize=8)
axes[1, 0].grid(alpha=0.3)
axes[1, 0].tick_params(axis="x", rotation=30)

# Plot 4: deficit hours bar chart
labels = ["S1\nFull grid", "S2\nIsolated", "S3\n+323 MW", "S4\n+887 MW\nnew"]
d_hours = [
    (p50_s1 < 0).sum(),
    (p50_s2 < 0).sum(),
    (p50_s3 < 0).sum(),
    (p50_s4 < 0).sum(),
]
s_hours = [
    (p50_s1 < -100).sum(),
    (p50_s2 < -100).sum(),
    (p50_s3 < -100).sum(),
    (p50_s4 < -100).sum(),
]
colors = ["green", "red", "orange", "gold"]

x = np.arange(4)
axes[1, 1].bar(
    x - 0.2, d_hours, 0.35, color=colors, alpha=0.8, label="Deficit hours (< 0 MW)"
)
axes[1, 1].bar(
    x + 0.2,
    s_hours,
    0.35,
    color=colors,
    alpha=0.4,
    hatch="//",
    label="Severe deficit (< −100 MW)",
)
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(labels, fontsize=9)
axes[1, 1].set_ylabel("Hours in January 2026")
axes[1, 1].set_title("Resilience: Deficit Hours by Scenario")
axes[1, 1].legend(fontsize=8)
axes[1, 1].grid(alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("stgnn_resilience_wind.png", dpi=150, bbox_inches="tight")
plt.show()

print("\n✓ Complete! Saved to stgnn_resilience_wind.png")
