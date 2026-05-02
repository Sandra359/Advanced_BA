"""
Total Available Energy Analysis
================================
The GNN predicts EE domestic PRODUCTION (MW) for 4 scenarios.

Key insight for the report:
  - S1 (full grid): Estonia produces less domestically because it can import.
    Total available energy = GNN production + actual net imports (from Elering).
  - S2/S3/S4 (isolated): No imports. Total available energy = production only.

This script:
  1. Loads GNN production quantiles (gnn_supply_scenarios_jan2026.csv)
  2. Loads actual Jan 2026 Elering flows from data_cache.pkl
  3. Computes net imports: net_import = -flow_fi - flow_lv
     (Elering sign convention: negative flow = EE importing)
  4. Adds net imports to S1 production → total available energy for S1
  5. Produces two side-by-side plot pairs:
       - production only (GNN output, all scenarios)
       - total available energy (production + imports for S1)
  6. Exports total_available_energy_jan2026.csv for use in surplus_deficit.py

Usage:
    python project/available_energy_analysis.py
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

_DIR     = os.path.dirname(os.path.abspath(__file__))
_DATA    = os.path.join(_DIR, "..", "data")
_FIGURES = os.path.join(_DIR, "..", "figures")
os.makedirs(_FIGURES, exist_ok=True)


# ── 1. Load GNN production scenario quantiles ────────────────────────────────
supply_csv = os.path.join(_DATA, "gnn_supply_scenarios_jan2026.csv")
supply = pd.read_csv(supply_csv, parse_dates=["timestamp"], index_col="timestamp")
if supply.index.tz is None:
    supply.index = supply.index.tz_localize("UTC")
print(f"[1/4] Loaded GNN supply scenarios: {len(supply)} hours")


# ── 2. Load actual Jan 2026 cross-border flows from Elering cache ────────────
cache_file = os.path.join(_DIR, "..", "data_cache.pkl")
with open(cache_file, "rb") as f:
    _, df_flows, _, _ = pickle.load(f)

flows_h = df_flows.resample("h").mean().fillna(0)
if flows_h.index.tz is None:
    flows_h.index = flows_h.index.tz_localize("UTC")

jan26 = (flows_h.index.year == 2026) & (flows_h.index.month == 1)
flows_jan26 = flows_h[jan26]
print(f"[2/4] Loaded Elering flows: {jan26.sum()} Jan 2026 hours from cache")


# ── 3. Compute net imports ────────────────────────────────────────────────────
# Elering sign convention: negative flow = EE is importing from that country.
# net_import = -(flow_ee→fi) - (flow_ee→lv)
# RU flows are ~0 since April 2024 synchronisation; EE is now Nordic-synchronised.
net_imports = (
    -flows_jan26[("ee", "fi")]
    - flows_jan26[("ee", "lv")]
).rename("net_imports_mw")

net_imports = net_imports.reindex(supply.index, method="nearest").fillna(0)

print(f"[3/4] Net imports Jan 2026: "
      f"mean={net_imports.mean():.0f} MW, "
      f"min={net_imports.min():.0f} MW, "
      f"max={net_imports.max():.0f} MW")
print(f"      (positive = EE importing, negative = EE exporting)")


# ── 4. Total available energy ─────────────────────────────────────────────────
# S1: GNN production + actual net imports
total_s1_p10 = supply["supply_s1_p10"].values + net_imports.values
total_s1_p50 = supply["supply_s1_p50"].values + net_imports.values
total_s1_p90 = supply["supply_s1_p90"].values + net_imports.values

# S2/S3/S4: isolated — production = total available energy
total_s2_p10 = supply["supply_s2_p10"].values
total_s2_p50 = supply["supply_s2_p50"].values
total_s2_p90 = supply["supply_s2_p90"].values

total_s3_p10 = supply["supply_s3_p10"].values
total_s3_p50 = supply["supply_s3_p50"].values
total_s3_p90 = supply["supply_s3_p90"].values

total_s4_p10 = supply["supply_s4_p10"].values
total_s4_p50 = supply["supply_s4_p50"].values
total_s4_p90 = supply["supply_s4_p90"].values

idx = supply.index.to_numpy()


# ── 5. Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(18, 12))

COLORS = {"s1": "green", "s2": "red", "s3": "orange", "s4": "gold"}
CONSUMPTION_REF = 900  # typical Jan consumption in MW

# ── Top-left: GNN production P50 per scenario (time series) ──────────────────
ax = axes[0, 0]
ax.plot(idx, supply["supply_s1_p50"].values, lw=2, color="green",
        label="S1: Full grid")
ax.plot(idx, supply["supply_s2_p50"].values, lw=2, color="red",
        label="S2: Isolated")
ax.plot(idx, supply["supply_s3_p50"].values, lw=1.5, color="orange",
        label="S3: Isolated + Scenario A (established plans)")
ax.plot(idx, supply["supply_s4_p50"].values, lw=1.5, color="gold",
        label="S4: Isolated + Scenario B (pipeline plans)")
ax.fill_between(idx,
                supply["supply_s1_p50"].values,
                supply["supply_s2_p50"].values,
                alpha=0.15, color="blue", label="Cost of isolation (S1–S2)")
ax.set_title("EE Domestic Production — January 2026\n(GNN P50 prediction)")
ax.set_ylabel("Production (MW)")
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.3)
ax.tick_params(axis="x", rotation=30)
ax.set_xlim(idx.min(), idx.max())

# ── Top-right: Total available energy (production + imports for S1) ───────────
ax = axes[0, 1]
ax.fill_between(idx, total_s1_p10, total_s1_p90,
                alpha=0.15, color="green", label="S1 P10–P90 band")
ax.plot(idx, total_s1_p50, lw=2, color="green",
        label="S1: Full grid (prod + imports)")
ax.plot(idx, total_s2_p50, lw=2, color="red",
        label="S2: Isolated (prod only)")
ax.plot(idx, total_s3_p50, lw=1.5, color="orange",
        label="S3: Isolated + Scenario A")
ax.plot(idx, total_s4_p50, lw=1.5, color="gold",
        label="S4: Isolated + Scenario B")
ax.fill_between(idx, total_s1_p50, total_s2_p50,
                alpha=0.15, color="blue", label="Import buffer (S1 vs isolated)")
ax.plot(idx, net_imports.values, lw=1, color="navy",
        linestyle=":", alpha=0.7, label="Actual net imports (Jan 2026)")
ax.set_title("Total Available Energy — January 2026\n(S1 = production + actual Elering net imports)")
ax.set_ylabel("Available energy (MW)")
ax.legend(fontsize=8, loc="upper right")
ax.grid(alpha=0.3)
ax.tick_params(axis="x", rotation=30)
ax.set_xlim(idx.min(), idx.max())

# ── Bottom-left: Bar — mean PRODUCTION per scenario ──────────────────────────
ax = axes[1, 0]
s_labels = ["S1\nFull grid", "S2\nIsolated", "S3\nScenario A", "S4\nScenario B"]
s_colors = ["green", "red", "orange", "gold"]

prod_p50 = [
    supply["supply_s1_p50"].mean(), supply["supply_s2_p50"].mean(),
    supply["supply_s3_p50"].mean(), supply["supply_s4_p50"].mean(),
]
prod_p10 = [
    supply["supply_s1_p10"].mean(), supply["supply_s2_p10"].mean(),
    supply["supply_s3_p10"].mean(), supply["supply_s4_p10"].mean(),
]
prod_p90 = [
    supply["supply_s1_p90"].mean(), supply["supply_s2_p90"].mean(),
    supply["supply_s3_p90"].mean(), supply["supply_s4_p90"].mean(),
]
err_lo = [max(0.0, p50 - p10) for p50, p10 in zip(prod_p50, prod_p10)]
err_hi = [max(0.0, p90 - p50) for p50, p90 in zip(prod_p50, prod_p90)]

x = np.arange(4)
bars = ax.bar(x, prod_p50, 0.5, color=s_colors, alpha=0.75)
ax.errorbar(x, prod_p50, yerr=[err_lo, err_hi],
            fmt="none", color="black", capsize=6, lw=1.5, label="P10–P90 range")
for bar, val in zip(bars, prod_p50):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
            f"{val:.0f}", ha="center", va="bottom", fontsize=8)
ax.axhline(CONSUMPTION_REF, color="black", lw=1.5, linestyle="--",
           label=f"Typical consumption (~{CONSUMPTION_REF} MW)")
ax.set_xticks(x)
ax.set_xticklabels(s_labels, fontsize=9)
ax.set_ylabel("Mean production (MW)")
ax.set_title("Mean EE Production by Scenario\n(error bars = P10–P90)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")

# ── Bottom-right: Bar — mean TOTAL AVAILABLE ENERGY per scenario ──────────────
ax = axes[1, 1]
total_p50 = [total_s1_p50.mean(), total_s2_p50.mean(),
             total_s3_p50.mean(), total_s4_p50.mean()]
total_p10 = [total_s1_p10.mean(), total_s2_p10.mean(),
             total_s3_p10.mean(), total_s4_p10.mean()]
total_p90 = [total_s1_p90.mean(), total_s2_p90.mean(),
             total_s3_p90.mean(), total_s4_p90.mean()]
err_lo_t = [max(0.0, p50 - p10) for p50, p10 in zip(total_p50, total_p10)]
err_hi_t = [max(0.0, p90 - p50) for p50, p90 in zip(total_p50, total_p90)]

bars = ax.bar(x, total_p50, 0.5, color=s_colors, alpha=0.75)
ax.errorbar(x, total_p50, yerr=[err_lo_t, err_hi_t],
            fmt="none", color="black", capsize=6, lw=1.5, label="P10–P90 range")
for bar, val in zip(bars, total_p50):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
            f"{val:.0f}", ha="center", va="bottom", fontsize=8)
ax.axhline(CONSUMPTION_REF, color="black", lw=1.5, linestyle="--",
           label=f"Typical consumption (~{CONSUMPTION_REF} MW)")
ax.set_xticks(x)
ax.set_xticklabels(s_labels, fontsize=9)
ax.set_ylabel("Mean available energy (MW)")
ax.set_title("Mean Total Available Energy by Scenario\n(S1 = production + actual imports)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")

plt.suptitle(
    "Estonian Energy Scenarios — January 2026\n"
    "Left: domestic production (GNN)   |   Right: total available energy (GNN + Elering flows)",
    fontsize=13, fontweight="bold",
)
plt.tight_layout()

out_fig = os.path.join(_FIGURES, "available_energy_scenarios.png")
plt.savefig(out_fig, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n[4/4] Saved figures → {out_fig}")


# ── 6. Export total available energy CSV for resilience simulator ─────────────
# Same column format as gnn_supply_scenarios_jan2026.csv so surplus_deficit.py
# can use it directly. S1 includes imports; S2/S3/S4 are production-only.
out_df = pd.DataFrame({
    "timestamp":    supply.index,
    "supply_s1_p10": total_s1_p10,
    "supply_s1_p50": total_s1_p50,
    "supply_s1_p90": total_s1_p90,
    "supply_s2_p10": total_s2_p10,
    "supply_s2_p50": total_s2_p50,
    "supply_s2_p90": total_s2_p90,
    "supply_s3_p10": total_s3_p10,
    "supply_s3_p50": total_s3_p50,
    "supply_s3_p90": total_s3_p90,
    "supply_s4_p10": total_s4_p10,
    "supply_s4_p50": total_s4_p50,
    "supply_s4_p90": total_s4_p90,
})
out_csv = os.path.join(_DATA, "total_available_energy_jan2026.csv")
out_df.to_csv(out_csv, index=False)
print(f"       Saved CSV  → {out_csv}")

print("\nMean available energy by scenario:")
labels = ["S1 (prod + imports)", "S2 (isolated)", "S3 (iso + scenA)", "S4 (iso + scenB)"]
for label, mean in zip(labels, total_p50):
    print(f"  {label}: {mean:.0f} MW")
print(f"\nFor context — mean net imports Jan 2026: {net_imports.mean():.0f} MW")
print("Run surplus_deficit.py with total_available_energy_jan2026.csv to get resilience metrics.")
