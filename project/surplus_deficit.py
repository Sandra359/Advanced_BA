"""
Resilience Analysis: Merge GNN Supply Scenarios with SARIMAX Monte Carlo Demand
================================================================================
Supply: GNN quantile forecasts (P10/P50/P90) for 4 scenarios (S1-S4)
Demand: SARIMAX Monte Carlo ensemble saved as CSV

Surplus = Supply - Demand
Deficit = Supply < Demand  (i.e. surplus < 0)

Usage:
    from resilience_analysis import run_resilience_analysis
    results = run_resilience_analysis(
        demand_mc_csv="../data/sarimax_demand_mc_jan2026.csv",
        supply_csv="../data/gnn_supply_scenarios_jan2026.csv"
    )
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==================================================
# LOAD & ALIGN
# ==================================================

def load_and_align(supply_csv: str, demand_mc_csv: str):
    """
    Load GNN supply scenarios and SARIMAX demand Monte Carlo from CSV,
    then align on common timestamps.

    Parameters
    ----------
    supply_csv    : path to gnn_supply_scenarios_jan2026.csv
    demand_mc_csv : path to sarimax_demand_mc_jan2026.csv

    Returns
    -------
    supply    : DataFrame (n_hours x scenario_columns)
    demand_mc : DataFrame (n_hours x n_sims)
    """
    supply    = pd.read_csv(supply_csv,    index_col=0, parse_dates=True)
    demand_mc = pd.read_csv(demand_mc_csv, index_col=0, parse_dates=True)

    # Ensure UTC timezone on both
    if supply.index.tz is None:
        supply.index = supply.index.tz_localize("UTC")
    if demand_mc.index.tz is None:
        demand_mc.index = demand_mc.index.tz_localize("UTC")

    # Align on common timestamps
    common_idx = supply.index.intersection(demand_mc.index)
    supply    = supply.loc[common_idx]
    demand_mc = demand_mc.loc[common_idx]

    print(f"  Aligned timestamps:  {len(common_idx)} hours")
    print(f"  Supply columns:      {list(supply.columns)}")
    print(f"  Demand simulations:  {demand_mc.shape[1]}")
    return supply, demand_mc


# ==================================================
# SURPLUS & DEFICIT COMPUTATION
# ==================================================

def compute_surplus_and_deficit(supply: pd.DataFrame,
                                demand: pd.DataFrame,
                                scenarios: list = None):
    """
    For each supply scenario and each demand simulation, compute:
      - surplus_mat[t, sim] = supply_p50[t] - demand_sim[t]
      - deficit_risk[t]     = fraction of sims where supply < demand
      - expected_shortfall  = mean shortfall in MW across all sims

    Parameters
    ----------
    supply    : DataFrame with columns supply_s1_p10/p50/p90, s2, s3, s4
    demand    : DataFrame (n_hours x n_sims) Monte Carlo demand
    scenarios : list of scenario names, default ["s1","s2","s3","s4"]

    Returns
    -------
    results : dict keyed by scenario, each containing:
              surplus_p50, surplus_p10, surplus_p90,
              deficit_risk, expected_shortfall, surplus_mat
    """
    if scenarios is None:
        scenarios = ["s1", "s2", "s3", "s4"]

    demand_arr  = demand.values            # (n_hours, n_sims)
    demand_mean = demand_arr.mean(axis=1)  # (n_hours,)
    results = {}

    for sc in scenarios:
        p10 = supply[f"supply_{sc}_p10"].values  # (n_hours,)
        p50 = supply[f"supply_{sc}_p50"].values
        p90 = supply[f"supply_{sc}_p90"].values

        # Surplus matrix: supply_p50 vs all demand sims → (n_hours, n_sims)
        surplus_mat = p50[:, np.newaxis] - demand_arr

        # Fraction of sims with deficit (supply < demand)
        deficit_risk = np.mean(surplus_mat < 0, axis=1)  # (n_hours,)

        # Expected shortfall: mean MW deficit across all sims
        shortfall          = np.where(surplus_mat < 0, -surplus_mat, 0.0)
        expected_shortfall = shortfall.mean(axis=1)      # (n_hours,)

        # Surplus using supply quantiles vs mean demand
        surplus_p50 = p50 - demand_mean
        surplus_p10 = p10 - demand_mean   # worst-case supply
        surplus_p90 = p90 - demand_mean   # best-case supply

        results[sc] = {
            "surplus_p50":        surplus_p50,
            "surplus_p10":        surplus_p10,
            "surplus_p90":        surplus_p90,
            "deficit_risk":       deficit_risk,
            "expected_shortfall": expected_shortfall,
            "surplus_mat":        surplus_mat,
        }

        # Print summary
        pct_deficit_hours = 100 * np.mean(deficit_risk > 0.5)
        mean_risk         = 100 * deficit_risk.mean()
        print(f"\n  Scenario {sc.upper()}:")
        print(f"    Mean deficit risk per hour:     {mean_risk:.1f}%")
        print(f"    Hours with >50% deficit risk:   {pct_deficit_hours:.1f}%")
        print(f"    Max deficit risk (single hour): {deficit_risk.max()*100:.1f}%")
        print(f"    Mean surplus (P50 - demand):    {surplus_p50.mean():.0f} MW")
        print(f"    Min surplus:                    {surplus_p50.min():.0f} MW")

    return results


# ==================================================
# PLOTTING
# ==================================================

def plot_resilience(supply: pd.DataFrame,
                    demand: pd.DataFrame,
                    results: dict,
                    scenarios: list = None):
    """
    Three figures:
      1. Supply P10/P50/P90 vs demand distribution — one panel per scenario
      2. Deficit risk + surplus over time — all scenarios overlaid
      3. Summary bar charts — mean deficit risk and mean surplus
    """
    if scenarios is None:
        scenarios = ["s1", "s2", "s3", "s4"]

    scenario_labels = {
        "s1": "S1: Full grid",
        "s2": "S2: Isolated",
        "s3": "S3: Isolated + 323 MW wind",
        "s4": "S4: Isolated + 887 MW wind",
    }
    scenario_colors = {
        "s1": "green",
        "s2": "red",
        "s3": "orange",
        "s4": "gold",
    }

    idx         = supply.index
    demand_mean = demand.mean(axis=1).values
    demand_q05  = demand.quantile(0.05, axis=1).values
    demand_q95  = demand.quantile(0.95, axis=1).values

    # ----------------------------------------------------------------
    # FIGURE 1: Supply vs Demand per scenario
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(len(scenarios), 1,
                             figsize=(16, 4 * len(scenarios)),
                             sharex=True)
    if len(scenarios) == 1:
        axes = [axes]

    for ax, sc in zip(axes, scenarios):
        color = scenario_colors[sc]

        # Demand band
        ax.fill_between(idx, demand_q05, demand_q95,
                        alpha=0.15, color="gray", label="Demand P5–P95")
        ax.plot(idx, demand_mean, color="black", lw=1.5,
                linestyle="--", label="Demand mean")

        # Supply band
        p10 = supply[f"supply_{sc}_p10"].values
        p50 = supply[f"supply_{sc}_p50"].values
        p90 = supply[f"supply_{sc}_p90"].values
        ax.fill_between(idx, p10, p90, alpha=0.2, color=color,
                        label="Supply P10–P90")
        ax.plot(idx, p50, color=color, lw=2, label="Supply P50")

        ax.set_title(scenario_labels[sc], fontsize=11)
        ax.set_ylabel("Power (MW)")
        ax.legend(fontsize=8, loc="upper right")
        ax.grid(alpha=0.3)
        ax.tick_params(axis="x", rotation=30)

    plt.suptitle("Supply vs Demand — January 2026", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("resilience_supply_vs_demand.png", dpi=150, bbox_inches="tight")
    plt.show()

    # ----------------------------------------------------------------
    # FIGURE 2: Deficit risk + surplus over time
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    for sc in scenarios:
        r     = results[sc]
        color = scenario_colors[sc]
        axes[0].plot(idx, r["deficit_risk"] * 100,
                     color=color, lw=1.5, label=scenario_labels[sc])
        axes[1].plot(idx, r["surplus_p50"],
                     color=color, lw=1.5, label=scenario_labels[sc])

    axes[0].axhline(50, color="black", lw=1, linestyle=":",
                    label="50% deficit risk")
    axes[0].set_ylabel("Deficit risk (%)")
    axes[0].set_title("Hourly deficit risk  (% of demand sims where supply < demand)")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    axes[0].set_ylim(0, 105)

    axes[1].axhline(0, color="black", lw=1.5, linestyle="--", label="Break-even")
    axes[1].set_ylabel("Surplus P50 (MW)")
    axes[1].set_title("Surplus = Supply P50 − Mean Demand  (negative = deficit)")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    axes[1].tick_params(axis="x", rotation=30)

    plt.suptitle("Resilience Analysis — January 2026", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("resilience_deficit_risk.png", dpi=150, bbox_inches="tight")
    plt.show()

    # ----------------------------------------------------------------
    # FIGURE 3: Summary bar charts
    # ----------------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    sc_colors    = [scenario_colors[sc]             for sc in scenarios]
    mean_risks   = [results[sc]["deficit_risk"].mean() * 100 for sc in scenarios]
    mean_surplus = [results[sc]["surplus_p50"].mean()        for sc in scenarios]

    # Bar 1: mean deficit risk
    axes[0].bar(range(len(scenarios)), mean_risks, color=sc_colors, alpha=0.8)
    axes[0].set_xticks(range(len(scenarios)))
    axes[0].set_xticklabels([sc.upper() for sc in scenarios])
    axes[0].set_ylabel("Mean hourly deficit risk (%)")
    axes[0].set_title("Mean Deficit Risk by Scenario")
    for i, v in enumerate(mean_risks):
        axes[0].text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=9)
    axes[0].grid(alpha=0.3, axis="y")

    # Bar 2: mean surplus
    axes[1].bar(range(len(scenarios)), mean_surplus, color=sc_colors, alpha=0.8)
    axes[1].axhline(0, color="black", lw=1.5)
    axes[1].set_xticks(range(len(scenarios)))
    axes[1].set_xticklabels([sc.upper() for sc in scenarios])
    axes[1].set_ylabel("Mean surplus (MW)")
    axes[1].set_title("Mean Surplus = Supply P50 − Mean Demand")
    for i, v in enumerate(mean_surplus):
        offset = 5 if v >= 0 else -20
        axes[1].text(i, v + offset, f"{v:.0f} MW", ha="center", fontsize=9)
    axes[1].grid(alpha=0.3, axis="y")

    plt.suptitle("Scenario Comparison Summary", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("resilience_summary.png", dpi=150, bbox_inches="tight")
    plt.show()


# ==================================================
# MAIN ENTRY POINT
# ==================================================

def run_resilience_analysis(
    demand_mc_csv: str = "../data/sarimax_demand_mc_jan2026.csv",
    supply_csv:    str = "../data/gnn_supply_scenarios_jan2026.csv",
):
    """
    Main entry point — load both CSVs, compute surplus/deficit, plot results.

    Parameters
    ----------
    demand_mc_csv : path to sarimax_demand_mc_jan2026.csv
                    (saved from SARIMAX notebook: demand_mc_2026.to_csv(...))
    supply_csv    : path to gnn_supply_scenarios_jan2026.csv
                    (saved from GNN notebook)

    Returns
    -------
    results : dict with surplus and deficit metrics per scenario
    """
    print("=" * 60)
    print("RESILIENCE ANALYSIS: Supply vs Demand")
    print("=" * 60)

    print("\n[1/3] Loading and aligning data...")
    supply, demand = load_and_align(supply_csv, demand_mc_csv)

    print("\n[2/3] Computing surplus and deficit risk...")
    scenarios = ["s1", "s2", "s3", "s4"]
    results   = compute_surplus_and_deficit(supply, demand, scenarios)

    print("\n[3/3] Plotting...")
    plot_resilience(supply, demand, results, scenarios)

    # Export summary table
    summary = pd.DataFrame({
        sc: {
            "mean_deficit_risk_%":   results[sc]["deficit_risk"].mean() * 100,
            "max_deficit_risk_%":    results[sc]["deficit_risk"].max()  * 100,
            "pct_hours_deficit>50%": np.mean(results[sc]["deficit_risk"] > 0.5) * 100,
            "mean_surplus_MW":       results[sc]["surplus_p50"].mean(),
            "min_surplus_MW":        results[sc]["surplus_p50"].min(),
            "mean_shortfall_MW":     results[sc]["expected_shortfall"].mean(),
        }
        for sc in scenarios
    }).T

    print("\n  Summary table:")
    print(summary.round(1).to_string())
    summary.to_csv("resilience_summary_table.csv")
    print("\n  Saved resilience_summary_table.csv")

    return results


# ==================================================
# STANDALONE RUN
# ==================================================

if __name__ == "__main__":
    results = run_resilience_analysis(
        demand_mc_csv="../data/demand_mc_jan2026.csv.csv",
        supply_csv="../data/gnn_supply_scenarios_jan2026.csv",
    )