"""
Resilience Analysis: Merge GNN Supply Scenarios with SARIMAX Monte Carlo Demand
================================================================================
Supply: GNN quantile forecasts (P10/P50/P90) for 4 scenarios (S1-S4)
Demand: SARIMAX Monte Carlo ensemble saved as CSV

Usage:
    from sur_def import ResilienceAnalysis

    ra = ResilienceAnalysis(
        supply_csv    = "../data/gnn_supply_scenarios_jan2026.csv",
        demand_mc_csv = "../data/demand_mc_full_jan2026.csv",
        figures_dir   = "../figures",
        data_dir      = "../data",
    )
    ra.load()
    ra.compute()
    ra.plot()           # shows and saves all four figures
    ra.export_summary()

    # access results apltfterwards:
    ra.results["s1"]["deficit_risk"]
    ra.summary
"""

import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from helper_functions_GNN import fetch_all, get_system_production

warnings.filterwarnings("ignore")

SCENARIO_LABELS = {
    "s1": "S1: Full grid",
    "s2": "S2: Isolated",
    "s3": "S3: Isolated + Scenario A (established wind plans)",
    "s4": "S4: Isolated + Scenario B (pipeline wind farms)",
}
SCENARIO_COLORS = {
    "s1": "#2ca02c",
    "s2": "#d62728",
    "s3": "#ff7f0e",
    "s4": "#bcbd22",
}
DEFAULT_SCENARIOS = ["s1", "s2", "s3", "s4"]


class ResilienceAnalysis:
    """
    End-to-end resilience analysis merging GNN supply scenarios
    with SARIMAX Monte Carlo demand simulations.

    Parameters
    ----------
    supply_csv    : path to gnn_supply_scenarios_jan2026.csv
    demand_mc_csv : path to demand_mc_full_jan2026.csv
    figures_dir   : directory where figures are saved  (default: ../figures)
    data_dir      : directory where summary CSV is saved (default: ../data)
    scenarios     : list of scenario keys, default ["s1","s2","s3","s4"]
    """

    def __init__(
        self,
        supply_csv: str,
        demand_mc_csv: str,
        figures_dir: str = None,
        data_dir: str    = None,
        scenarios: list  = None,
    ):
        self.supply_csv    = supply_csv
        self.demand_mc_csv = demand_mc_csv
        self.scenarios     = scenarios or DEFAULT_SCENARIOS

        _here = os.path.dirname(os.path.abspath(__file__))
        self.figures_dir = figures_dir or os.path.join(_here, "..", "figures")
        self.data_dir    = data_dir    or os.path.join(_here, "..", "data")
        os.makedirs(self.figures_dir, exist_ok=True)
        os.makedirs(self.data_dir,    exist_ok=True)

        self.supply        = None
        self.demand        = None
        self.actual_demand = None
        self.results       = None
        self.summary       = None

    # --------------------------------------------------
    # PUBLIC API
    # --------------------------------------------------

    def run(self):
        """Convenience: load → compute → plot → export in one call."""
        self.load()
        self.compute()
        self.plot()
        self.export_summary()
        return self

    def load(self):
        self.supply, self.demand = self._load_and_align()
        df_system = fetch_all(get_system_production, start="2026-01-01", end="2026-02-01")
        self.actual_demand = df_system["consumption"]
        return self

    def compute(self):
        self._check_loaded()
        self.results = self._compute_surplus_and_deficit()
        return self

    def plot(self):
        self._check_computed()
        self._plot_supply_vs_demand()
        self._plot_mc_surplus_fan()
        self._plot_deficit_risk_and_surplus()
        self._plot_summary_bars()
        return self

    def export_summary(self):
        self._check_computed()
        self.summary = pd.DataFrame({
            sc: {
                "mean_deficit_risk_%":   self.results[sc]["deficit_risk"].mean() * 100,
                "max_deficit_risk_%":    self.results[sc]["deficit_risk"].max()  * 100,
                "pct_hours_deficit>50%": np.mean(self.results[sc]["deficit_risk"] > 0.5) * 100,
                "mean_surplus_MW":       self.results[sc]["surplus_p50"].mean(),
                "min_surplus_MW":        self.results[sc]["surplus_p50"].min(),
                "mean_shortfall_MW":     self.results[sc]["expected_shortfall"].mean(),
            }
            for sc in self.scenarios
        }).T
        out = os.path.join(self.data_dir, "resilience_summary_table.csv")
        self.summary.to_csv(out)
        return self

    # --------------------------------------------------
    # INTERNAL: LOAD
    # --------------------------------------------------

    def _load_and_align(self):
        supply    = pd.read_csv(self.supply_csv, index_col=0, parse_dates=True)
        demand_mc = pd.read_csv(self.demand_mc_csv, index_col=0)
        demand_mc.index = pd.date_range(
            start="2025-12-31 23:00:00",
            periods=len(demand_mc),
            freq="h",
            tz="UTC"
        )

        if demand_mc.shape[0] < demand_mc.shape[1]:
            demand_mc = demand_mc.T
            demand_mc.index = pd.to_datetime(demand_mc.index)

        if supply.index.tz is None:
            supply.index = supply.index.tz_localize("UTC")
        if demand_mc.index.tz is None:
            demand_mc.index = demand_mc.index.tz_localize("UTC")

        common_idx = supply.index.intersection(demand_mc.index)
        if len(common_idx) == 0:
            raise ValueError(
                "No overlapping timestamps!\n"
                f"  Supply:  {supply.index[0]} -> {supply.index[-1]}\n"
                f"  Demand:  {demand_mc.index[0]} -> {demand_mc.index[-1]}"
            )

        return supply.loc[common_idx], demand_mc.loc[common_idx]

    # --------------------------------------------------
    # INTERNAL: COMPUTE
    # --------------------------------------------------

    def _compute_surplus_and_deficit(self):
        demand_arr  = self.demand.values.astype(float)
        demand_mean = demand_arr.mean(axis=1)
        results = {}

        for sc in self.scenarios:
            p10 = self.supply[f"supply_{sc}_p10"].values.astype(float)
            p50 = self.supply[f"supply_{sc}_p50"].values.astype(float)
            p90 = self.supply[f"supply_{sc}_p90"].values.astype(float)

            surplus_mat        = p50[:, np.newaxis] - demand_arr
            deficit_risk       = np.mean(surplus_mat < 0, axis=1)
            shortfall          = np.where(surplus_mat < 0, -surplus_mat, 0.0)
            expected_shortfall = shortfall.mean(axis=1)

            results[sc] = {
                "surplus_p50":        p50 - demand_mean,
                "surplus_p10":        p10 - demand_mean,
                "surplus_p90":        p90 - demand_mean,
                "surplus_mat":        surplus_mat,
                "deficit_risk":       deficit_risk,
                "expected_shortfall": expected_shortfall,
            }

        return results

    # --------------------------------------------------
    # INTERNAL: PLOTS
    # --------------------------------------------------

    def _figpath(self, name):
        return os.path.join(self.figures_dir, name)

    def _plot_supply_vs_demand(self):
        idx        = self.supply.index
        demand_arr = self.demand.values.astype(float)

        fig, axes = plt.subplots(len(self.scenarios), 1,
                                 figsize=(16, 4 * len(self.scenarios)), sharex=True)
        if len(self.scenarios) == 1:
            axes = [axes]

        for ax, sc in zip(axes, self.scenarios):
            color = SCENARIO_COLORS[sc]
            ax.fill_between(idx,
                            np.percentile(demand_arr, 5,  axis=1),
                            np.percentile(demand_arr, 95, axis=1),
                            alpha=0.15, color="gray", label="Demand P5-P95")
            ax.plot(idx, demand_arr.mean(axis=1), color="black", lw=1.5,
                    linestyle="--", label="Demand mean")
            p10 = self.supply[f"supply_{sc}_p10"].values
            p50 = self.supply[f"supply_{sc}_p50"].values
            p90 = self.supply[f"supply_{sc}_p90"].values
            ax.fill_between(idx, p10, p90, alpha=0.25, color=color, label="Supply P10-P90")
            ax.plot(idx, p50, color=color, lw=2, label="Supply P50")
            ax.set_title(SCENARIO_LABELS[sc], fontsize=11)
            ax.set_ylabel("Power (MW)")
            ax.legend(fontsize=8, loc="upper right")
            ax.grid(alpha=0.3)
            ax.tick_params(axis="x", rotation=30)

        plt.suptitle("Supply vs Demand - January 2026", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(self._figpath("resilience_supply_vs_demand.png"), dpi=150, bbox_inches="tight")
        plt.show()

    def _plot_mc_surplus_fan(self):
        idx = self.supply.index
        fig, axes = plt.subplots(len(self.scenarios), 1,
                                 figsize=(16, 4 * len(self.scenarios)), sharex=True)
        if len(self.scenarios) == 1:
            axes = [axes]

        for ax, sc in zip(axes, self.scenarios):
            color    = SCENARIO_COLORS[sc]
            surp_mat = self.results[sc]["surplus_mat"]

            s_p05 = np.percentile(surp_mat, 5,  axis=1)
            s_p25 = np.percentile(surp_mat, 25, axis=1)
            s_p50 = np.percentile(surp_mat, 50, axis=1)
            s_p75 = np.percentile(surp_mat, 75, axis=1)
            s_p95 = np.percentile(surp_mat, 95, axis=1)

            for sim_i in range(surp_mat.shape[1]):
                ax.plot(idx, surp_mat[:, sim_i], color=color, alpha=0.08, lw=0.6)

            ax.fill_between(idx, s_p05, s_p95, alpha=0.20, color=color, label="P5-P95 surplus")
            ax.fill_between(idx, s_p25, s_p75, alpha=0.35, color=color, label="P25-P75 surplus")
            ax.plot(idx, s_p50, color=color, lw=2.5, label="Median surplus")

            if self.actual_demand is not None:
                p50_vals = self.supply[f"supply_{sc}_p50"].values
                actual_sc_surplus = p50_vals - self.actual_demand.reindex(idx).values
                ax.plot(idx, actual_sc_surplus, color="black", lw=2, linestyle="--",
                        label="Supply P50 - Actual demand (Elering)", zorder=10)

            ax.axhline(0, color="red", lw=1.5, linestyle="--", alpha=0.8, label="Break-even (0 MW)")
            ax.fill_between(idx, s_p05, np.minimum(s_p05, 0),
                            where=(s_p05 < 0), color="red", alpha=0.12, label="Deficit zone (worst 5%)")
            ax.set_title(f"{SCENARIO_LABELS[sc]}  -  Supply P50 - each demand simulation", fontsize=10)
            ax.set_ylabel("Surplus (MW)")
            ax.legend(fontsize=7, loc="upper right", ncol=2)
            ax.grid(alpha=0.3)
            ax.tick_params(axis="x", rotation=30)

        plt.suptitle("MC Surplus Fan - Uncertainty from Demand Simulations\n"
                     "(each line = Supply P50 - one simulated demand path)",
                     fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(self._figpath("resilience_mc_surplus_fan.png"), dpi=150, bbox_inches="tight")
        plt.show()

    def _plot_deficit_risk_and_surplus(self):
        idx = self.supply.index
        fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

        for sc in self.scenarios:
            r     = self.results[sc]
            color = SCENARIO_COLORS[sc]
            axes[0].plot(idx, r["deficit_risk"] * 100, color=color, lw=1.5, label=SCENARIO_LABELS[sc])
            axes[1].plot(idx, r["surplus_p50"],         color=color, lw=1.5, label=SCENARIO_LABELS[sc])

        axes[0].axhline(50, color="black", lw=1, linestyle=":", label="50% threshold")
        axes[0].set_ylabel("Deficit risk (%)")
        axes[0].set_title("Hourly deficit risk (% of demand sims where Supply P50 < demand)")
        axes[0].legend(fontsize=8)
        axes[0].grid(alpha=0.3)
        axes[0].set_ylim(0, 105)

        axes[1].axhline(0, color="black", lw=1.5, linestyle="--", label="Break-even")
        axes[1].set_ylabel("Surplus P50 (MW)")
        axes[1].set_title("Surplus = Supply P50 - Mean Demand  (negative = deficit)")
        axes[1].legend(fontsize=8)
        axes[1].grid(alpha=0.3)
        axes[1].tick_params(axis="x", rotation=30)

        plt.suptitle("Resilience Analysis - January 2026", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(self._figpath("resilience_deficit_risk.png"), dpi=150, bbox_inches="tight")
        plt.show()

    def _plot_summary_bars(self):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        sc_colors    = [SCENARIO_COLORS[sc]                           for sc in self.scenarios]
        mean_risks   = [self.results[sc]["deficit_risk"].mean() * 100 for sc in self.scenarios]
        mean_surplus = [self.results[sc]["surplus_p50"].mean()        for sc in self.scenarios]

        axes[0].bar(range(len(self.scenarios)), mean_risks, color=sc_colors, alpha=0.85, edgecolor="white")
        axes[0].set_xticks(range(len(self.scenarios)))
        axes[0].set_xticklabels([sc.upper() for sc in self.scenarios])
        axes[0].set_ylabel("Mean hourly deficit risk (%)")
        axes[0].set_title("Mean Deficit Risk by Scenario")
        for i, v in enumerate(mean_risks):
            axes[0].text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=9)
        axes[0].grid(alpha=0.3, axis="y")

        axes[1].bar(range(len(self.scenarios)), mean_surplus, color=sc_colors, alpha=0.85, edgecolor="white")
        axes[1].axhline(0, color="black", lw=1.5)
        axes[1].set_xticks(range(len(self.scenarios)))
        axes[1].set_xticklabels([sc.upper() for sc in self.scenarios])
        axes[1].set_ylabel("Mean surplus (MW)")
        axes[1].set_title("Mean Surplus = Supply P50 - Mean Demand")
        for i, v in enumerate(mean_surplus):
            offset = 5 if v >= 0 else -20
            axes[1].text(i, v + offset, f"{v:.0f} MW", ha="center", fontsize=9)
        axes[1].grid(alpha=0.3, axis="y")

        plt.suptitle("Scenario Comparison Summary", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(self._figpath("resilience_summary.png"), dpi=150, bbox_inches="tight")
        plt.show()

    # --------------------------------------------------
    # GUARDS
    # --------------------------------------------------

    def _check_loaded(self):
        if self.supply is None or self.demand is None:
            raise RuntimeError("Call .load() first.")

    def _check_computed(self):
        self._check_loaded()
        if self.results is None:
            raise RuntimeError("Call .compute() first.")


# ==================================================
# STANDALONE RUN
# ==================================================

if __name__ == "__main__":
    _here = os.path.dirname(os.path.abspath(__file__))

    ra = ResilienceAnalysis(
        supply_csv    = os.path.join(_here, "..", "data", "gnn_supply_scenarios_jan2026.csv"),
        demand_mc_csv = os.path.join(_here, "..", "data", "demand_mc_full_jan2026.csv"),
        figures_dir   = os.path.join(_here, "..", "figures"),
        data_dir      = os.path.join(_here, "..", "data"),
    )
    ra.run()