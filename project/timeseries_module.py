"""
timeseries_module.py

Module-style version of the Estonia January demand modelling pipeline.

Nothing runs on import. Use `run_pipeline(...)` to execute the full workflow,
or import individual functions into a notebook.

Main outputs:
- demand_series: hourly January demand series per year
- weather_series: hourly January weather data per year
- combined_dfs: demand + weather + engineered features per year
- all_results_df / score_summary: SARIMAX candidate comparison
- fitted_models: fitted yearly SARIMAX models for training years only
- coef_table / beta_df / beta_stats: estimated parameter tables
- demand_mc: Monte Carlo demand simulations for target year
- summary: p5 / p50 / p95 / mean / actual demand summary
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
from matplotlib import pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.statespace.sarimax import SARIMAX


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

ELERING_BASE_URL = "https://dashboard.elering.ee/api"

DEFAULT_YEARS = tuple(range(2019, 2027))
DEFAULT_TRAIN_YEARS = tuple(range(2019, 2026))
DEFAULT_TARGET_YEAR = 2026

DEFAULT_LAT = 59.437
DEFAULT_LON = 24.753

DEFAULT_FEATURES = ["temperature_roll24", "wind_speed", "weekend"]

DEFAULT_CANDIDATE_MODELS = [
    ((1, 0, 2), (1, 1, 1, 24)),
    ((2, 0, 2), (1, 1, 1, 24)),
    ((2, 0, 2), (2, 1, 1, 24)),
    ((2, 0, 2), (1, 1, 2, 24)),
]


@dataclass
class PipelineResult:
    demand_series: Dict[int, pd.Series]
    weather_series: Dict[int, pd.DataFrame]
    combined_dfs: Dict[int, pd.DataFrame]
    missing_hours: Dict[int, pd.DatetimeIndex]
    all_results_df: pd.DataFrame
    score_summary: pd.DataFrame
    best_order: Tuple[int, int, int]
    best_seasonal_order: Tuple[int, int, int, int]
    fitted_models: Dict[int, object]
    coef_table: pd.DataFrame
    beta_df: pd.DataFrame
    beta_stats: pd.DataFrame
    X_future: pd.DataFrame
    demand_mc: pd.DataFrame
    actual_target: pd.Series
    summary: pd.DataFrame


# ---------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------

def inspect_elering_system_endpoint(
    start: str = "2026-01-01T00:00:00.000Z",
    end: str = "2026-01-01T02:00:00.000Z",
    verbose: bool = True,
) -> dict:
    """
    Small API sanity check. Useful during exploration, but not needed
    for the modelling pipeline.
    """
    response = requests.get(
        f"{ELERING_BASE_URL}/system",
        params={"start": start, "end": end},
        timeout=60,
    )
    response.raise_for_status()
    data = response.json()

    if verbose:
        items = data.get("data", [])
        print("=" * 50)
        print("system (/system)")
        if isinstance(items, list) and len(items) > 0:
            first = items[0]
            print(f"Number of datapoints: {len(items)}")
            print(f"Fields: {list(first.keys()) if isinstance(first, dict) else type(first)}")
            print(f"First row: {first}")
        else:
            print(str(data)[:500])

    return data


def fetch_january_demand(year: int) -> pd.Series:
    """
    Fetch January electricity demand from the Elering dashboard API.

    The API returns 5-minute data. This function resamples it to hourly
    means and returns a Series named 'consumption'.
    """
    response = requests.get(
        f"{ELERING_BASE_URL}/system",
        params={
            "start": f"{year}-01-01T00:00:00.000Z",
            "end": f"{year}-01-31T23:59:59.000Z",
        },
        timeout=60,
    )
    response.raise_for_status()

    data = response.json()
    df = pd.DataFrame(data["data"])

    if df.empty:
        raise ValueError(f"No demand data returned for January {year}.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
    df = df.set_index("timestamp").sort_index()

    demand = df["consumption"].dropna().copy()

    # Known bad 2019 values from the original project script.
    if year == 2019:
        demand.loc[pd.Timestamp("2019-01-03 11:40:00")] = demand.loc[
            pd.Timestamp("2019-01-03 11:35:00")
        ]
        demand.loc[pd.Timestamp("2019-01-03 11:45:00")] = demand.loc[
            pd.Timestamp("2019-01-03 11:50:00")
        ]

    demand_hourly = demand.resample("h").mean().dropna()
    demand_hourly.name = "consumption"
    return demand_hourly


def fetch_all_january_demand(
    years: Iterable[int] = DEFAULT_YEARS,
    verbose: bool = True,
) -> Dict[int, pd.Series]:
    """Fetch hourly January demand for all requested years."""
    demand_series = {}

    for year in years:
        if verbose:
            print(f"Fetching demand: January {year}")
        demand_series[year] = fetch_january_demand(year)

    return demand_series


def fetch_weather_from_openmeteo(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    location_name: str = "",
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Fetch hourly temperature and wind speed from the Open-Meteo archive API.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,wind_speed_10m",
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "timezone": "UTC",
    }

    if verbose:
        print(f"Fetching weather: {location_name}")

    response = requests.get(url, params=params, timeout=60)
    response.raise_for_status()
    data = response.json()

    hourly = data["hourly"]

    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(hourly["time"]),
            "temperature": hourly["temperature_2m"],
            "wind_speed": hourly["wind_speed_10m"],
        }
    )

    return df.set_index("timestamp").sort_index()


def fetch_all_january_weather(
    years: Iterable[int] = DEFAULT_YEARS,
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
    verbose: bool = True,
) -> Dict[int, pd.DataFrame]:
    """Fetch hourly January weather for all requested years."""
    weather_series = {}

    for year in years:
        weather_series[year] = fetch_weather_from_openmeteo(
            lat=lat,
            lon=lon,
            start_date=f"{year}-01-01",
            end_date=f"{year}-01-31",
            location_name=f"Estonia January {year}",
            verbose=verbose,
        )

    return weather_series


# ---------------------------------------------------------------------
# Feature engineering and missing-value handling
# ---------------------------------------------------------------------

def build_combined_dfs(
    demand_series: Dict[int, pd.Series],
    weather_series: Dict[int, pd.DataFrame],
    years: Iterable[int] = DEFAULT_YEARS,
    temp_roll_window: int = 24,
    verbose: bool = True,
) -> Dict[int, pd.DataFrame]:
    """
    Combine demand and weather into one DataFrame per year.

    Output columns:
    - consumption
    - temperature
    - wind_speed
    - temperature_roll24
    - weekend
    """
    combined_dfs = {}

    for year in years:
        demand = demand_series[year].rename("consumption")

        weather = weather_series[year][["temperature", "wind_speed"]].copy()
        weather["temperature_roll24"] = (
            weather["temperature"]
            .rolling(temp_roll_window, min_periods=1)
            .mean()
        )

        df = pd.concat([demand, weather], axis=1).dropna()
        df["weekend"] = (df.index.dayofweek >= 5).astype(int)
        df = df.sort_index()

        combined_dfs[year] = df

        if verbose:
            print(year, df.shape, df.index.min(), df.index.max())

    return combined_dfs


def find_missing_hours(
    df: pd.DataFrame,
    year: int,
) -> pd.DatetimeIndex:
    """Find missing hourly timestamps in January for a given year."""
    complete_index = pd.date_range(
        start=pd.Timestamp(f"{year}-01-01"),
        end=pd.Timestamp(f"{year}-02-01"),
        freq="h",
        inclusive="left",
    )

    return complete_index.difference(df.index)


def find_missing_hours_by_year(
    combined_dfs: Dict[int, pd.DataFrame],
    years: Iterable[int],
    verbose: bool = True,
) -> Dict[int, pd.DatetimeIndex]:
    """Find missing January hourly timestamps for multiple years."""
    missing_hours = {}

    for year in years:
        missing = find_missing_hours(combined_dfs[year], year)
        missing_hours[year] = missing

        if verbose:
            print(f"\nJanuary {year}: {len(missing)} missing hourly timestamp(s)")
            for ts in missing:
                print(f"  {ts}")

    return missing_hours


def interpolate_single_missing_hour(
    df: pd.DataFrame,
    target_timestamp: str | pd.Timestamp,
    previous_timestamp: str | pd.Timestamp,
    next_timestamp: str | pd.Timestamp,
) -> pd.DataFrame:
    """
    Fill one missing row by averaging the previous and next known rows.
    Returns the DataFrame sorted by index.
    """
    df = df.copy()

    target_timestamp = pd.Timestamp(target_timestamp)
    previous_timestamp = pd.Timestamp(previous_timestamp)
    next_timestamp = pd.Timestamp(next_timestamp)

    df.loc[target_timestamp] = (
        df.loc[previous_timestamp] + df.loc[next_timestamp]
    ) / 2

    return df.sort_index()


def fill_known_missing_values(
    combined_dfs: Dict[int, pd.DataFrame],
) -> Dict[int, pd.DataFrame]:
    """
    Fill the known missing timestamps used in the original script.
    """
    combined_dfs = {year: df.copy() for year, df in combined_dfs.items()}

    replacements = {
        2025: [
            (
                "2025-01-06 14:00:00",
                "2025-01-06 13:00:00",
                "2025-01-06 15:00:00",
            ),
        ],
        2026: [
            (
                "2026-01-08 13:00:00",
                "2026-01-08 12:00:00",
                "2026-01-08 14:00:00",
            ),
            (
                "2026-01-15 13:00:00",
                "2026-01-15 12:00:00",
                "2026-01-15 14:00:00",
            ),
            (
                "2026-01-29 05:00:00",
                "2026-01-29 04:00:00",
                "2026-01-29 06:00:00",
            ),
        ],
    }

    for year, rows in replacements.items():
        if year not in combined_dfs:
            continue

        for target, previous, next_ in rows:
            combined_dfs[year] = interpolate_single_missing_hour(
                combined_dfs[year],
                target_timestamp=target,
                previous_timestamp=previous,
                next_timestamp=next_,
            )

    return combined_dfs


def prepare_future_exog(
    weather_df: pd.DataFrame,
    features: Sequence[str] = DEFAULT_FEATURES,
    temp_roll_window: int = 24,
) -> pd.DataFrame:
    """
    Prepare exogenous variables for forecasting/simulation.
    Usually called with January 2026 weather data.
    """
    X = weather_df[["temperature", "wind_speed"]].copy()
    X["temperature_roll24"] = (
        X["temperature"]
        .rolling(temp_roll_window, min_periods=1)
        .mean()
    )
    X["weekend"] = (X.index.dayofweek >= 5).astype(int)

    return X[list(features)].dropna()


# ---------------------------------------------------------------------
# SARIMAX model selection and fitting
# ---------------------------------------------------------------------

def fit_single_sarimax(
    y: pd.Series,
    X: pd.DataFrame,
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
    maxiter: int = 300,
    method: str = "powell",
):
    """Fit one SARIMAX model."""
    model = SARIMAX(
        y,
        exog=X,
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    return model.fit(disp=False, maxiter=maxiter, method=method)


def fit_sarimax_candidates(
    combined_dfs: Dict[int, pd.DataFrame],
    features: Sequence[str] = DEFAULT_FEATURES,
    candidate_models: Sequence[
        Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]
    ] = DEFAULT_CANDIDATE_MODELS,
    train_years: Iterable[int] = DEFAULT_TRAIN_YEARS,
    rank_metric: str = "BIC",
    ljungbox_lag: int = 24,
    maxiter: int = 300,
    method: str = "powell",
    verbose: bool = True,
) -> Tuple[pd.DataFrame, Dict[int, pd.DataFrame]]:
    """
    Test candidate SARIMAX models separately for each training year.

    Returns:
    - all_results_df: combined result table across all years
    - yearly_tables: dictionary of result tables per year
    """
    yearly_tables = {}
    all_results = []

    for year in train_years:
        df = combined_dfs[year]
        df_model = df[["consumption", *features]].dropna()

        y = df_model["consumption"]
        X = df_model[list(features)]

        yearly_results = []

        for model_id, (order, seasonal_order) in enumerate(candidate_models):
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")

                    fit = fit_single_sarimax(
                        y=y,
                        X=X,
                        order=order,
                        seasonal_order=seasonal_order,
                        maxiter=maxiter,
                        method=method,
                    )

                lb = acorr_ljungbox(
                    fit.resid.dropna(),
                    lags=[ljungbox_lag],
                    return_df=True,
                )

                yearly_results.append(
                    {
                        "year": year,
                        "model_id": model_id,
                        "order": order,
                        "seasonal_order": seasonal_order,
                        "AIC": fit.aic,
                        "BIC": fit.bic,
                        f"Ljung-Box p-value lag {ljungbox_lag}": lb.loc[
                            ljungbox_lag, "lb_pvalue"
                        ],
                        "converged": fit.mle_retvals.get("converged", None),
                        "error": None,
                    }
                )

            except Exception as exc:
                yearly_results.append(
                    {
                        "year": year,
                        "model_id": model_id,
                        "order": order,
                        "seasonal_order": seasonal_order,
                        "AIC": np.nan,
                        "BIC": np.nan,
                        f"Ljung-Box p-value lag {ljungbox_lag}": np.nan,
                        "converged": False,
                        "error": str(exc),
                    }
                )

        yearly_table = pd.DataFrame(yearly_results)
        yearly_table = yearly_table.sort_values(
            rank_metric,
            ascending=True,
            na_position="last",
        ).reset_index(drop=True)

        # Score only successfully fitted models.
        yearly_table["score"] = 0
        valid_idx = yearly_table[rank_metric].notna()
        n_valid = int(valid_idx.sum())
        yearly_table.loc[valid_idx, "score"] = range(n_valid, 0, -1)

        yearly_tables[year] = yearly_table
        all_results.append(yearly_table)

        if verbose:
            print(f"\nSARIMAX comparison for January {year}, ranked by {rank_metric}")
            print(yearly_table)

    all_results_df = pd.concat(all_results, ignore_index=True)
    return all_results_df, yearly_tables


def summarize_model_scores(
    all_results_df: pd.DataFrame,
    n_candidate_models: int,
) -> pd.DataFrame:
    """
    Aggregate yearly candidate-model results into one overall score table.
    """
    score_summary = (
        all_results_df
        .groupby(["order", "seasonal_order"], as_index=False)
        .agg(
            total_score=("score", "sum"),
            mean_score=("score", "mean"),
            mean_AIC=("AIC", "mean"),
            mean_BIC=("BIC", "mean"),
            median_AIC=("AIC", "median"),
            median_BIC=("BIC", "median"),
            first_places=("score", lambda x: (x == n_candidate_models).sum()),
            years_tested=("score", "count"),
        )
        .sort_values(
            ["total_score", "first_places", "mean_BIC"],
            ascending=[False, False, True],
        )
        .reset_index(drop=True)
    )

    return score_summary


def fit_yearly_sarimax_models(
    combined_dfs: Dict[int, pd.DataFrame],
    order: Tuple[int, int, int],
    seasonal_order: Tuple[int, int, int, int],
    features: Sequence[str] = DEFAULT_FEATURES,
    train_years: Iterable[int] = DEFAULT_TRAIN_YEARS,
    maxiter: int = 300,
    method: str = "powell",
    verbose: bool = True,
) -> Dict[int, object]:
    """
    Fit the selected SARIMAX order separately for each training year.
    """
    fitted_models = {}

    for year in train_years:
        df = combined_dfs[year]
        df_model = df[["consumption", *features]].dropna()

        fit = fit_single_sarimax(
            y=df_model["consumption"],
            X=df_model[list(features)],
            order=order,
            seasonal_order=seasonal_order,
            maxiter=maxiter,
            method=method,
        )

        fitted_models[year] = fit

        if verbose:
            print(f"{year}: AIC = {fit.aic:.2f}, BIC = {fit.bic:.2f}")

    return fitted_models


# ---------------------------------------------------------------------
# Coefficients and Monte Carlo simulation
# ---------------------------------------------------------------------

def build_coef_table(fitted_models: Dict[int, object]) -> pd.DataFrame:
    """Extract parameter estimates, standard errors, and p-values."""
    coef_rows = []

    for year, fit in fitted_models.items():
        for name in fit.params.index:
            coef_rows.append(
                {
                    "year": year,
                    "parameter": name,
                    "estimate": fit.params[name],
                    "standard_dev": fit.bse[name],
                    "p_value": fit.pvalues[name],
                }
            )

    return pd.DataFrame(coef_rows)


def build_beta_tables(
    fitted_models: Dict[int, object],
    features: Sequence[str] = DEFAULT_FEATURES,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Extract beta coefficients for the exogenous features and compute
    cross-year mean/std for each feature.
    """
    beta_rows = []

    for year, fit in fitted_models.items():
        for feature in features:
            beta_rows.append(
                {
                    "year": year,
                    "feature": feature,
                    "beta": fit.params[feature],
                }
            )

    beta_df = pd.DataFrame(beta_rows)

    beta_stats = (
        beta_df
        .groupby("feature")["beta"]
        .agg(["mean", "std"])
        .reset_index()
    )

    return beta_df, beta_stats


def simulate_demand_ensemble_with_random_betas(
    fitted_models: Dict[int, object],
    X_future: pd.DataFrame,
    beta_stats: pd.DataFrame,
    features: Sequence[str] = DEFAULT_FEATURES,
    n_sims: int = 1000,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate Monte Carlo demand trajectories.

    Each simulation:
    1. Randomly selects one historical SARIMAX model.
    2. Forecasts target-period demand using target-period exogenous variables.
    3. Draws new beta coefficients from cross-year beta distributions.
    4. Adjusts the forecast relative to the chosen model's original beta.
    5. Adds random residual noise from the selected model.
    """
    rng = np.random.default_rng(seed)

    years = list(fitted_models.keys())
    if len(years) == 0:
        raise ValueError("fitted_models is empty.")

    sims = []

    beta_mean = beta_stats.set_index("feature")["mean"].to_dict()
    beta_std = beta_stats.set_index("feature")["std"].fillna(0.0).to_dict()

    for i in range(n_sims):
        year = rng.choice(years)
        fit = fitted_models[year]

        forecast = fit.get_forecast(
            steps=len(X_future),
            exog=X_future[list(features)],
        ).predicted_mean

        beta_adjustment = np.zeros(len(X_future))

        for feature in features:
            beta_draw = rng.normal(
                beta_mean[feature],
                beta_std[feature],
            )
            beta_original = fit.params[feature]

            beta_adjustment += (
                beta_draw - beta_original
            ) * X_future[feature].values

        resid = fit.resid.dropna()
        sigma = resid.std()

        noise = rng.normal(
            loc=0,
            scale=sigma,
            size=len(X_future),
        )

        sim = pd.Series(
            forecast.values + beta_adjustment + noise,
            index=X_future.index,
            name=f"sim_{i + 1}",
        )

        sims.append(sim)

    return pd.concat(sims, axis=1)


def build_mc_summary(
    demand_mc: pd.DataFrame,
    actual: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """
    Build p5 / p50 / p95 / mean summary table from Monte Carlo simulations.
    """
    summary = pd.DataFrame(
        {
            "timestamp": demand_mc.index,
            "demand_p5": demand_mc.quantile(0.05, axis=1).values,
            "demand_p50": demand_mc.quantile(0.50, axis=1).values,
            "demand_p95": demand_mc.quantile(0.95, axis=1).values,
            "demand_mean": demand_mc.mean(axis=1).values,
        }
    )

    if actual is not None:
        actual_aligned = actual.reindex(demand_mc.index)
        summary.insert(1, "demand_actual", actual_aligned.values)

    return summary


def save_mc_outputs(
    demand_mc: pd.DataFrame,
    summary: pd.DataFrame,
    output_dir: str = "../data",
    full_filename: str = "demand_mc_full_jan2026.csv",
    summary_filename: str = "demand_mc_jan2026.csv",
) -> None:
    """
    Save full simulations and summary table to CSV.
    """
    os.makedirs(output_dir, exist_ok=True)

    demand_mc.to_csv(os.path.join(output_dir, full_filename), index=True)
    summary.to_csv(os.path.join(output_dir, summary_filename), index=False)


# ---------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------

def plot_january_demand_grid(
    demand_series: Dict[int, pd.Series],
    years: Optional[Iterable[int]] = None,
    figsize: Tuple[int, int] = (14, 12),
    sharey: bool = True,
) -> None:
    """Plot January demand for multiple years in a grid."""
    if years is None:
        years = sorted(demand_series.keys())
    years = list(years)

    n_plots = len(years)
    n_cols = 2
    n_rows = int(np.ceil(n_plots / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize, sharey=sharey)
    axes = np.array(axes).reshape(-1)

    for ax, year in zip(axes, years):
        series = demand_series[year]
        ax.plot(series.index, series.values)
        ax.set_title(f"January {year} consumption")
        ax.set_ylabel("Consumption")
        ax.grid(True)

    for ax in axes[n_plots:]:
        ax.axis("off")

    plt.tight_layout()
    plt.show()


def plot_acf_pacf(
    series: pd.Series,
    title: str,
    lags: int = 168,
) -> None:
    """Plot ACF and PACF for one series."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))

    plot_acf(series.dropna(), lags=lags, ax=axes[0])
    axes[0].set_title(f"ACF: {title}")
    axes[0].grid(True)

    plot_pacf(series.dropna(), lags=lags, ax=axes[1], method="ywm")
    axes[1].set_title(f"PACF: {title}")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


def plot_beta_by_year(beta_df: pd.DataFrame) -> None:
    """Plot yearly beta values for each exogenous feature."""
    features = beta_df["feature"].unique()

    for feature in features:
        subset = beta_df[beta_df["feature"] == feature]
        plt.plot(subset["year"], subset["beta"], marker="o", label=feature)

    plt.axhline(0, linewidth=1)
    plt.title("Estimated beta coefficients by year")
    plt.xlabel("Year")
    plt.ylabel("Beta estimate")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_mc_simulations(
    demand_mc: pd.DataFrame,
    actual: Optional[pd.Series] = None,
    include_sim_lines: bool = True,
    alpha: float = 0.15,
) -> None:
    """
    Plot Monte Carlo simulations, MC mean, actual demand, and p95 line.
    """
    demand_p95 = demand_mc.quantile(0.95, axis=1)

    plt.figure(figsize=(14, 6))

    if include_sim_lines:
        plt.plot(demand_mc.index, demand_mc.iloc[:, :], alpha=alpha)

    plt.plot(
        demand_mc.index,
        demand_mc.mean(axis=1),
        color="black",
        linewidth=2,
        label="MC mean",
    )

    if actual is not None:
        actual_aligned = actual.reindex(demand_mc.index)
        plt.plot(
            demand_mc.index,
            actual_aligned,
            color="red",
            linewidth=2,
            label="Actual demand",
        )

    plt.plot(
        demand_mc.index,
        demand_p95,
        color="blue",
        linewidth=2,
        label="MC 95th percentile",
    )

    plt.title("Monte Carlo demand simulations")
    plt.ylabel("Consumption")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_mc_uncertainty_band(
    demand_mc: pd.DataFrame,
    actual: Optional[pd.Series] = None,
) -> None:
    """
    Clean report-style plot with p5-p95 band and mean.
    """
    demand_p5 = demand_mc.quantile(0.05, axis=1)
    demand_p95 = demand_mc.quantile(0.95, axis=1)
    demand_mean = demand_mc.mean(axis=1)

    plt.figure(figsize=(14, 6))
    plt.fill_between(demand_mc.index, demand_p5, demand_p95, alpha=0.3, label="5-95% band")
    plt.plot(demand_mc.index, demand_mean, color="black", linewidth=2, label="MC mean")

    if actual is not None:
        actual_aligned = actual.reindex(demand_mc.index)
        plt.plot(demand_mc.index, actual_aligned, color="red", linewidth=2, label="Actual demand")

    plt.title("Demand uncertainty band")
    plt.ylabel("Consumption")
    plt.legend()
    plt.grid(True)
    plt.show()


# ---------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------

def run_pipeline(
    years: Iterable[int] = DEFAULT_YEARS,
    train_years: Iterable[int] = DEFAULT_TRAIN_YEARS,
    target_year: int = DEFAULT_TARGET_YEAR,
    lat: float = DEFAULT_LAT,
    lon: float = DEFAULT_LON,
    features: Sequence[str] = DEFAULT_FEATURES,
    model_order: Optional[Tuple[int, int, int]] = None,
    seasonal_model_order: Optional[Tuple[int, int, int, int]] = None,
    candidate_models: Sequence[
        Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]
    ] = DEFAULT_CANDIDATE_MODELS,
    rank_metric: str = "BIC",
    n_sims: int = 1000,
    seed: int = 42,
    fill_known_missing: bool = True,
    save_outputs: bool = False,
    output_dir: str = "../data",
    verbose: bool = True,
) -> PipelineResult:
    """
    Run the full demand modelling pipeline.

    This is the easiest entry point from a notebook:

        from timeseries_module import run_pipeline
        result = run_pipeline(n_sims=1000, save_outputs=True)

    To skip the expensive candidate search, provide both model orders directly:

        result = run_pipeline(
            model_order=(2, 0, 2),
            seasonal_model_order=(1, 1, 1, 24),
            n_sims=1000,
        )

    If either model_order or seasonal_model_order is omitted, the pipeline
    tests all candidate_models and selects the best one as before.

    Then access results as:

        result.demand_mc
        result.summary
        result.score_summary
        result.beta_df
    """
    years = tuple(years)
    train_years = tuple(train_years)

    demand_series = fetch_all_january_demand(years=years, verbose=verbose)
    weather_series = fetch_all_january_weather(
        years=years,
        lat=lat,
        lon=lon,
        verbose=verbose,
    )

    combined_dfs = build_combined_dfs(
        demand_series=demand_series,
        weather_series=weather_series,
        years=years,
        verbose=verbose,
    )

    missing_hours = find_missing_hours_by_year(
        combined_dfs=combined_dfs,
        years=[year for year in years if year >= 2025],
        verbose=verbose,
    )

    if fill_known_missing:
        combined_dfs = fill_known_missing_values(combined_dfs)

    if (model_order is None) != (seasonal_model_order is None):
        raise ValueError(
            "Provide both model_order and seasonal_model_order, or omit both "
            "to run candidate-model selection."
        )

    if model_order is not None and seasonal_model_order is not None:
        best_order = tuple(model_order)
        best_seasonal_order = tuple(seasonal_model_order)

        # Candidate search is intentionally skipped when the user provides
        # the SARIMAX orders directly. Keep these outputs empty but valid so
        # PipelineResult has a consistent structure.
        all_results_df = pd.DataFrame(
            columns=[
                "year",
                "model_id",
                "order",
                "seasonal_order",
                "AIC",
                "BIC",
                f"Ljung-Box p-value lag 24",
                "converged",
                "error",
                "score",
            ]
        )
        score_summary = pd.DataFrame(
            [
                {
                    "order": best_order,
                    "seasonal_order": best_seasonal_order,
                    "total_score": np.nan,
                    "mean_score": np.nan,
                    "mean_AIC": np.nan,
                    "mean_BIC": np.nan,
                    "median_AIC": np.nan,
                    "median_BIC": np.nan,
                    "first_places": np.nan,
                    "years_tested": 0,
                }
            ]
        )

        if verbose:
            print("\nSkipping SARIMAX candidate search because model orders were provided.")
            print(f"order = {best_order}")
            print(f"seasonal_order = {best_seasonal_order}")

    else:
        all_results_df, _yearly_tables = fit_sarimax_candidates(
            combined_dfs=combined_dfs,
            features=features,
            candidate_models=candidate_models,
            train_years=train_years,
            rank_metric=rank_metric,
            verbose=verbose,
        )

        score_summary = summarize_model_scores(
            all_results_df=all_results_df,
            n_candidate_models=len(candidate_models),
        )

        best_overall_model = score_summary.iloc[0]
        best_order = best_overall_model["order"]
        best_seasonal_order = best_overall_model["seasonal_order"]

        if verbose:
            print("\nOverall SARIMAX model score across all years")
            print(score_summary)
            print("\nBest overall SARIMAX model:")
            print(f"order = {best_order}")
            print(f"seasonal_order = {best_seasonal_order}")

    fitted_models = fit_yearly_sarimax_models(
        combined_dfs=combined_dfs,
        order=best_order,
        seasonal_order=best_seasonal_order,
        features=features,
        train_years=train_years,
        verbose=verbose,
    )

    coef_table = build_coef_table(fitted_models)
    beta_df, beta_stats = build_beta_tables(fitted_models, features=features)

    X_future = prepare_future_exog(
        weather_df=weather_series[target_year],
        features=features,
    )

    demand_mc = simulate_demand_ensemble_with_random_betas(
        fitted_models=fitted_models,
        X_future=X_future,
        beta_stats=beta_stats,
        features=features,
        n_sims=n_sims,
        seed=seed,
    )

    actual_target = demand_series[target_year].reindex(demand_mc.index)
    summary = build_mc_summary(demand_mc=demand_mc, actual=actual_target)

    if save_outputs:
        save_mc_outputs(
            demand_mc=demand_mc,
            summary=summary,
            output_dir=output_dir,
        )

    if verbose:
        print(f"\nMC ensemble shape: {demand_mc.shape}")
        print(f"Actual {target_year} mean: {actual_target.mean():.0f} MW")
        print(f"MC mean: {demand_mc.mean(axis=1).mean():.0f} MW")

    return PipelineResult(
        demand_series=demand_series,
        weather_series=weather_series,
        combined_dfs=combined_dfs,
        missing_hours=missing_hours,
        all_results_df=all_results_df,
        score_summary=score_summary,
        best_order=best_order,
        best_seasonal_order=best_seasonal_order,
        fitted_models=fitted_models,
        coef_table=coef_table,
        beta_df=beta_df,
        beta_stats=beta_stats,
        X_future=X_future,
        demand_mc=demand_mc,
        actual_target=actual_target,
        summary=summary,
    )


if __name__ == "__main__":
    # Optional: run full pipeline when executing this file directly.
    # Nothing runs when importing this module into a notebook.
    result = run_pipeline(
        n_sims=100,
        save_outputs=True,
        verbose=True,
    )

    plot_mc_simulations(
        demand_mc=result.demand_mc,
        actual=result.actual_target,
    )
