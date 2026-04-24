
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ------------------------------------------------------------
# 1) Load and clean the Nord Pool UMM export
# ------------------------------------------------------------

UMM_FILE = Path("UMM_Messages_2026-04.03T12_03_19.xlsx")

YEAR = 2025
START = pd.Timestamp(f"{YEAR}-01-01 00:00:00")
END = pd.Timestamp(f"{YEAR}-12-31 23:59:59")


def parse_dt(series: pd.Series) -> pd.Series:
    return pd.to_datetime(
        series.astype(str).str.replace(".", "-", regex=False),
        errors="coerce"
    )


def parse_capacity_number(x):
    """
    Handles values like:
      83
      "72,372"
      "112,149"
      NaN

    We keep both:
      - sum of all numbers
      - max of all numbers
    because transmission entries often contain one value per direction.
    """
    if pd.isna(x):
        return []
    nums = re.findall(r"-?\d+(?:\.\d+)?", str(x).replace(" ", ""))
    return [float(n) for n in nums]


def overlap_with_year(start_s: pd.Series, stop_s: pd.Series, year_start, year_end) -> pd.Series:
    stop_filled = stop_s.fillna(year_end)
    return (start_s <= year_end) & (stop_filled >= year_start)


def load_umm_logs(path: Path = UMM_FILE) -> pd.DataFrame:
    df = pd.read_excel(path, header=3).copy()

    for col in ["Event Start", "Event Stop", "Published"]:
        df[col + "_dt"] = parse_dt(df[col])

    df["event_start"] = df["Event Start_dt"]
    df["event_stop"] = df["Event Stop_dt"]
    df["published"] = df["Published_dt"]

    df["unavailable_list"] = df["Unavailable"].apply(parse_capacity_number)
    df["available_list"] = df["Available"].apply(parse_capacity_number)

    df["unavailable_sum_mw"] = df["unavailable_list"].apply(lambda x: sum(x) if x else 0.0)
    df["unavailable_max_mw"] = df["unavailable_list"].apply(lambda x: max(x) if x else 0.0)
    df["available_sum_mw"] = df["available_list"].apply(lambda x: sum(x) if x else 0.0)
    df["available_max_mw"] = df["available_list"].apply(lambda x: max(x) if x else 0.0)

    df["text"] = (
        df["Infrastructure"].fillna("").astype(str) + " | " +
        df["Assets"].fillna("").astype(str) + " | " +
        df["Remarks"].fillna("").astype(str)
    )

    df["event_family"] = np.select(
        [
            df["Event"].str.contains("Production", case=False, na=False),
            df["Event"].str.contains("Transmission", case=False, na=False),
            df["Event"].str.contains("Consumption", case=False, na=False),
            df["Event"].str.contains("Market Information", case=False, na=False),
        ],
        ["production", "transmission", "consumption", "market_info"],
        default="other"
    )

    df["overlaps_2025"] = overlap_with_year(df["event_start"], df["event_stop"], START, END)
    return df


# ------------------------------------------------------------
# 2) Country relevance rules
# ------------------------------------------------------------

COUNTRY_RULES = {
    "EE": {
        "line_color": "#1f4e79",
        "event_color": "#2a6fba",
        "label": "Estonia (EE)",
        "infra_regex": (
            r"(^|[,\s|])EE([,\s|]|$)|"
            r"EE\s*→|→\s*EE|"
            r"Estlink|Auvere|Kiisa|Balti|Estonian"
        ),
        "market_info_regex": (
            r"Estonia|Estonian|Estlink|Auvere|Kiisa|Balti|"
            r"synchronous operation|Baltic CCR|balancing capacity|"
            r"emergency power plant|ramping limit"
        ),
    },
    "FI": {
        "line_color": "#183a63",
        "event_color": "#4f88c6",
        "label": "Finland (FI)",
        "infra_regex": (
            r"(^|[,\s|])FI([,\s|]|$)|"
            r"FI\s*→|→\s*FI|"
            r"Estlink|Olkiluoto|Fingrid|Tornio|Aurora Line"
        ),
        "market_info_regex": (
            r"Finland|Finnish|Fingrid|Estlink|Olkiluoto|Aurora Line|"
            r"synchronous operation|Baltic CCR|balancing capacity|"
            r"ramping limit"
        ),
    },
    "LV": {
        "line_color": "#7f1d1d",
        "event_color": "#d46a6a",
        "label": "Latvia (LV)",
        "infra_regex": (
            r"(^|[,\s|])LV([,\s|]|$)|"
            r"LV\s*→|→\s*LV|"
            r"Plavinas|Riga|AST BESS|CHP-1|CHP-2|Augstsprieguma"
        ),
        "market_info_regex": (
            r"Latvia|Latvian|Plavinas|Riga|AST BESS|CHP-1|CHP-2|"
            r"balancing capacity|synchronous operation|Baltic CCR|"
            r"ramping limit"
        ),
    },
    "LT": {
        "line_color": "#1f6f5f",
        "event_color": "#2da67d",
        "label": "Lithuania (LT)",
        "infra_regex": (
            r"(^|[,\s|])LT([,\s|]|$)|"
            r"LT\s*→|→\s*LT|"
            r"Kruonis|Lithuanian PP|ORLEN LT|Nordbalt|Litgrid"
        ),
        "market_info_regex": (
            r"Lithuania|Lithuanian|Kruonis|ORLEN LT|Nordbalt|Litgrid|"
            r"synchronous operation|Baltic CCR|balancing capacity|"
            r"ramping limit"
        ),
    },
}


def simplify_event_title(row: pd.Series, country: str) -> str:
    infra = str(row.get("Infrastructure", "") or "").strip()
    remarks = str(row.get("Remarks", "") or "").strip()
    event_family = row.get("event_family", "")

    if infra:
        base = infra
    else:
        base = str(row.get("Event", "")).replace("Unavailability of electricity facilities, ", "")

    if event_family == "transmission":
        title = f"Transmission: {base}"
    elif event_family == "production":
        title = f"Production: {base}"
    elif event_family == "consumption":
        title = f"Consumption: {base}"
    else:
        title = base

    if "synchronous operation" in remarks.lower():
        title = "Baltic synchronization"
    elif "balancing capacity" in remarks.lower():
        title = "Balancing-capacity change"
    elif "ramping limit" in remarks.lower():
        title = "Ramping-limit change"
    elif "Aurora Line" in remarks:
        title = "Aurora Line commissioning"

    return title[:75]


def filter_relevant_logs_for_country(df: pd.DataFrame, country: str) -> pd.DataFrame:
    rules = COUNTRY_RULES[country]
    out = df.copy()

    infra_hit = out["Infrastructure"].fillna("").str.contains(rules["infra_regex"], case=False, regex=True)
    text_hit = out["text"].fillna("").str.contains(rules["market_info_regex"], case=False, regex=True)

    is_structural = out["event_family"].isin(["production", "transmission", "consumption"])
    is_market_info = out["event_family"].eq("market_info")

    keyword_hit = out["Remarks"].fillna("").str.contains(
        r"outage|failure|maintenance|reduction|revision|capacity|commission|synchronous|balancing|ramping|emergency",
        case=False,
        regex=True,
    )

    out["country_relevant"] = (
        (out["overlaps_2025"]) &
        (
            (is_structural & infra_hit) |
            (is_market_info & (infra_hit | text_hit) & keyword_hit)
        )
    )

    out = out[out["country_relevant"]].copy()

    # priority score for annotation selection
    base = (
        out["unavailable_sum_mw"].fillna(0)
        + 0.25 * out["unavailable_max_mw"].fillna(0)
    )

    event_weight = np.select(
        [
            out["event_family"].eq("transmission"),
            out["event_family"].eq("production"),
            out["event_family"].eq("consumption"),
            out["event_family"].eq("market_info"),
        ],
        [1.35, 1.25, 1.10, 0.60],
        default=1.0
    )

    keyword_bonus = (
        out["Remarks"].fillna("").str.contains(
            r"outage|failure|maintenance|capacity|revision|synchronous|balancing|ramping|emergency|commission",
            case=False,
            regex=True
        ).astype(int) * 12
    )

    cross_border_bonus = (
        out["Infrastructure"].fillna("").str.contains(r"→|,|Estlink|Nordbalt|Aurora", regex=True).astype(int) * 15
    )

    out["event_score"] = base * event_weight + keyword_bonus + cross_border_bonus

    # event date to place vertical markers
    out["event_date"] = out["event_start"].fillna(out["published"]).dt.floor("D")

    # cleaner label
    out["short_label"] = out.apply(lambda row: simplify_event_title(row, country), axis=1)

    # de-duplicate obvious repeated/revised events
    out["dedupe_key"] = (
        out["short_label"].fillna("") + " | " +
        out["event_date"].astype(str)
    )
    out = out.sort_values(["event_score", "Revision"], ascending=[False, False])
    out = out.drop_duplicates("dedupe_key", keep="first")

    return out.sort_values("event_date")


# ------------------------------------------------------------
# 3) Price loading helpers
# ------------------------------------------------------------

def standardize_price_df(df: pd.DataFrame, country_code: str) -> pd.DataFrame:
    """
    Make the plotting code tolerant to different price column names.

    Expected output columns:
        timestamp
        price
        date
    """
    out = df.copy()
    cols_lower = {c.lower(): c for c in out.columns}

    possible_time_cols = [
        "timestamp", "datetime", "time", "date_time", "hour", "delivery_time", "deliverydate"
    ]
    possible_price_cols = [
        "price", "eur_mwh", "eur/mwh", "day_ahead_price", "spot_price", "ee", "fi", "lv", "lt"
    ]

    time_col = None
    for c in possible_time_cols:
        if c in cols_lower:
            time_col = cols_lower[c]
            break
    if time_col is None:
        raise ValueError(
            f"Could not find a timestamp column for {country_code}. "
            f"Available columns: {list(out.columns)}"
        )

    price_col = None
    country_lc = country_code.lower()
    # first try exact country code if the df has wide format
    if country_lc in cols_lower:
        price_col = cols_lower[country_lc]
    else:
        for c in possible_price_cols:
            if c in cols_lower:
                price_col = cols_lower[c]
                break

    if price_col is None:
        raise ValueError(
            f"Could not find a price column for {country_code}. "
            f"Available columns: {list(out.columns)}"
        )

    out["timestamp"] = pd.to_datetime(out[time_col], errors="coerce")
    out["price"] = pd.to_numeric(out[price_col], errors="coerce")
    out = out.dropna(subset=["timestamp", "price"]).copy()
    out["date"] = out["timestamp"].dt.floor("D")

    return out


def aggregate_daily_price(price_df: pd.DataFrame) -> pd.DataFrame:
    return (
        price_df.groupby("date", as_index=False)["price"]
        .mean()
        .rename(columns={"price": "avg_price"})
    )


# ------------------------------------------------------------
# 4) Plotting
# ------------------------------------------------------------

def plot_country_price_with_logs(price_df: pd.DataFrame,
                                 events_df: pd.DataFrame,
                                 country: str,
                                 top_n: int = 6,
                                 figsize=(14, 7),
                                 save_path=None):
    rules = COUNTRY_RULES[country]
    price_daily = aggregate_daily_price(standardize_price_df(price_df, country))

    events_plot = events_df.copy()
    events_plot = events_plot[
        (events_plot["event_date"] >= START.normalize()) &
        (events_plot["event_date"] <= END.normalize())
    ].copy()

    # choose top events for annotation, but keep only one every ~10 days to avoid clutter
    top = events_plot.sort_values("event_score", ascending=False).copy()

    selected_rows = []
    selected_dates = []
    for _, row in top.iterrows():
        d = row["event_date"]
        if pd.isna(d):
            continue
        if not any(abs((d - x).days) < 10 for x in selected_dates):
            selected_rows.append(row)
            selected_dates.append(d)
        if len(selected_rows) >= top_n:
            break

    selected = pd.DataFrame(selected_rows)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(
        price_daily["date"],
        price_daily["avg_price"],
        linewidth=2.8,
        color=rules["line_color"],
        alpha=0.95
    )

    # faint background series effect
    ax.plot(
        price_daily["date"],
        price_daily["avg_price"].rolling(14, min_periods=1).mean(),
        linewidth=1.5,
        color=rules["line_color"],
        alpha=0.25
    )

    ymin, ymax = ax.get_ylim()
    yspan = ymax - ymin if ymax > ymin else 1.0

    if not selected.empty:
        y_positions = np.linspace(
            ymin + 0.85 * yspan,
            ymin + 0.35 * yspan,
            len(selected)
        )

        for i, (_, row) in enumerate(selected.iterrows()):
            x = row["event_date"]
            ax.axvline(x, linestyle="--", linewidth=1.3, alpha=0.45, color=rules["event_color"])

            y_point = float(
                price_daily.loc[price_daily["date"] == x, "avg_price"].iloc[0]
            ) if (price_daily["date"] == x).any() else float(price_daily["avg_price"].mean())

            ax.scatter([x], [y_point], s=55, color=rules["event_color"], edgecolor="white", linewidth=1.1, zorder=5)

            ax.annotate(
                row["short_label"],
                xy=(x, y_point),
                xytext=(x, y_positions[i]),
                textcoords="data",
                fontsize=10,
                fontweight="bold",
                color=rules["event_color"],
                ha="center",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=rules["event_color"], alpha=0.95),
                arrowprops=dict(arrowstyle="->", color=rules["event_color"], lw=1.1, alpha=0.85),
            )

    ax.set_title(f"{rules['label']} electricity prices vs. selected UMM logs ({YEAR})", fontsize=16, fontweight="bold")
    ax.set_ylabel("Average daily price (EUR/MWh)", fontsize=12)
    ax.set_xlabel("")
    ax.grid(True, alpha=0.25)

    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    plt.setp(ax.get_xticklabels(), rotation=0)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches="tight")

    return fig, ax


# ------------------------------------------------------------
# 5) Example usage
# ------------------------------------------------------------
#
# IMPORTANT:
# Replace the four price dataframes below with the ones you already build
# in your own Price_analysis.ipynb.
#
# Example:
#   ee_prices = pd.read_csv("ee_prices_2025.csv")
#   fi_prices = pd.read_csv("fi_prices_2025.csv")
#   lv_prices = pd.read_csv("lv_prices_2025.csv")
#   lt_prices = pd.read_csv("lt_prices_2025.csv")
#
# Then run:
#
#   umm = load_umm_logs()
#   ee_logs = filter_relevant_logs_for_country(umm, "EE")
#   fi_logs = filter_relevant_logs_for_country(umm, "FI")
#   lv_logs = filter_relevant_logs_for_country(umm, "LV")
#   lt_logs = filter_relevant_logs_for_country(umm, "LT")
#
#   plot_country_price_with_logs(ee_prices, ee_logs, "EE", save_path="ee_price_vs_logs_2025.png")
#   plot_country_price_with_logs(fi_prices, fi_logs, "FI", save_path="fi_price_vs_logs_2025.png")
#   plot_country_price_with_logs(lv_prices, lv_logs, "LV", save_path="lv_price_vs_logs_2025.png")
#   plot_country_price_with_logs(lt_prices, lt_logs, "LT", save_path="lt_price_vs_logs_2025.png")
#
#   ee_logs.to_csv("relevant_umm_logs_ee_2025.csv", index=False)
#   fi_logs.to_csv("relevant_umm_logs_fi_2025.csv", index=False)
#   lv_logs.to_csv("relevant_umm_logs_lv_2025.csv", index=False)
#   lt_logs.to_csv("relevant_umm_logs_lt_2025.csv", index=False)


if __name__ == "__main__":
    umm = load_umm_logs()

    ee_logs = filter_relevant_logs_for_country(umm, "EE")
    fi_logs = filter_relevant_logs_for_country(umm, "FI")
    lv_logs = filter_relevant_logs_for_country(umm, "LV")
    lt_logs = filter_relevant_logs_for_country(umm, "LT")

    ee_logs.to_csv("relevant_umm_logs_ee_2025.csv", index=False)
    fi_logs.to_csv("relevant_umm_logs_fi_2025.csv", index=False)
    lv_logs.to_csv("relevant_umm_logs_lv_2025.csv", index=False)
    lt_logs.to_csv("relevant_umm_logs_lt_2025.csv", index=False)

    print("Saved:")
    print("  relevant_umm_logs_ee_2025.csv")
    print("  relevant_umm_logs_fi_2025.csv")
    print("  relevant_umm_logs_lv_2025.csv")
    print("  relevant_umm_logs_lt_2025.csv")
