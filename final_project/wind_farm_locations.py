"""
Wind Farm Location Extraction — Estonian Land Board GIS Data
=============================================================
Extracts centroid coordinates from 6 Estonian wind-energy planning-stage
GeoJSON files and maps which polygons feed into the counterfactual scenarios.

Run this script once to inspect and verify farm coordinates:
    python wind_farm_locations.py

The coordinates printed here are the ones hardcoded in wind_counterfactual.py.

Data source: Estonian Land Board open geodata (maaamet.ee / geoportal.ee)
  tuul_ep_kehtestatud.geojson        — legally enacted special plans
  tuulealad_DP.geojson               — enacted detailed plans
  tuuleenergeetika_KOV_EP_alad.geojson — municipal special plans (pipeline)
  RMK_MARU_lepingus.geojson          — RMK / MARU state-forest agreements
  Riigimaade_enampakkumine_voor_alad.geojson    — state-land auction areas
  Riigimaade_enampakkumine_voor_kataster.geojson — state-land auction cadastral
"""

import os
import geopandas as gpd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

_DIR  = os.path.dirname(os.path.abspath(__file__))
_DATA = os.path.join(_DIR, "..", "data")
GIS_DIR = os.path.join(_DATA, "wind_farm_locations")

# ── GeoJSON files and their planning-stage labels ────────────────────────────
FILES = {
    "tuul_ep_kehtestatud.geojson":                    "established",
    "tuulealad_DP.geojson":                           "detailed_plan",
    "tuuleenergeetika_KOV_EP_alad.geojson":           "municipal_plan",
    "RMK_MARU_lepingus.geojson":                      "contractual",
    "Riigimaade_enampakkumine_voor_alad.geojson":     "auction_areas",
    "Riigimaade_enampakkumine_voor_kataster.geojson": "auction_cadastral",
}

# Name field(s) per file — primary name, optional municipality/developer
NAME_FIELDS = {
    "tuul_ep_kehtestatud.geojson":                    ("Plan_nimi",     "KOV"),
    "tuulealad_DP.geojson":                           ("DP_nimi",       None),
    "tuuleenergeetika_KOV_EP_alad.geojson":           ("Projekti_nimi", "Kohalik_omavalitsus"),
    "RMK_MARU_lepingus.geojson":                      ("Ala_nimi",      "Arendaja"),
    "Riigimaade_enampakkumine_voor_alad.geojson":     ("Ala_nim",       "KOV"),
    "Riigimaade_enampakkumine_voor_kataster.geojson": ("Ala_nim",       "KOV"),
}

# Scenario A — 3 farms used in wind_counterfactual.py (established/detailed plans)
SCENARIO_A = {
    "Lääneranna valla tuuleparkide eriplaneeringu kehtestamine ala 2 osas": ("Lääneranna (area 2)",    137),
    "Pärnu linna tuuleenergeetika eriplaneeringu kehtestamine (Põlendmaa)": ("Pärnu+Tori (Põlendmaa)", 86),
    "Aidu ja Aidu-Liiva":                                                   ("Aidu renewable energy park", 100),
}

# Scenario B extra — 5 municipal-plan farms (tuuleenergeetika_KOV_EP_alad)
SCENARIO_B_KEYWORDS = [
    "Lääneranna",      # Lääneranna pipeline areas
    "Tori valla",      # Tori rural municipality
    "Lääne-Nigula",    # Lääne-Nigula
    "Põhja-Pärnumaa",  # Põhja-Pärnumaa
    "Evecon",          # Lüganuse (Evecon + Enery Estonia)
]


def load_all() -> pd.DataFrame:
    """Load all 6 GeoJSON files, reproject to WGS84, extract centroids."""
    rows = []
    for fname, stage in FILES.items():
        path = os.path.join(GIS_DIR, fname)
        if not os.path.exists(path):
            print(f"  WARNING: not found — {fname}")
            continue

        gdf = gpd.read_file(path)
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326)
        elif gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)

        name_col, muni_col = NAME_FIELDS[fname]
        for _, row in gdf.iterrows():
            c = row.geometry.centroid
            rows.append({
                "source_file":    fname,
                "planning_stage": stage,
                "area_name":      str(row.get(name_col, "")).strip(),
                "municipality":   str(row.get(muni_col, "")).strip() if muni_col else "",
                "centroid_lat":   round(c.y, 6),
                "centroid_lon":   round(c.x, 6),
            })

    return pd.DataFrame(rows)


def tag_scenarios(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'scenario' column: A, B, or Other."""
    def _tag(row):
        if row["area_name"] in SCENARIO_A:
            return "A"
        if row["planning_stage"] == "municipal_plan":
            for kw in SCENARIO_B_KEYWORDS:
                if kw.lower() in row["area_name"].lower():
                    return "B"
        return "Other"

    df = df.copy()
    df["scenario"] = df.apply(_tag, axis=1)
    return df


def print_summary(df: pd.DataFrame) -> None:
    print(f"\nTotal polygons loaded:  {len(df)}")
    print(f"By planning stage:")
    print(df.groupby("planning_stage").size().to_string())

    print("\n── Scenario A farms (established/detailed plans) ──────────────────")
    scen_a = df[df["scenario"] == "A"][["area_name", "centroid_lat", "centroid_lon"]]
    for _, row in scen_a.iterrows():
        label, mw = SCENARIO_A[row["area_name"]]
        print(f"  {label:30s}  lat={row['centroid_lat']}  lon={row['centroid_lon']}  ({mw} MW)")

    print("\n── Scenario B farms (municipal plans, pipeline) ─────────────────")
    scen_b = df[df["scenario"] == "B"][["area_name", "centroid_lat", "centroid_lon"]]
    for _, row in scen_b.iterrows():
        print(f"  {row['area_name'][:55]:55s}  lat={row['centroid_lat']}  lon={row['centroid_lon']}")


def plot_map(df: pd.DataFrame) -> None:
    """Map of all planning areas coloured by scenario selection."""
    colors = {"established": "#2ecc71", "detailed_plan": "#27ae60",
              "municipal_plan": "#3498db", "contractual": "#95a5a6",
              "auction_areas": "#bdc3c7", "auction_cadastral": "#ecf0f1"}

    LON_MIN, LON_MAX = 21.5, 28.5
    LAT_MIN, LAT_MAX = 57.5, 60.2

    fig, ax = plt.subplots(figsize=(13, 9))
    ax.set_facecolor("#c9e8f5")

    # Land background — try geodatasets (geopandas ≥0.14) then deprecated API
    import importlib
    land = None
    try:
        _gds = importlib.import_module("geodatasets")
        land = gpd.read_file(_gds.get_path("naturalearth.land"))
    except Exception:
        try:
            _ds = getattr(gpd, "datasets", None)
            if _ds is not None:
                land = gpd.read_file(_ds.get_path("naturalearth_lowres"))
        except Exception:
            pass
    if land is not None:
        land.cx[LON_MIN:LON_MAX, LAT_MIN:LAT_MAX].plot(
            ax=ax, color="#f0ede0", edgecolor="#aaaaaa", linewidth=0.5, zorder=1)

    for fname, stage in FILES.items():
        path = os.path.join(GIS_DIR, fname)
        if not os.path.exists(path):
            continue
        gdf = gpd.read_file(path)
        if gdf.crs is None:
            gdf = gdf.set_crs(epsg=4326)
        elif gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
        gdf.plot(ax=ax, color=colors[stage], alpha=0.55, edgecolor="white", linewidth=0.4, zorder=2)

    # Overlay Scenario A and B centroid points
    tagged = tag_scenarios(df)
    for scenario, marker, color, zorder in [("A", "★", "#e74c3c", 5), ("B", "●", "#e67e22", 4)]:
        sub = tagged[tagged["scenario"] == scenario]
        ax.scatter(sub["centroid_lon"], sub["centroid_lat"],
                   s=120, color=color, zorder=zorder,
                   label=f"Scenario {scenario} farm centroid")
        for _, row in sub.iterrows():
            if row["scenario"] == "A":
                label = SCENARIO_A.get(row["area_name"], ("", 0))[0]
            else:
                label = row["area_name"][:30]
            ax.annotate(label, (row["centroid_lon"], row["centroid_lat"]),
                        textcoords="offset points", xytext=(5, 3),
                        fontsize=6.5, color=color)

    # Legend
    patches = [mpatches.Patch(color=v, alpha=0.6, label=k.replace("_", " "))
               for k, v in colors.items()]
    ax.legend(handles=patches + [
        plt.Line2D([0], [0], marker="*", color="w", markerfacecolor="#e74c3c",
                   markersize=10, label="Scenario A centroid"),
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#e67e22",
                   markersize=8,  label="Scenario B centroid"),
    ], fontsize=7.5, loc="upper right")

    ax.set_title("Estonian wind-energy planning areas — GIS source polygons\n"
                 "(centroids feed into wind_counterfactual.py)", fontsize=11)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    plt.tight_layout()
    plt.savefig(os.path.join(_DIR, "..", "figures", "wind_farm_gis_map.png"),
                dpi=150, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    print("=== Loading GIS planning-area polygons ===")
    df = load_all()
    df = tag_scenarios(df)
    print_summary(df)
    print("\n=== Generating map ===")
    plot_map(df)
    print("\nDone. Coordinates above match the hardcoded values in wind_counterfactual.py.")
