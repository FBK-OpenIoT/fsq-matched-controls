"""LOOCV fold definitions for CropGlobe and future multi-dataset experiments.

CropGlobe LOOCV (country-level)
--------------------------------
For CropGlobe-only experiments, the LOOCV unit is the country.
Hold out one country at a time; train on the other 9.
This gives a preliminary within-dataset generalisation estimate before
the full multi-dataset LOOCV (which requires other datasets from the proposal).

Multi-dataset LOOCV (§3.2 of the research proposal)
----------------------------------------------------
The full experiment uses climate-zone-level LOOCV over four zones:
  Z1 — Mediterranean     (S2AGRI100, EuroCropsML-PT)
  Z2 — Oceanic temperate (BreizhCrops, PASTIS)
  Z3 — Continental        (ZueriCrop, EuroCropsML-EE/LV)
  Z4 — S. Hemisphere     (SACT, South Africa)

CropGlobe countries can be assigned to these zones where appropriate
and added to the multi-dataset pool in later stages.
"""

from __future__ import annotations

# ── CropGlobe country codes ──────────────────────────────────────────────────

ALL_CROPGLOBE_COUNTRIES: list[str] = [
    "ARG",    # Argentina           — temperate continental / semi-arid
    "AUS",    # Australia           — semi-arid / mediterranean
    "BEL",    # Belgium             — oceanic temperate (Cfb)
    "CHN",    # China               — mixed (continental + subtropical)
    "FRA",    # France              — oceanic (Cfb) + mediterranean (Csa)
    "GBR",    # United Kingdom      — oceanic temperate (Cfb)
    "NLD",    # Netherlands         — oceanic temperate (Cfb)
    "USA",    # USA (full)         — multiple climate zones
    "USA_17", # USA 2017 subset
    "USA_20", # USA 2020 subset
]

#: CropGlobe harmonised class names (best-effort from Tong et al. 2024 Table 1).
#: Verify against the paper before using in publications.
CROPGLOBE_LABEL_NAMES: dict[int, str] = {
    0: "Corn",
    1: "Soybeans",
    2: "Rice",
    3: "Wheat",
    4: "Sugarcane",
    5: "Cotton",
    6: "Other",
}

#: Sentinel-2 L2A band names for the 10 bands used in CropGlobe.
S2_BAND_NAMES: list[str] = [
    "B2 (Blue)", "B3 (Green)", "B4 (Red)",
    "B5 (RE1)",  "B6 (RE2)",  "B7 (RE3)",
    "B8 (NIR)",  "B8A (RE4)",
    "B11 (SWIR1)", "B12 (SWIR2)",
]

# ── CropGlobe country → approximate Köppen zone mapping ──────────────────────
# Used to design country-level LOOCV folds that respect climate diversity.
# This is approximate — some countries span multiple zones.

CROPGLOBE_COUNTRY_ZONE: dict[str, str] = {
    "BEL": "Z2_oceanic",
    "GBR": "Z2_oceanic",
    "NLD": "Z2_oceanic",
    "FRA": "Z2_oceanic",       # majority Cfb (Brittany / Nord), some Csa south
    "CHN": "Z3_continental",
    "USA": "Z3_continental",   # mixed; treating as continental/temperate
    "USA_17": "Z3_continental",
    "USA_20": "Z3_continental",
    "ARG": "Z4_sh_semiarid",
    "AUS": "Z4_sh_semiarid",
}

#: Suggested country-level LOOCV folds for CropGlobe-only experiments.
#: Each fold: key=fold_name, value=list of held-out countries.
CROPGLOBE_LOOCV_FOLDS: dict[str, list[str]] = {
    "hold_CHN":    ["CHN"],
    "hold_BEL":    ["BEL"],
    "hold_GBR":    ["GBR"],
    "hold_NLD":    ["NLD"],
    "hold_ARG":    ["ARG"],
    "hold_AUS":    ["AUS"],
    "hold_SH":     ["ARG", "AUS"],          # hold out S. Hemisphere together
    "hold_oceanic": ["BEL", "GBR", "NLD"],  # hold out N. European zone
}

# ── Multi-dataset fold definitions (future — once other datasets are added) ──

#: Zone assignments from §3.2 of the proposal.
#: key = zone_id, value = dict with datasets and Köppen primary class.
MULTI_DATASET_LOOCV_ZONES: dict[str, dict] = {
    "Z1_mediterranean": {
        "koppen": "Csa",
        "hemisphere": "NH",
        "datasets": ["s2agri100", "eurocropml_PT"],
        "cropglobe_countries": [],  # FRA partly, but FRA is also Z2
    },
    "Z2_oceanic": {
        "koppen": "Cfb",
        "hemisphere": "NH",
        "datasets": ["breizhcrops", "pastis"],
        "cropglobe_countries": ["BEL", "GBR", "NLD", "FRA"],
    },
    "Z3_continental": {
        "koppen": "Dfb",
        "hemisphere": "NH",
        "datasets": ["zuericrop", "eurocropml_EE", "eurocropml_LV"],
        "cropglobe_countries": ["CHN"],
    },
    "Z4_southern_hemisphere": {
        "koppen": "BSk/Csb",
        "hemisphere": "SH",
        "datasets": ["sact"],
        "cropglobe_countries": ["ARG", "AUS"],
    },
}


def get_train_countries(hold_out: list[str]) -> list[str]:
    """Return training countries for a given hold-out set."""
    return [c for c in ALL_CROPGLOBE_COUNTRIES if c not in hold_out]

