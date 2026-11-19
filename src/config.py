"""Project-wide configuration. The ONLY place column names are mapped.

Stage 1 leaves COLUMN_MAP empty on purpose: real column names must be read from
the inspection output first (see results/inspection/), then filled in here.
"""
from __future__ import annotations

import os
from pathlib import Path

SEED = 42

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
RESULTS = ROOT / "results"
INSPECTION = RESULTS / "inspection"
FIGURES = ROOT / "figures"

# Base URL is overridable only so the download code can be tested offline.
ZENODO_BASE = os.environ.get("ZENODO_BASE", "https://zenodo.org")

DATASETS = {
    "ashrae": {
        "record_id": "4993952",
        "title": "ASHRAE Global Thermal Comfort Database II v2.1",
        "license": "CC BY 4.0",
        "url": "https://zenodo.org/records/4993952",
    },
    "manikin": {
        "record_id": "20179324",
        "title": "Heat balance of sleeping sets (Newton thermal manikin)",
        "license": "CC BY 4.0",
        "url": "https://zenodo.org/records/20179324",
    },
}

# ---------------------------------------------------------------------------
# Column mapping: the ONLY place real column names appear. Confirmed against the
# stage 1 inspection output (db_measurements_v2.1.0.csv.gz and db_metadata.csv).
# ---------------------------------------------------------------------------
COLUMN_MAP: dict[str, dict[str, str | None]] = {
    "ashrae": {
        # row-level file
        "record_id": "record_id",
        "building_id": "building_id",       # also the GroupKFold group (decision 4)
        "thermal_sensation": "thermal_sensation",
        "thermal_acceptability": "thermal_acceptability",  # sensitivity check only
        "clo": "clo",
        "air_temp": "ta",
        "rel_humidity": "rh",
        "air_speed": "vel",
        "met": "met",
        "mean_radiant_temp": "tr",          # 34.8% missing, used later for PMV only
        "fan": "fan",                       # 85.3% missing, descriptive only
        # building-level metadata file (joined on building_id)
        "building_type": "building_type",
        "climate": "climate",
        "quality_assurance": "quality_assurance",
        "contributor": "contributor",
        "country": "country",
        "city": "city",
    },
    "manikin": {
        # parsed by label, not by column position (see src/manikin.py)
        "files": {
            "supine_table": "Dataset_Tab 1.xlsx",
            "side_table": "Dataset_Tab 2.xlsx",
            "both_positions": "Dataset_Figure 4.xlsx",
            "torso_segments": "Dataset_Figure 3.xlsx",
        },
    },
}

# ---------------------------------------------------------------------------
# Analysis decisions (confirmed by the user after the stage 1 inspection).
# ---------------------------------------------------------------------------
# Explicit class list (replaces the stage 2 regex, which wrongly matched highland/dry classes and missed
# "wet equatorial"). Matched exactly, case-insensitive, on the ashrae `climate` column.
HOT_HUMID_CLASSES = [
    "tropical wet savanna", "tropical savanna", "tropical monsoon", "tropical", "tropical rainforest",
    "wet equatorial", "humid subtropical", "monsoon-influenced humid subtropical",
]
RESIDENTIAL_TYPES = ["multifamily housing"]       # the only residential class in `building_type`
MET_MAX = 1.3                                     # "low activity" stand-in for rest (sedentary limit)
COMFORT_VOTE_RANGE = (-1, 1)                      # comfortable = rounded sensation vote in [-1, +1]
CLO_PER_M2KW = 1 / 0.155                          # 1 clo = 0.155 m2K/W

# Physical plausibility bounds, applied in order after the completeness filters.
RANGE_RULES = {
    "air_temp": (15.0, 40.0),
    "rel_humidity": (10.0, 100.0),
    "air_speed": (0.0, 2.0),
    "clo": (0.1, 2.5),
}
PROCESSED = ROOT / "data" / "processed"

# Keywords used ONLY to suggest candidate columns in the inspection report.
# They are hints for a human, never used to pick columns automatically.
CONCEPT_KEYWORDS: dict[str, list[str]] = {
    "thermal_sensation": ["sensation", "tsv", "vote"],
    "thermal_comfort": ["comfort", "acceptab"],
    "clo": ["clo", "insulation", "clothing"],
    "air_temp": ["air temp", "ta ", "temperature", "(c)", "tdb"],
    "rel_humidity": ["humid", "rh"],
    "air_speed": ["velocity", "air speed", "airspeed", "v_a"],
    "met": ["met", "metabolic"],
    "building_type": ["building type", "building"],
    "climate": ["climate", "koppen", "köppen"],
    "group_id": ["study", "publication", "contributor", "building id", "dataset", "source", "citation"],
    "set_id": ["set", "garment", "pajama", "pyjama", "sleepwear", "ensemble"],
    "posture": ["posture", "position", "supine", "lateral", "prone"],
    "total_resistance": ["total", "rt", "resistance", "insulation", "clo"],
    "torso_resistance": ["torso", "trunk", "chest", "segment"],
}

# ---------------------------------------------------------------------------
# Modelling decisions (stage 3).
# ---------------------------------------------------------------------------
N_SPLITS = 5                  # StratifiedGroupKFold by building_id
N_BOOT = 500                  # cluster bootstrap replicates (resampling buildings)
COMFORT_THRESHOLD = 0.80      # "stays comfortable" = predicted P(comfortable) >= 0.80 (confirmed)
SLEEP_MET = 0.7               # met used for sleep scenarios (ASHRAE 55 value for sleeping)

# Decisions after the stage 3 review (confirmed by the user):
TARGET_KIND = "warm"          # main target: warm-side discomfort = rounded sensation vote >= +1
WARM_P_MAX = 0.20             # scenario rule: "stays comfortable" = predicted P(warm-side discomfort) <= 0.20
PMV_WARM_CUT = 0.5            # theory-based rule: PMV < +0.5 (rounds to a vote below +1)
N_BOOT_SCEN = 200             # building bootstrap refits for scenario uncertainty
