"""ASHRAE subsetting and cleaning with a full audit log.

Every filter is applied sequentially and logged (rows before/after, buildings
remaining) to results/filter_log.csv, so the report can state exactly how many
rows each rule removed.
"""
from __future__ import annotations

import pandas as pd

try:
    from . import config
except ImportError:  # pragma: no cover
    import config  # type: ignore

M = config.COLUMN_MAP["ashrae"]


def load_ashrae() -> pd.DataFrame:
    folder = config.DATA_RAW / "ashrae"
    meas_path = next(folder.glob("db_measurements*.csv.gz"))
    meta_path = folder / "db_metadata.csv"
    meas = pd.read_csv(meas_path, low_memory=False)
    meta = pd.read_csv(meta_path)
    meta_cols = [M[k] for k in ("building_id", "building_type", "climate", "quality_assurance",
                                "contributor", "country", "city")]
    df = meas.merge(meta[meta_cols], on=M["building_id"], how="left", validate="many_to_one")
    unmatched = int(df[M["building_type"]].isna().sum())
    print(f"Joined metadata on {M['building_id']}: {len(df):,} rows, {unmatched:,} without metadata")
    return df


class FilterLog:
    def __init__(self, df: pd.DataFrame):
        self.rows: list[dict] = []
        self._add("0. raw measurements joined with metadata", "none", len(df), len(df), df)

    def _add(self, step, rule, before, after, df):
        self.rows.append({
            "step": step, "rule": rule, "rows_before": before, "rows_after": after,
            "rows_dropped": before - after,
            "buildings_after": int(df[M["building_id"]].nunique()) if len(df) else 0,
        })

    def apply(self, df: pd.DataFrame, mask: pd.Series, step: str, rule: str) -> pd.DataFrame:
        out = df[mask.fillna(False)]
        self._add(step, rule, len(df), len(out), out)
        return out

    def table(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows)


def _is_hot_humid(x) -> bool:
    return isinstance(x, str) and x.strip().lower() in config.HOT_HUMID_CLASSES


def climate_report(df: pd.DataFrame) -> pd.DataFrame:
    """Rows and buildings per climate class, flagged by the hot-humid rule."""
    g = df.groupby(M["climate"]).agg(rows=(M["record_id"], "size"),
                                      buildings=(M["building_id"], "nunique")).reset_index()
    g["matches_hot_humid_rule"] = g[M["climate"]].map(_is_hot_humid)
    return g.sort_values("rows", ascending=False)


def build_subset(df: pd.DataFrame):
    """Return (clean_df, filter_log_table). Order matters and is documented in the log."""
    log = FilterLog(df)
    d = df

    d = log.apply(d, d[M["quality_assurance"]].eq("pass"), "1. quality assurance",
                  "metadata quality_assurance == 'pass'")
    d = log.apply(d, d[M["climate"]].map(_is_hot_humid),
                  "2. hot-humid climate", "climate in explicit hot-humid class list (config.HOT_HUMID_CLASSES)")
    d = log.apply(d, d[M["building_type"]].isin(config.RESIDENTIAL_TYPES), "3. residential",
                  f"building_type in {config.RESIDENTIAL_TYPES}")
    d = log.apply(d, d[M["met"]] <= config.MET_MAX, "4. low metabolic rate",
                  f"met <= {config.MET_MAX} (rows with missing met are dropped)")

    for key in ("thermal_sensation", "air_temp", "rel_humidity", "air_speed", "clo"):
        col = M[key]
        d = log.apply(d, d[col].notna(), f"5. complete cases: {col}", f"{col} is not missing")

    for key, (lo, hi) in config.RANGE_RULES.items():
        col = M[key]
        d = log.apply(d, d[col].between(lo, hi), f"6. plausible range: {col}", f"{lo} <= {col} <= {hi}")

    d = d.copy()
    d["vote_rounded"] = d[M["thermal_sensation"]].round().clip(-3, 3).astype(int)
    lo, hi = config.COMFORT_VOTE_RANGE
    d["comfortable"] = d["vote_rounded"].between(lo, hi).astype(int)
    return d, log.table()


def met_sensitivity(df: pd.DataFrame, cutoffs=(1.0, 1.2, 1.3, 1.5)) -> pd.DataFrame:
    """Rows left after climate+residential+met filters for alternative met cutoffs."""
    base = df[df[M["climate"]].map(_is_hot_humid)
              & df[M["building_type"]].isin(config.RESIDENTIAL_TYPES)
              & df[M["quality_assurance"]].eq("pass")]
    rows = [{"met_max": c, "rows": int((base[M["met"]] <= c).sum())} for c in cutoffs]
    rows.append({"met_max": "any (met not required)", "rows": len(base)})
    return pd.DataFrame(rows)


def save_processed(d: pd.DataFrame) -> str:
    config.PROCESSED.mkdir(parents=True, exist_ok=True)
    keep = [M[k] for k in ("record_id", "building_id", "thermal_sensation", "thermal_acceptability", "clo",
                           "air_temp", "rel_humidity", "air_speed", "met", "mean_radiant_temp", "fan",
                           "building_type", "climate", "contributor", "country", "city")]
    keep += ["vote_rounded", "comfortable"]
    path = config.PROCESSED / "ashrae_clean.csv.gz"
    d[keep].to_csv(path, index=False)
    return str(path)
