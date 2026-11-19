"""Inspect every raw table: real column names, dtypes, missing counts, ranges.

Nothing here assumes a column name. Outputs go to results/inspection/ so they
can be reviewed (and sent back) before any cleaning or modelling code is written.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from . import config
except ImportError:  # pragma: no cover
    import config  # type: ignore

TABLE_SUFFIXES = {".csv", ".gz", ".xlsx", ".xls", ".tsv"}
TEXT_SUFFIXES = {".txt", ".md", ".json", ".rst"}
NA_LIKE = {"", "na", "n/a", "nan", "none", "null", "-", "--", "?", "NA"}


def _safe_name(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", s).strip("_")


def find_files(folder: Path) -> list[Path]:
    skip = {"record_metadata.json"}
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.name not in skip)


def _read_csv(path: Path) -> pd.DataFrame:
    sep = "\t" if ".tsv" in path.name.lower() else ","
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return pd.read_csv(path, encoding=enc, low_memory=False, sep=sep)
        except UnicodeDecodeError:
            continue
    raise RuntimeError(f"Could not decode {path}")


def load_tables(folder: Path) -> dict[str, pd.DataFrame]:
    """Return {label: DataFrame} for every csv/xlsx table under folder."""
    tables: dict[str, pd.DataFrame] = {}
    for p in find_files(folder):
        suf = p.suffix.lower()
        rel = str(p.relative_to(folder))
        if suf in {".csv", ".tsv"} or (suf == ".gz" and p.name.lower().endswith((".csv.gz", ".tsv.gz"))):
            tables[rel] = _read_csv(p)
        elif suf in {".xlsx", ".xls"}:
            xl = pd.ExcelFile(p)
            for sh in xl.sheet_names:
                tables[f"{rel}::{sh}"] = xl.parse(sh)
    return tables


def profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n = len(df)
    for col in df.columns:
        s = df[col]
        r = {
            "column": col,
            "dtype": str(s.dtype),
            "n_rows": n,
            "n_missing": int(s.isna().sum()),
            "pct_missing": round(100 * s.isna().mean(), 2) if n else np.nan,
            "n_unique": int(s.nunique(dropna=True)),
        }
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            r.update(min=s.min(), max=s.max(), mean=round(float(s.mean()), 4) if s.notna().any() else np.nan,
                     median=s.median())
        else:
            as_str = s.dropna().astype(str)
            r["na_like_strings"] = int(as_str.str.strip().str.lower().isin(NA_LIKE).sum())
            num = pd.to_numeric(as_str, errors="coerce")
            r["numeric_like_share"] = round(float(num.notna().mean()), 3) if len(as_str) else np.nan
        if r["n_unique"] <= 15 or not pd.api.types.is_numeric_dtype(s):
            vc = s.value_counts(dropna=True).head(6)
            r["top_values"] = "; ".join(f"{k} ({v})" for k, v in vc.items())
        rows.append(r)
    return pd.DataFrame(rows)


def raw_preview(path: Path, nrows: int = 15) -> str:
    """First rows with NO header interpretation: reveals unit rows / multi-row headers."""
    suf = path.suffix.lower()
    try:
        if suf in {".xlsx", ".xls"}:
            xl = pd.ExcelFile(path)
            parts = []
            for sh in xl.sheet_names:
                d = xl.parse(sh, header=None, nrows=nrows)
                parts.append(f"--- sheet '{sh}' (raw, header=None) ---\n{d.to_string(max_cols=20)}")
            return "\n".join(parts)
        if suf in {".csv", ".tsv"}:
            d = pd.read_csv(path, header=None, nrows=nrows, encoding="latin-1", sep=None, engine="python")
            return d.to_string(max_cols=20)
    except Exception as e:  # preview is best effort
        return f"(preview failed: {e})"
    return ""


def text_files(folder: Path, limit: int = 4000) -> dict[str, str]:
    out = {}
    for p in find_files(folder):
        if p.suffix.lower() in TEXT_SUFFIXES:
            out[str(p.relative_to(folder))] = p.read_text(errors="replace")[:limit]
    return out


def suggest_candidates(tables: dict[str, pd.DataFrame], concepts: list[str]) -> pd.DataFrame:
    rows = []
    for concept in concepts:
        kws = [k.lower() for k in config.CONCEPT_KEYWORDS[concept]]
        for label, df in tables.items():
            for col in df.columns:
                c = f"{str(col).lower()} "
                if any(k in c for k in kws):
                    rows.append({"concept": concept, "table": label, "column": col})
    return pd.DataFrame(rows)


def inspect_dataset(key: str) -> dict[str, pd.DataFrame]:
    folder = config.DATA_RAW / key
    out = config.INSPECTION / key
    out.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 250, "display.max_columns", 40, "display.max_colwidth", 80)

    print("=" * 100)
    print(f"DATASET: {key}  |  {config.DATASETS[key]['title']}")
    print("=" * 100)
    files = find_files(folder)
    print("Files on disk:")
    for p in files:
        print(f"  {p.relative_to(folder)}  ({p.stat().st_size:,} bytes)")

    for name, txt in text_files(folder).items():
        print(f"\n--- text file: {name} (first 4000 chars) ---\n{txt}")
        (out / f"textfile__{_safe_name(name)}.txt").write_text(txt)

    for p in files:
        if p.suffix.lower() in {".xlsx", ".xls", ".csv", ".tsv"} and p.stat().st_size < 5_000_000:
            prev = raw_preview(p)
            print(f"\n--- RAW PREVIEW: {p.name} ---\n{prev}")
            (out / f"rawpreview__{_safe_name(p.name)}.txt").write_text(prev)

    tables = load_tables(folder)
    for label, df in tables.items():
        print(f"\n### TABLE {label}: {df.shape[0]:,} rows x {df.shape[1]} columns")
        prof = profile(df)
        print(prof.drop(columns=["n_rows"]).to_string(index=False))
        prof.to_csv(out / f"columns__{_safe_name(label)}.csv", index=False)
        df.head(10).to_csv(out / f"head__{_safe_name(label)}.csv", index=False)
        dups = int(df.duplicated().sum())
        print(f"Fully duplicated rows: {dups:,}")
    return tables


def write_candidates(all_tables: dict[str, dict[str, pd.DataFrame]]) -> pd.DataFrame:
    parts = []
    for key, concepts in (("ashrae", ["thermal_sensation", "thermal_comfort", "clo", "air_temp", "rel_humidity",
                                      "air_speed", "met", "building_type", "climate", "group_id"]),
                          ("manikin", ["set_id", "posture", "total_resistance", "torso_resistance"])):
        cand = suggest_candidates(all_tables.get(key, {}), concepts)
        if len(cand):
            cand.insert(0, "dataset", key)
            parts.append(cand)
    res = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    config.INSPECTION.mkdir(parents=True, exist_ok=True)
    res.to_csv(config.INSPECTION / "candidate_columns.csv", index=False)
    return res


if __name__ == "__main__":
    tabs = {k: inspect_dataset(k) for k in config.DATASETS}
    print(write_candidates(tabs).to_string(index=False))
