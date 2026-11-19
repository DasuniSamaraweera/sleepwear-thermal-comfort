"""Parse the manikin workbooks into tidy tables, by label rather than position.

Dataset: Heat balance of sleeping sets (Newton thermal manikin).
The 'sets' are bedding configurations (Without cover, Cotton cover, Blanket,
Summer duvet), measured on the back (supine) and on the left side. They are NOT
garment types; the README does not describe the garments worn.
"""
from __future__ import annotations

import re

import pandas as pd

try:
    from . import config
except ImportError:  # pragma: no cover
    import config  # type: ignore

FILES = config.COLUMN_MAP["manikin"]["files"]
SET_ORDER = ["Without cover", "Cotton cover", "Blanket", "Summer duvet"]


def _find(name: str):
    root = config.DATA_RAW / "manikin"
    hits = list(root.rglob(name))
    if not hits:
        raise FileNotFoundError(f"{name} not found under {root}")
    return hits[0]


def _raw(name: str) -> pd.DataFrame:
    return pd.read_excel(_find(name), header=None)


def _norm_set(label: str) -> str:
    s = re.sub(r"^\s*\d+\)\s*", "", str(label)).strip()
    s = s.replace("duver", "duvet")  # typo present in the source file
    return s[:1].upper() + s[1:]


def parse_table(key: str, posture: str) -> pd.DataFrame:
    """Tab 1 (supine) or Tab 2 (side): replicate and mean resistances per set."""
    raw = _raw(FILES[key])
    labels = raw.iloc[:, 0].fillna("").astype(str).str.strip()
    set_row = next(i for i in range(len(raw)) if raw.iloc[i, 1:].fillna("").astype(str).str.contains("cover|Blanket|duvet", case=False).any())
    set_cols = [c for c in range(1, raw.shape[1]) if pd.notna(raw.iloc[set_row, c])]
    out = []
    for c in set_cols:
        name = _norm_set(raw.iloc[set_row, c])
        for i, lab in enumerate(labels):
            if re.fullmatch(r"Rtot\d*|UR", lab):
                out.append({"set": name, "posture": posture, "stat": lab,
                            "m2K_per_W": float(raw.iloc[i, c]), "clo": float(raw.iloc[i, c + 1])})
    return pd.DataFrame(out)


def parse_figure4() -> pd.DataFrame:
    raw = _raw(FILES["both_positions"])
    rows = []
    for i in range(len(raw)):
        lab = str(raw.iloc[i, 0])
        if re.match(r"^\s*\d+\)", lab):
            rows.append({"set": _norm_set(lab),
                         "side_clo": float(raw.iloc[i, 1]), "side_err": float(raw.iloc[i, 2]),
                         "supine_clo": float(raw.iloc[i, 3]), "supine_err": float(raw.iloc[i, 4])})
    return pd.DataFrame(rows)


def parse_segments() -> pd.DataFrame:
    raw = _raw(FILES["torso_segments"])
    out, posture, cols = [], None, None
    for i in range(len(raw)):
        c1 = str(raw.iloc[i, 1])
        if "Position" in c1:
            posture = "side" if "side" in c1.lower() else "supine"
        if str(raw.iloc[i, 2]) in ("Chest", "Abdomen", "Back"):
            cols = [str(raw.iloc[i, k]) for k in (2, 3, 4)]
        lab = str(raw.iloc[i, 1])
        if re.match(r"^\s*\d+\)", lab) and posture and cols:
            for k, seg in zip((2, 3, 4), cols):
                out.append({"set": _norm_set(lab), "posture": posture, "segment": seg,
                            "clo": float(raw.iloc[i, k])})
    seg = pd.DataFrame(out)
    seg["suspect_artifact"] = seg["clo"] > 5  # >5 clo is not physical for a torso segment
    return seg


def build_clo_table() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (clo_table, replicate_table, segments) and print the cross-checks."""
    reps = pd.concat([parse_table("supine_table", "supine"), parse_table("side_table", "side")],
                     ignore_index=True)
    mean = reps[reps["stat"] == "Rtot"].rename(columns={"clo": "clo_total", "m2K_per_W": "rtot_m2K_per_W"})
    ur = reps[reps["stat"] == "UR"].rename(columns={"clo": "clo_uncertainty"})[["set", "posture", "clo_uncertainty"]]
    tbl = mean.drop(columns="stat").merge(ur, on=["set", "posture"])
    tbl["set"] = pd.Categorical(tbl["set"], SET_ORDER, ordered=True)
    tbl = tbl.sort_values(["posture", "set"]).reset_index(drop=True)

    f4 = parse_figure4()
    chk = []
    for _, r in f4.iterrows():
        for post in ("supine", "side"):
            t = tbl[(tbl["set"] == r["set"]) & (tbl["posture"] == post)]["clo_total"].iloc[0]
            chk.append({"set": r["set"], "posture": post, "table_clo": t, "figure4_clo": r[f"{post}_clo"],
                        "abs_diff": abs(t - r[f"{post}_clo"])})
    chk = pd.DataFrame(chk)
    print("Cross-check Tab 1/2 vs Figure 4 (max abs diff = %.3f clo)" % chk["abs_diff"].max())
    unit = (tbl["rtot_m2K_per_W"] * config.CLO_PER_M2KW - tbl["clo_total"]).abs().max()
    print("Unit check: max |m2K/W / 0.155 - clo| = %.3f clo" % unit)
    return tbl, reps, parse_segments()
