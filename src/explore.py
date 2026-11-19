"""Exploratory figures for the cleaned ASHRAE subset and the manikin clo values.

Every figure is saved as PNG (300 dpi) and SVG into figures/.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    from . import config
except ImportError:  # pragma: no cover
    import config  # type: ignore

M = config.COLUMN_MAP["ashrae"]
MIN_BIN = 30  # bins with fewer rows are not plotted

plt.rcParams.update({
    "figure.dpi": 100, "savefig.dpi": 300, "font.size": 10, "axes.titlesize": 11,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.alpha": 0.25, "font.family": "DejaVu Sans",
})
C = {"main": "#1f5f8b", "alt": "#c4572b", "grey": "#6b7280"}


def save(fig, name: str) -> None:
    config.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(config.FIGURES / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(config.FIGURES / f"{name}.svg", bbox_inches="tight")
    plt.close(fig)


def _binned_share(d: pd.DataFrame, xcol: str, bins) -> pd.DataFrame:
    b = pd.cut(d[xcol], bins)
    g = d.groupby(b, observed=True)["comfortable"].agg(["mean", "size"]).reset_index()
    g["x"] = g[xcol].apply(lambda i: i.mid)
    g["se"] = np.sqrt(g["mean"] * (1 - g["mean"]) / g["size"])
    return g[g["size"] >= MIN_BIN]


def fig_vote_distribution(d):
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    vc = d["vote_rounded"].value_counts().reindex(range(-3, 4), fill_value=0)
    ax.bar(vc.index, vc.values, color=C["main"])
    ax.axvspan(-1.5, 1.5, color=C["alt"], alpha=0.08)
    ax.set_xlabel("Thermal sensation vote (-3 cold ... +3 hot)")
    ax.set_ylabel("Number of votes")
    ax.set_title(f"Votes in hot-humid residential subset (n = {len(d):,})")
    save(fig, "fig01_vote_distribution")


def fig_comfort_vs_clo(d):
    g = _binned_share(d, M["clo"], np.arange(0.1, 1.6, 0.15))
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.errorbar(g["x"], 100 * g["mean"], yerr=100 * 1.96 * g["se"], marker="o", color=C["main"], capsize=3)
    ax.set_xlabel("Clothing insulation (clo)")
    ax.set_ylabel("Votes within -1 to +1 (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Comfort share by clothing insulation (95% CI)")
    save(fig, "fig02_comfort_vs_clo")


def fig_comfort_vs_temp_by_speed(d):
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    for label, mask, col in (("Air speed < 0.2 m/s", d[M["air_speed"]] < 0.2, C["main"]),
                             ("Air speed >= 0.2 m/s", d[M["air_speed"]] >= 0.2, C["alt"])):
        g = _binned_share(d[mask], M["air_temp"], np.arange(18, 38, 1.5))
        if len(g):
            ax.errorbar(g["x"], 100 * g["mean"], yerr=100 * 1.96 * g["se"], marker="o", capsize=3,
                        color=col, label=f"{label} (n = {int(mask.sum()):,})")
    ax.set_xlabel("Air temperature (degC)")
    ax.set_ylabel("Votes within -1 to +1 (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False)
    ax.set_title("Comfort share by air temperature and air speed")
    save(fig, "fig03_comfort_vs_temp_by_airspeed")


def fig_sensation_heatmap(d):
    tb = pd.cut(d[M["air_temp"]], np.arange(20, 38, 2))
    rb = pd.cut(d[M["rel_humidity"]], np.arange(20, 101, 16))
    piv = d.assign(t=tb, r=rb).pivot_table(index="r", columns="t", values=M["thermal_sensation"],
                                           aggfunc="mean", observed=True)
    cnt = d.assign(t=tb, r=rb).pivot_table(index="r", columns="t", values=M["thermal_sensation"],
                                           aggfunc="size", observed=True)
    piv = piv.where(cnt >= MIN_BIN)
    fig, ax = plt.subplots(figsize=(6.2, 3.9))
    im = ax.imshow(piv.values, origin="lower", aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    ax.set_xticks(range(piv.shape[1]), [f"{i.left:.0f}-{i.right:.0f}" for i in piv.columns])
    ax.set_yticks(range(piv.shape[0]), [f"{i.left:.0f}-{i.right:.0f}" for i in piv.index])
    ax.grid(False)
    ax.set_xlabel("Air temperature (degC)")
    ax.set_ylabel("Relative humidity (%)")
    ax.set_title(f"Mean sensation vote (cells with n >= {MIN_BIN})")
    fig.colorbar(im, ax=ax, label="Mean vote")
    save(fig, "fig04_sensation_heatmap_temp_rh")


def fig_manikin_clo(tbl: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    sets = list(tbl["set"].cat.categories)
    w = 0.38
    for k, (post, col, lab) in enumerate((("supine", C["main"], "On the back (supine)"),
                                          ("side", C["alt"], "On the left side"))):
        t = tbl[tbl["posture"] == post].set_index("set").loc[sets]
        ax.bar(np.arange(len(sets)) + (k - 0.5) * w, t["clo_total"], w, yerr=t["clo_uncertainty"],
               capsize=3, color=col, label=lab)
    ax.set_xticks(range(len(sets)), sets)
    ax.set_ylabel("Total insulation (clo)")
    ax.set_title("Thermal manikin: total insulation of sleeping sets")
    ax.legend(frameon=False)
    save(fig, "fig05_manikin_total_clo")


def fig_clo_coverage(d: pd.DataFrame, tbl: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    ax.hist(d[M["clo"]], bins=np.arange(0.1, 2.6, 0.1), color=C["grey"], alpha=0.8)
    ax.axvspan(tbl["clo_total"].min(), tbl["clo_total"].max(), color=C["alt"], alpha=0.15,
               label="Manikin sleeping-set range")
    ax.set_xlabel("Clothing insulation (clo)")
    ax.set_ylabel("ASHRAE votes in subset")
    ax.set_title("Training-data clo range vs manikin sleeping-set clo")
    ax.legend(frameon=False)
    save(fig, "fig06_clo_range_coverage")


def run_all(d: pd.DataFrame, tbl: pd.DataFrame) -> None:
    fig_vote_distribution(d)
    fig_comfort_vs_clo(d)
    fig_comfort_vs_temp_by_speed(d)
    fig_sensation_heatmap(d)
    fig_manikin_clo(tbl)
    fig_clo_coverage(d, tbl)


# ----------------------------------------------------------------------------- stage 3 figures
def fig_roc(oof: pd.DataFrame) -> None:
    from sklearn.metrics import roc_auc_score, roc_curve
    y = oof["y"].to_numpy()
    series = [("PMV (ISO 7730)", -np.abs(oof["pmv_iso"]), C["grey"]),
              ("PMV (ASHRAE 55, cooling effect)", -np.abs(oof["pmv_ashrae"]), "#9aa3ad"),
              ("Logistic (quadratic)", oof["p_comfort__Logistic regression (quadratic)"], C["main"]),
              ("XGBoost", oof["p_comfort__XGBoost"], C["alt"])]
    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    for name, sc, col in series:
        fpr, tpr, _ = roc_curve(y, sc)
        ax.plot(fpr, tpr, color=col, label=f"{name} (AUC {roc_auc_score(y, sc):.2f})")
    ax.plot([0, 1], [0, 1], "--", color="black", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Comfort classification, held-out buildings")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, "fig07_roc_curves")


def fig_calibration(oof: pd.DataFrame) -> None:
    from sklearn.calibration import calibration_curve
    y = oof["y"].to_numpy()
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    for name, col in (("Logistic (quadratic)", C["main"]), ("XGBoost", C["alt"])):
        key = "p_comfort__Logistic regression (quadratic)" if "Logistic" in name else "p_comfort__XGBoost"
        frac, mean_p = calibration_curve(y, oof[key], n_bins=8, strategy="quantile")
        ax.plot(100 * mean_p, 100 * frac, marker="o", color=col, label=name)
    ax.plot([40, 100], [40, 100], "--", color="black", lw=0.8)
    ax.set_xlabel("Predicted probability comfortable (%)")
    ax.set_ylabel("Observed share comfortable (%)")
    ax.set_title("Calibration on held-out buildings")
    ax.legend(frameon=False)
    save(fig, "fig08_calibration")


def fig_roc_warm(oof: pd.DataFrame) -> None:
    from sklearn.metrics import roc_auc_score, roc_curve
    y = oof["y"].to_numpy()
    series = [("PMV (ISO 7730)", oof["pmv_iso"], C["grey"]),
              ("PMV (ASHRAE 55, cooling effect)", oof["pmv_ashrae"], "#9aa3ad"),
              ("Logistic regression", oof["p_comfort__Logistic regression (linear)"], C["main"]),
              ("XGBoost", oof["p_comfort__XGBoost"], C["alt"])]
    fig, ax = plt.subplots(figsize=(5.0, 4.6))
    for name, sc_, col in series:
        fpr, tpr, _ = roc_curve(y, sc_)
        ax.plot(fpr, tpr, color=col, label=f"{name} (AUC {roc_auc_score(y, sc_):.2f})")
    ax.plot([0, 1], [0, 1], "--", color="black", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("Warm-side discomfort, held-out buildings")
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    save(fig, "fig07_roc_curves_warm")


def fig_calibration_warm(oof: pd.DataFrame) -> None:
    from sklearn.calibration import calibration_curve
    y = oof["y"].to_numpy()
    fig, ax = plt.subplots(figsize=(4.8, 4.6))
    for name, key, col in (("Logistic regression", "p_comfort__Logistic regression (linear)", C["main"]),
                           ("XGBoost", "p_comfort__XGBoost", C["alt"])):
        frac, mean_p = calibration_curve(y, oof[key], n_bins=8, strategy="quantile")
        ax.plot(100 * mean_p, 100 * frac, marker="o", color=col, label=name)
    ax.plot([0, 70], [0, 70], "--", color="black", lw=0.8)
    ax.set_xlabel("Predicted warm-side discomfort (%)")
    ax.set_ylabel("Observed share with vote >= +1 (%)")
    ax.set_title("Calibration on held-out buildings")
    ax.legend(frameon=False)
    save(fig, "fig08_calibration_warm")
