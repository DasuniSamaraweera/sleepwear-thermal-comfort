"""What-if scenarios: predicted warm-side discomfort across total clo, temperature, RH and air speed.

Insulation axis: ASHRAE `clo` is clothing only, while the manikin totals include bedding. Scenarios therefore use
TOTAL clo = garment clo + manikin cover increment (cover set minus 'Without cover', same posture). Garment clo is
a grid (quartiles of the data), not named garments: no garment clo values exist in the datasets.

Rules: ML "stays comfortable" = predicted P(warm-side discomfort) <= config.WARM_P_MAX (0.20);
PMV "stays comfortable" = PMV < config.PMV_WARM_CUT (+0.5). Sleep met = config.SLEEP_MET.
"""
from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone

try:
    from . import config, explore, models
except ImportError:  # pragma: no cover
    import config, explore, models  # type: ignore

F = models.FEATURES  # [clo, ta, rh, vel, met]
CLO_GRID = np.round(np.arange(0.10, 2.501, 0.01), 2)


def frame(clo, ta, rh, vel, met=None) -> pd.DataFrame:
    met = config.SLEEP_MET if met is None else met
    n = max(len(np.atleast_1d(a)) for a in (clo, ta, rh, vel, met))
    return pd.DataFrame({F[0]: np.broadcast_to(clo, n), F[1]: np.broadcast_to(ta, n), F[2]: np.broadcast_to(rh, n),
                         F[3]: np.broadcast_to(vel, n), F[4]: np.broadcast_to(met, n)}).astype(float)


def p_warm(model, X: pd.DataFrame) -> np.ndarray:
    return model.predict_proba(X[F].to_numpy(float))[:, 1]


def pmv(X: pd.DataFrame) -> pd.DataFrame:
    from pythermalcomfort.models import pmv_ppd_ashrae, pmv_ppd_iso
    from pythermalcomfort.utilities import v_relative
    ta = X[F[1]].to_numpy(float)
    vr = np.asarray(v_relative(v=X[F[3]].to_numpy(float), met=X[F[4]].to_numpy(float)), float)
    kw = dict(tdb=ta, tr=ta, vr=vr, rh=X[F[2]].to_numpy(float), met=X[F[4]].to_numpy(float),
              clo=X[F[0]].to_numpy(float), limit_inputs=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        iso, ash = pmv_ppd_iso(**kw), pmv_ppd_ashrae(**kw)
    return pd.DataFrame({"pmv_iso": np.asarray(iso.pmv, float), "pmv_ashrae": np.asarray(ash.pmv, float)})


def fan_levels(df: pd.DataFrame) -> dict:
    """Air speeds for 'no fan' / 'fan': median measured speed in rows where the fan flag is 0 / 1."""
    f = df[config.COLUMN_MAP["ashrae"]["fan"]]
    v = df[config.COLUMN_MAP["ashrae"]["air_speed"]]
    return {"no fan": float(v[f == 0].median()), "fan": float(v[f == 1].median()),
            "n_no_fan": int((f == 0).sum()), "n_fan": int((f == 1).sum())}


def cover_increments(clo_tbl: pd.DataFrame) -> pd.DataFrame:
    t = clo_tbl.copy()
    base = t[t["set"] == "Without cover"].set_index("posture")["clo_total"]
    t["increment"] = t.apply(lambda r: r["clo_total"] - base[r["posture"]], axis=1)
    return t[["set", "posture", "clo_total", "increment"]]


def highest_ok(clo_grid, ok: np.ndarray) -> float:
    """Highest clo such that the rule holds at EVERY clo from the grid start up to it (nan if it fails at the start)."""
    bad = np.flatnonzero(~ok)
    if len(bad) == 0:
        return float(clo_grid[-1])
    if bad[0] == 0:
        return float("nan")
    return float(clo_grid[bad[0] - 1])


def curve_table(models_by_name: dict, ta, rh, vel, label) -> pd.DataFrame:
    X = frame(CLO_GRID, ta, rh, vel)
    out = pd.DataFrame({"clo_total": CLO_GRID, "condition": label})
    for n, m in models_by_name.items():
        out[f"p_warm__{n}"] = p_warm(m, X)
    out = pd.concat([out, pmv(X)], axis=1)
    return out


def limits(curves: pd.DataFrame, primary: str, support_max: float) -> dict:
    r = {}
    for col, rule in ((f"p_warm__{primary}", lambda c: c <= config.WARM_P_MAX),
                      ("p_warm__XGBoost", lambda c: c <= config.WARM_P_MAX),
                      ("pmv_iso", lambda c: c < config.PMV_WARM_CUT),
                      ("pmv_ashrae", lambda c: c < config.PMV_WARM_CUT)):
        if col in curves:
            h = highest_ok(curves["clo_total"].to_numpy(), rule(curves[col].to_numpy()))
            r[col] = {"highest_clo": h, "beyond_data_range": bool(h > support_max) if h == h else False}
    return r


def bootstrap_limit(df: pd.DataFrame, primary_est, scen: dict, n_boot=config.N_BOOT_SCEN) -> dict:
    """Refit the primary model on building-resampled data; return percentile bands of the P(warm) curves and of the
    highest-comfortable-clo limit for each scenario in `scen` = {label: (ta, rh, vel)}."""
    rng = np.random.default_rng(config.SEED)
    ids = df[models.GROUP].unique()
    where = {g: np.flatnonzero(df[models.GROUP].to_numpy() == g) for g in ids}
    X, y = df[F].to_numpy(float), df[models.Y].to_numpy()
    curves = {k: [] for k in scen}
    lim = {k: [] for k in scen}
    for _ in range(n_boot):
        pick = rng.choice(ids, size=len(ids), replace=True)
        idx = np.concatenate([where[g] for g in pick])
        m = clone(primary_est).fit(X[idx], y[idx])
        for k, (ta, rh, vel) in scen.items():
            p = m.predict_proba(frame(CLO_GRID, ta, rh, vel)[F].to_numpy(float))[:, 1]
            curves[k].append(p)
            lim[k].append(highest_ok(CLO_GRID, p <= config.WARM_P_MAX))
    res = {}
    for k in scen:
        c = np.array(curves[k]); l = np.array(lim[k], float)
        res[k] = {"lo": np.percentile(c, 2.5, 0), "hi": np.percentile(c, 97.5, 0),
                  "limit_median": float(np.nanmedian(l)) if np.isfinite(l).any() else float("nan"),
                  "limit_lo": float(np.nanpercentile(l, 2.5)) if np.isfinite(l).any() else float("nan"),
                  "limit_hi": float(np.nanpercentile(l, 97.5)) if np.isfinite(l).any() else float("nan"),
                  "share_none": float(np.mean(~np.isfinite(l))), "share_at_grid_top": float(np.mean(l >= CLO_GRID[-1]))}
    return res


def support(df: pd.DataFrame, ta, rh, vel) -> dict:
    M = config.COLUMN_MAP["ashrae"]
    near = (df[M["air_temp"]].sub(ta).abs() <= 1) & (df[M["rel_humidity"]].sub(rh).abs() <= 7)
    return {"rows_within_1C_and_7pctRH": int(near.sum()),
            "rows_within_1C_7pctRH_and_vel_within_0.2": int((near & df[M["air_speed"]].sub(vel).abs().le(0.2)).sum()),
            "rows_met_le_0.8": int((df[M["met"]] <= 0.8).sum())}


# ------------------------------------------------------------------ figures
def fig_curves(curves: dict, boot: dict, primary: str, support_p99: float, fan: dict):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9))
    cols = {"no fan": explore.C["main"], "fan": explore.C["alt"]}
    ax = axes[0]
    for lab, c in curves.items():
        col = cols[lab]
        ax.plot(c["clo_total"], 100 * c[f"p_warm__{primary}"], color=col, label=f"{lab} ({fan[lab]:.2f} m/s)")
        ax.fill_between(c["clo_total"], 100 * boot[lab]["lo"], 100 * boot[lab]["hi"], color=col, alpha=0.15)
    ax.axhline(100 * config.WARM_P_MAX, ls="--", color="black", lw=0.8)
    ax.axvspan(support_p99, CLO_GRID[-1], color="grey", alpha=0.12)
    ax.set_xlim(0.1, 2.2); ax.set_ylim(0, 100)
    ax.set_xlabel("Total insulation, garment + bedding (clo)")
    ax.set_ylabel("Predicted warm-side discomfort (%)")
    ax.set_title("Data-driven model (95% band over buildings)")
    ax.text(support_p99 + 0.03, 92, "beyond 99th pct\nof training clo", fontsize=7, va="top")
    ax.legend(frameon=False, loc="upper left")
    ax = axes[1]
    for lab, c in curves.items():
        ax.plot(c["clo_total"], c["pmv_iso"], color=cols[lab], label=f"{lab} ISO 7730")
        ax.plot(c["clo_total"], c["pmv_ashrae"], color=cols[lab], ls=":", label=f"{lab} ASHRAE 55")
    ax.axhline(config.PMV_WARM_CUT, ls="--", color="black", lw=0.8)
    ax.set_xlim(0.1, 2.2)
    ax.set_xlabel("Total insulation, garment + bedding (clo)")
    ax.set_ylabel("PMV (+3 hot)")
    ax.set_title("PMV theory (not fitted to these votes)")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.suptitle("29 degC, 85% RH, sleeping met", y=1.02, fontsize=11)
    fig.tight_layout()
    explore.save(fig, "fig12_scenario_curves_29C_85RH")


def fig_heatmaps(model, fan: dict, clo_ref: float):
    ta = np.arange(25, 34.01, 0.5); rh = np.arange(50, 90.01, 2.5)
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    for ax, lab in zip(axes, ("no fan", "fan")):
        T, R = np.meshgrid(ta, rh)
        p = p_warm(model, frame(clo_ref, T.ravel(), R.ravel(), fan[lab])).reshape(T.shape)
        im = ax.pcolormesh(T, R, 100 * p, cmap="RdBu_r", vmin=0, vmax=60, shading="auto")
        ax.contour(T, R, p, levels=[config.WARM_P_MAX], colors="black", linewidths=1.2)
        ax.plot(29, 85, "k*", ms=10)
        ax.set_xlabel("Air temperature (degC)")
        ax.set_title(f"{lab} ({fan[lab]:.2f} m/s)")
    axes[0].set_ylabel("Relative humidity (%)")
    fig.colorbar(im, ax=axes, label="Predicted warm-side discomfort (%)")
    fig.suptitle(f"Garment clo {clo_ref:.2f}; black line = 20% rule; star = 29 degC / 85% RH", y=1.02, fontsize=11)
    explore.save(fig, "fig13_scenario_heatmaps_temp_rh")


def fig_sets(tab: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    sets = list(tab["set"].unique())
    for ax, lab in zip(axes, ("no fan", "fan")):
        w = 0.38
        for k, (post, col) in enumerate((("supine", explore.C["main"]), ("side", explore.C["alt"]))):
            t = tab[(tab["condition"] == lab) & (tab["posture"] == post)].set_index("set").loc[sets]
            ax.bar(np.arange(len(sets)) + (k - 0.5) * w, 100 * t["p_warm_primary"], w, color=col,
                   label="Back (supine)" if post == "supine" else "Left side")
        ax.axhline(100 * config.WARM_P_MAX, ls="--", color="black", lw=0.8)
        ax.set_xticks(range(len(sets)), sets, rotation=15)
        ax.set_title(lab)
    axes[0].set_ylabel("Predicted warm-side discomfort (%)")
    axes[0].legend(frameon=False)
    fig.suptitle("29 degC, 85% RH, typical garment clo plus manikin cover increment", y=1.02, fontsize=11)
    fig.tight_layout()
    explore.save(fig, "fig14_scenario_bedding_sets")
