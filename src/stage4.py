"""Stage 4 driver: warm-side models vs PMV, SHAP, permutation importance, scenarios. Writes results/ and figures/."""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd

try:
    from . import config, explain, explore, models, scenarios as sc
except ImportError:  # pragma: no cover
    import config, explain, explore, models, scenarios as sc  # type: ignore

warnings.filterwarnings("ignore")


def run(verbose: bool = True) -> dict:
    R = config.RESULTS
    R.mkdir(exist_ok=True)
    out: dict = {}
    df = models.set_target(models.load_clean(), config.TARGET_KIND)
    M = config.COLUMN_MAP["ashrae"]
    out["n_rows"], out["n_buildings"] = int(len(df)), int(df[models.GROUP].nunique())
    out["warm_prevalence"] = float(df[models.Y].mean())

    # ---- 1. model comparison (warm-side discomfort) --------------------------------------------
    cb, sb, fb, ob, bb = models.run_experiment(df, models.building_splits(df), "building 5-fold", signed_pmv=True)
    cl, sl, fl, ol, bl = models.run_experiment(df, models.contributor_splits(df), "leave-one-contributor-out",
                                               signed_pmv=True)
    cls = pd.concat([cb, cl], ignore_index=True); sens = pd.concat([sb, sl], ignore_index=True)
    cls.to_csv(R / "model_comparison_warm.csv", index=False)
    sens.to_csv(R / "model_comparison_sensation.csv", index=False)
    pd.concat([fb, fl]).to_csv(R / "fold_metrics.csv", index=False)
    ob.to_csv(R / "oof_predictions_building_cv.csv.gz", index=False)
    models.save_params({"building_5fold": bb, "leave_one_contributor_out": bl})
    out["cls"], out["sens"] = cls, sens

    ml = ["Logistic regression (linear)", "XGBoost"]
    auc = cls[cls.model.isin(ml)].pivot(index="model", columns="validation", values="roc_auc")
    primary = auc.mean(axis=1).idxmax()
    out["primary_model"] = primary
    out["auc_by_validation"] = auc

    explore.fig_roc_warm(ob)
    explore.fig_calibration_warm(ob)

    # ---- 2. final fits, SHAP, permutation importance -----------------------------------------------
    fitted, params = explain.fit_final(df)
    out["final_params"] = params
    X = df[models.FEATURES]
    sv_x = explain.shap_xgb(fitted["XGBoost"], X)
    sv_l = explain.shap_linear(fitted["Logistic regression (linear)"], X)
    imp = explain.global_importance(sv_x, sv_l, list(X.columns))
    imp.to_csv(R / "shap_global_importance.csv", index=False)
    pi = explain.permutation_importance(df, params, models.contributor_splits(df))
    pi.to_csv(R / "permutation_importance_loco.csv", index=False)
    lr = fitted["Logistic regression (linear)"].steps[-1][1]
    coef = pd.DataFrame({"feature": models.FEATURES, "std_coef_log_odds": lr.coef_[0]})
    coef["odds_ratio_per_sd"] = np.exp(coef["std_coef_log_odds"])
    sd = df[models.FEATURES].std().to_numpy()
    coef["sd"] = sd
    coef.to_csv(R / "logistic_coefficients.csv", index=False)
    explain.fig_shap_global(imp); explain.fig_shap_beeswarm(sv_x, X); explain.fig_shap_dependence(sv_x, X)
    out["shap"], out["perm"], out["coef"] = imp, pi, coef

    # ---- 3. scenarios ----------------------------------------------------------------------------------
    clo_tbl = pd.read_csv(R / "manikin_total_clo.csv")
    incr = sc.cover_increments(clo_tbl); incr.to_csv(R / "manikin_cover_increments.csv", index=False)
    fan = sc.fan_levels(df); out["fan"] = fan
    p99 = float(df[M["clo"]].quantile(0.99)); p95 = float(df[M["clo"]].quantile(0.95))
    q = df[M["clo"]].quantile([0.25, 0.5, 0.75]).round(2).tolist()
    out["clo_quantiles"] = {"p25": q[0], "p50": q[1], "p75": q[2], "p95": p95, "p99": p99}
    est = {n: fitted[n] for n in ml}
    TA, RH = 29.0, 85.0
    out["support_29_85"] = {lab: sc.support(df, TA, RH, v) for lab, v in ((k, fan[k]) for k in ("no fan", "fan"))}

    curves = {lab: sc.curve_table(est, TA, RH, fan[lab], lab) for lab in ("no fan", "fan")}
    pd.concat(curves.values()).to_csv(R / "scenario_curves_29C_85RH.csv", index=False)
    lim = {lab: sc.limits(curves[lab], primary, p99) for lab in curves}
    out["limits_29_85"] = lim

    primary_est = est[primary]
    scen = {lab: (TA, RH, fan[lab]) for lab in curves}
    boot = sc.bootstrap_limit(df, primary_est, scen)
    out["boot_29_85"] = {k: {kk: vv for kk, vv in v.items() if kk not in ("lo", "hi")} for k, v in boot.items()}
    sc.fig_curves(curves, boot, primary, p99, fan)
    sc.fig_heatmaps(est[primary], fan, q[1])

    # limits across temperature (RH 85) and across RH (29 C)
    rows = []
    for ta in (27, 28, 29, 30, 31):
        for lab in ("no fan", "fan"):
            c = sc.curve_table(est, ta, RH, fan[lab], lab)
            l = sc.limits(c, primary, p99)
            rows.append({"ta": ta, "rh": RH, "fan": lab, **{f"{k}": v["highest_clo"] for k, v in l.items()},
                         "support_rows": sc.support(df, ta, RH, fan[lab])["rows_within_1C_and_7pctRH"]})
    for rh in (60, 70, 80):
        for lab in ("no fan", "fan"):
            c = sc.curve_table(est, TA, rh, fan[lab], lab)
            l = sc.limits(c, primary, p99)
            rows.append({"ta": TA, "rh": rh, "fan": lab, **{f"{k}": v["highest_clo"] for k, v in l.items()},
                         "support_rows": sc.support(df, TA, rh, fan[lab])["rows_within_1C_and_7pctRH"]})
    lt = pd.DataFrame(rows); lt.to_csv(R / "scenario_highest_clo_by_conditions.csv", index=False)
    out["limits_table"] = lt

    # manikin sets: garment quartile + cover increment, both postures, both fan states
    rows = []
    for g_lab, g in (("p25", q[0]), ("median", q[1]), ("p75", q[2])):
        for _, r in incr.iterrows():
            tot = g + r["increment"]
            for lab in ("no fan", "fan"):
                X1 = sc.frame(tot, TA, RH, fan[lab])
                pm = sc.pmv(X1)
                rows.append({"garment_level": g_lab, "garment_clo": g, "set": r["set"], "posture": r["posture"],
                             "cover_increment": round(r["increment"], 3), "total_clo": round(tot, 3),
                             "condition": lab, "p_warm_primary": float(sc.p_warm(est[primary], X1)[0]),
                             "p_warm_xgboost": float(sc.p_warm(est["XGBoost"], X1)[0]),
                             "pmv_iso": float(pm["pmv_iso"][0]), "pmv_ashrae": float(pm["pmv_ashrae"][0]),
                             "total_above_p99": bool(tot > p99)})
    st = pd.DataFrame(rows); st.to_csv(R / "scenario_bedding_sets_29C_85RH.csv", index=False)
    sc.fig_sets(st[st.garment_level == "median"])
    out["sets"] = st

    # ---- 4. numbers file ---------------------------------------------------------------------------------
    def tojson(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, (pd.DataFrame,)):
            return o.to_dict("records")
        return str(o)
    (R / "key_numbers.json").write_text(json.dumps({k: v for k, v in out.items() if k not in ("cls", "sens", "shap", "perm", "coef", "sets", "limits_table", "auc_by_validation")}, indent=2, default=tojson))
    return out
