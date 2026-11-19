"""PMV/PPD baselines and interpretable ML models with building-grouped validation.

Inputs for every model (same information for fairness): clo, air temperature,
relative humidity, air speed, met. PMV additionally uses mean radiant temperature
(tr), falling back to air temperature where tr is missing.

Validation: StratifiedGroupKFold by building_id (never a random row split), plus a
leave-one-contributor-out check. Metrics are pooled out-of-fold, with 95% cluster
bootstrap intervals (resampling whole buildings).
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, roc_auc_score
from sklearn.model_selection import (GridSearchCV, GroupKFold, LeaveOneGroupOut,
                                     StratifiedGroupKFold)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from xgboost import XGBClassifier, XGBRegressor

try:
    from . import config
except ImportError:  # pragma: no cover
    import config  # type: ignore

M = config.COLUMN_MAP["ashrae"]
FEATURES = [M["clo"], M["air_temp"], M["rel_humidity"], M["air_speed"], M["met"]]
SENS = M["thermal_sensation"]
GROUP = M["building_id"]
Y = "y"   # binary target column; set with set_target()


# ----------------------------------------------------------------------------- data
def load_clean() -> pd.DataFrame:
    return pd.read_csv(config.PROCESSED / "ashrae_clean.csv.gz")


def set_target(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """kind: 'warm' (vote >= +1, MAIN target) or 'comfortable' (vote in [-1,+1], stage 3 negative result)."""
    d = df.copy()
    d[Y] = (d["vote_rounded"] >= 1).astype(int) if kind == "warm" else d["comfortable"].astype(int)
    return d


# ----------------------------------------------------------------------------- PMV baseline
def pmv_baseline(df: pd.DataFrame) -> pd.DataFrame:
    """PMV/PPD with pythermalcomfort: ISO 7730 and ASHRAE 55 (adds the elevated-air-speed cooling effect)."""
    from pythermalcomfort.models import pmv_ppd_ashrae, pmv_ppd_iso
    from pythermalcomfort.utilities import v_relative

    tdb = df[M["air_temp"]].to_numpy(float)
    tr = df[M["mean_radiant_temp"]].fillna(df[M["air_temp"]]).to_numpy(float)
    vr = np.asarray(v_relative(v=df[M["air_speed"]].to_numpy(float), met=df[M["met"]].to_numpy(float)), float)
    kw = dict(tdb=tdb, tr=tr, vr=vr, rh=df[M["rel_humidity"]].to_numpy(float),
              met=df[M["met"]].to_numpy(float), clo=df[M["clo"]].to_numpy(float), limit_inputs=False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # pythermalcomfort warns per row when the cooling effect is 0
        iso, ash = pmv_ppd_iso(**kw), pmv_ppd_ashrae(**kw)
    out = pd.DataFrame({"pmv_iso": np.asarray(iso.pmv, float), "ppd_iso": np.asarray(iso.ppd, float),
                        "pmv_ashrae": np.asarray(ash.pmv, float), "ppd_ashrae": np.asarray(ash.ppd, float)},
                       index=df.index)
    out.attrs["n_tr_imputed"] = int(df[M["mean_radiant_temp"]].isna().sum())
    return out


# ----------------------------------------------------------------------------- model factories
def _logit(poly: bool):
    steps = [StandardScaler()]
    if poly:
        steps += [PolynomialFeatures(2, include_bias=False), StandardScaler()]
    steps.append(LogisticRegression(max_iter=5000))
    return make_pipeline(*steps), {"logisticregression__C": [0.03, 0.1, 0.3, 1, 3]}


def _ridge(poly: bool):
    steps = [StandardScaler()]
    if poly:
        steps += [PolynomialFeatures(2, include_bias=False), StandardScaler()]
    steps.append(Ridge())
    return make_pipeline(*steps), {"ridge__alpha": [0.1, 1, 10, 100]}


XGB_GRID = {"max_depth": [2, 3, 4], "n_estimators": [100, 250]}
XGB_FIXED = dict(learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
                 tree_method="hist", n_jobs=2, random_state=config.SEED, verbosity=0)


def classifiers():
    return {
        "Logistic regression (linear)": (*_logit(False), "roc_auc"),
        "Logistic regression (quadratic)": (*_logit(True), "roc_auc"),
        "XGBoost": (XGBClassifier(**XGB_FIXED, eval_metric="logloss"), XGB_GRID, "roc_auc"),
    }


def regressors():
    return {
        "Linear (ridge)": (*_ridge(False), "neg_mean_absolute_error"),
        "Quadratic (ridge)": (*_ridge(True), "neg_mean_absolute_error"),
        "XGBoost": (XGBRegressor(**XGB_FIXED), XGB_GRID, "neg_mean_absolute_error"),
    }


# ----------------------------------------------------------------------------- metrics
def cls_metrics(y, score, thresh=0.5, higher_is_comfort=True) -> dict:
    s = score if higher_is_comfort else -score
    pred = (score >= thresh).astype(int) if higher_is_comfort else (score <= thresh).astype(int)
    return {"accuracy": accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, average="macro"),
            "roc_auc": roc_auc_score(y, s)}


def matched_pred(score: np.ndarray, y: np.ndarray, splits) -> np.ndarray:
    """Top-share rule: within each test fold, call the share of rows equal to the TRAINING comfortable
    prevalence 'comfortable' (highest scores). Uses no test labels and treats every model, PMV included,
    the same way, so macro F1 is not dominated by one model predicting 'comfortable' for everything."""
    pred = np.zeros(len(score), int)
    for tr, te in splits:
        prev = y[tr].mean()
        thr = np.quantile(score[te], 1 - prev)
        pred[te] = (score[te] >= thr).astype(int)
    return pred


def sens_metrics(vote, vote_rounded, pred) -> dict:
    pred = np.clip(pred, -3, 3)
    return {"mae": mean_absolute_error(vote, pred),
            "within_1": float((np.abs(np.round(pred) - vote_rounded) <= 1).mean())}


def cluster_bootstrap(groups, fn, n_boot=config.N_BOOT, seed=config.SEED) -> dict:
    """95% CI by resampling whole buildings; fn(idx) -> dict of metrics."""
    rng = np.random.default_rng(seed)
    ids = pd.unique(groups)
    where = {g: np.flatnonzero(groups == g) for g in ids}
    res = []
    for _ in range(n_boot):
        pick = rng.choice(ids, size=len(ids), replace=True)
        idx = np.concatenate([where[g] for g in pick])
        try:
            res.append(fn(idx))
        except ValueError:  # resample with a single class
            continue
    r = pd.DataFrame(res)
    return {k: (r[k].quantile(0.025), r[k].quantile(0.975)) for k in r.columns}


# ----------------------------------------------------------------------------- cross-validation
def _fit_predict(name, est, grid, scoring, Xtr, ytr, gtr, Xte, kind):
    inner = GroupKFold(n_splits=3)
    gs = GridSearchCV(est, grid, scoring=scoring, cv=inner, n_jobs=1, refit=True)
    gs.fit(Xtr, ytr, groups=gtr)
    if kind == "cls":
        return gs.predict_proba(Xte)[:, 1], gs.best_params_
    return gs.predict(Xte), gs.best_params_


def oof_predictions(df: pd.DataFrame, splits, kind: str):
    """Pooled out-of-fold predictions for every model of the given kind ('cls' or 'reg')."""
    X = df[FEATURES].to_numpy(float)
    y = df[Y].to_numpy() if kind == "cls" else df[SENS].to_numpy(float)
    g = df[GROUP].to_numpy()
    models = classifiers() if kind == "cls" else regressors()
    oof = {n: np.full(len(df), np.nan) for n in models}
    folds = {n: np.full(len(df), -1) for n in models}
    best = {n: [] for n in models}
    for k, (tr, te) in enumerate(splits):
        for n, (est, grid, scoring) in models.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                p, bp = _fit_predict(n, est, grid, scoring, X[tr], y[tr], g[tr], X[te], kind)
            oof[n][te] = p
            folds[n][te] = k
            best[n].append({kk: (vv.item() if hasattr(vv, "item") else vv) for kk, vv in bp.items()})
    return oof, folds, best


def building_splits(df: pd.DataFrame):
    sgk = StratifiedGroupKFold(n_splits=config.N_SPLITS, shuffle=True, random_state=config.SEED)
    return list(sgk.split(df[FEATURES], df[Y], df[GROUP]))


def contributor_splits(df: pd.DataFrame):
    return list(LeaveOneGroupOut().split(df[FEATURES], groups=df[M["contributor"]]))


# ----------------------------------------------------------------------------- main experiment
def _ci_cols(row: dict, ci: dict) -> None:
    for k, (lo, hi) in ci.items():
        row[f"{k}_ci95"] = f"[{lo:.3f}, {hi:.3f}]"


def run_experiment(df: pd.DataFrame, splits, label: str, signed_pmv: bool = False):
    """Return (comfort_table, sensation_table, fold_table, oof_frame, best_params)."""
    pmv = pmv_baseline(df)
    y = df[Y].to_numpy()
    vote = df[SENS].to_numpy(float)
    vr = df["vote_rounded"].to_numpy()
    g = df[GROUP].to_numpy()
    n = len(df)

    oof_c, fold_c, best_c = oof_predictions(df, splits, "cls")
    oof_r, fold_r, best_r = oof_predictions(df, splits, "reg")

    # ---- comfort (binary) -------------------------------------------------
    cls_rows, fold_rows = [], []

    def add_cls(model, fn, scores, thresh, fold_ids, matched=True):
        mp = matched_pred(scores, y, splits) if matched else None
        full = (lambda i: {**fn(i), "macro_f1_matched": f1_score(y[i], mp[i], average="macro", zero_division=0)}) \
            if matched else (lambda i: {**fn(i), "macro_f1_matched": np.nan})
        row = {"validation": label, "model": model, **full(np.arange(n))}
        _ci_cols(row, cluster_bootstrap(g, full))
        cls_rows.append(row)
        if fold_ids is not None:
            for f in sorted(set(fold_ids[fold_ids >= 0])):
                m = fold_ids == f
                if len(np.unique(y[m])) == 2:
                    fold_rows.append({"validation": label, "model": model, "task": "comfort", "fold": int(f),
                                      "n": int(m.sum()), **cls_metrics(y[m], scores[m], thresh)})

    ref_folds = np.full(n, -1)
    for k, (_, te) in enumerate(splits):
        ref_folds[te] = k
    for name, col in (("PMV (ISO 7730)", "pmv_iso"), ("PMV (ASHRAE 55, cooling effect)", "pmv_ashrae")):
        if signed_pmv:      # positive class = warm-side discomfort: predicted when PMV >= +0.5
            a, th = pmv[col].to_numpy(), 0.5
        else:               # positive class = comfortable: predicted when |PMV| <= 1
            a, th = -np.abs(pmv[col].to_numpy()), -1.0
        add_cls(name, (lambda a, th: lambda i: cls_metrics(y[i], a[i], th))(a, th), a, th, ref_folds)
    for name, p in oof_c.items():
        add_cls(name, (lambda p: lambda i: cls_metrics(y[i], p[i]))(p), p, 0.5, fold_c[name])
    add_cls("Always predict the majority class (reference)",
            lambda i: {"accuracy": max(y[i].mean(), 1 - y[i].mean()),
                       "macro_f1": f1_score(y[i], np.full(len(i), int(y.mean() >= 0.5)), average="macro", zero_division=0),
                       "roc_auc": 0.5},
            np.ones(n), 0.5, None, matched=False)

    # ---- sensation vote (regression) -------------------------------------
    sens_rows = []

    def add_sens(model, pred, fold_ids):
        fn = lambda i: sens_metrics(vote[i], vr[i], pred[i])
        row = {"validation": label, "model": model, **fn(np.arange(n))}
        _ci_cols(row, cluster_bootstrap(g, fn))
        sens_rows.append(row)
        if fold_ids is not None:
            for f in sorted(set(fold_ids[fold_ids >= 0])):
                m = fold_ids == f
                fold_rows.append({"validation": label, "model": model, "task": "sensation", "fold": int(f),
                                  "n": int(m.sum()), **sens_metrics(vote[m], vr[m], pred[m])})

    add_sens("PMV (ISO 7730)", pmv["pmv_iso"].to_numpy(), None)
    add_sens("PMV (ASHRAE 55, cooling effect)", pmv["pmv_ashrae"].to_numpy(), None)
    for name, p in oof_r.items():
        add_sens(name, p, fold_r[name])
    add_sens("Predict training mean (reference)", np.full(n, vote.mean()), None)

    oof = df[[M["record_id"], GROUP, Y, "vote_rounded", SENS]].copy()
    oof = pd.concat([oof, pmv], axis=1)
    for name, p in oof_c.items():
        oof["p_comfort__" + name] = p
    for name, p in oof_r.items():
        oof["sens_pred__" + name] = p
    return (pd.DataFrame(cls_rows), pd.DataFrame(sens_rows), pd.DataFrame(fold_rows), oof,
            {"classifiers": best_c, "regressors": best_r})


def save_params(best: dict) -> None:
    (config.RESULTS / "best_params.json").write_text(json.dumps(best, indent=2, default=str))
