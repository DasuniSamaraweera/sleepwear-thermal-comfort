"""Final models, SHAP explanations and held-out permutation importance."""
from __future__ import annotations

import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.base import clone
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold

try:
    from . import config, explore, models
except ImportError:  # pragma: no cover
    import config, explore, models  # type: ignore

LABELS = {"clo": "Clothing insulation (clo)", "ta": "Air temperature (degC)", "rh": "Relative humidity (%)",
          "vel": "Air speed (m/s)", "met": "Metabolic rate (met)"}
NICE = {"clo": "Clothing insulation", "ta": "Air temperature", "rh": "Relative humidity",
        "vel": "Air speed", "met": "Metabolic rate"}


def fit_final(df: pd.DataFrame):
    """Tune on all data with building-grouped 5-fold CV (final fit only; the honest estimates come from the
    nested validation in stage 3/4). Returns ({name: fitted estimator}, {name: best params})."""
    X, y, g = df[models.FEATURES].to_numpy(float), df[models.Y].to_numpy(), df[models.GROUP].to_numpy()
    fitted, params = {}, {}
    for name, key in (("Logistic regression (linear)", "Logistic regression (linear)"), ("XGBoost", "XGBoost")):
        est, grid, scoring = models.classifiers()[key]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            gs = GridSearchCV(est, grid, scoring=scoring, cv=GroupKFold(5), refit=True).fit(X, y, groups=g)
        fitted[name], params[name] = gs.best_estimator_, gs.best_params_
    return fitted, params


def shap_xgb(model, X: pd.DataFrame) -> np.ndarray:
    return shap.TreeExplainer(model).shap_values(X)


def shap_linear(pipe, X: pd.DataFrame) -> np.ndarray:
    scaler, lr = pipe.steps[0][1], pipe.steps[-1][1]
    Xs = scaler.transform(X)
    return shap.LinearExplainer(lr, shap.maskers.Independent(Xs, max_samples=len(Xs))).shap_values(Xs)


def global_importance(sv_x: np.ndarray, sv_l: np.ndarray, cols) -> pd.DataFrame:
    t = pd.DataFrame({"feature": cols, "mean_abs_shap_xgboost": np.abs(sv_x).mean(0),
                      "mean_abs_shap_logistic": np.abs(sv_l).mean(0)})
    for c in ("xgboost", "logistic"):
        t[f"share_{c}_pct"] = 100 * t[f"mean_abs_shap_{c}"] / t[f"mean_abs_shap_{c}"].sum()
    return t.sort_values("mean_abs_shap_xgboost", ascending=False).reset_index(drop=True)


def permutation_importance(df: pd.DataFrame, params: dict, splits, n_rep: int = 30) -> pd.DataFrame:
    """Held-out ROC AUC drop when one feature is shuffled in the test fold (leave-one-contributor-out)."""
    X, y = df[models.FEATURES].to_numpy(float), df[models.Y].to_numpy()
    rng = np.random.default_rng(config.SEED)
    rows = []
    for name in ("Logistic regression (linear)", "XGBoost"):
        est = models.classifiers()[name][0]
        est = clone(est).set_params(**params[name])
        for k, (tr, te) in enumerate(splits):
            if len(np.unique(y[te])) < 2:
                continue
            est.fit(X[tr], y[tr])
            base = roc_auc_score(y[te], est.predict_proba(X[te])[:, 1])
            for j, f in enumerate(models.FEATURES):
                drops = []
                for _ in range(n_rep):
                    Xp = X[te].copy()
                    Xp[:, j] = rng.permutation(Xp[:, j])
                    drops.append(base - roc_auc_score(y[te], est.predict_proba(Xp)[:, 1]))
                rows.append({"model": name, "fold": k, "feature": f, "auc_drop": float(np.mean(drops))})
    r = pd.DataFrame(rows)
    return r.groupby(["model", "feature"])["auc_drop"].agg(["mean", "std"]).reset_index()


# ------------------------------------------------------------------ figures
def fig_shap_global(imp: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    t = imp.sort_values("mean_abs_shap_xgboost")
    y = np.arange(len(t))
    ax.barh(y + 0.2, t["mean_abs_shap_xgboost"], 0.38, color=explore.C["alt"], label="XGBoost")
    ax.barh(y - 0.2, t["mean_abs_shap_logistic"], 0.38, color=explore.C["main"], label="Logistic regression")
    ax.set_yticks(y, [NICE[f] for f in t["feature"]])
    ax.set_xlabel("Mean |SHAP value| (log-odds of warm-side discomfort)")
    ax.set_title("Global feature importance")
    ax.legend(frameon=False, loc="lower right")
    explore.save(fig, "fig09_shap_global_importance")


def fig_shap_beeswarm(sv: np.ndarray, X: pd.DataFrame):
    plt.figure(figsize=(6.2, 3.8))
    shap.summary_plot(sv, X.rename(columns=NICE), show=False, plot_size=None)
    ax = plt.gca()
    ax.set_xlabel("SHAP value (log-odds of warm-side discomfort)")
    ax.set_title("XGBoost: how each feature moves the prediction")
    fig = plt.gcf()
    explore.save(fig, "fig10_shap_beeswarm_xgboost")


def fig_shap_dependence(sv: np.ndarray, X: pd.DataFrame):
    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6))
    for ax, f in zip(axes.ravel(), ["clo", "ta", "rh", "vel"]):
        j = list(X.columns).index(f)
        ax.scatter(X[f], sv[:, j], s=6, alpha=0.25, color=explore.C["grey"])
        bins = pd.qcut(X[f], 12, duplicates="drop")
        m = pd.DataFrame({"x": X[f], "s": sv[:, j], "b": bins}).groupby("b", observed=True).mean()
        ax.plot(m["x"], m["s"], color=explore.C["alt"], lw=2)
        ax.axhline(0, color="black", lw=0.7)
        ax.set_xlabel(LABELS[f])
        ax.set_ylabel("SHAP value (log-odds)")
    fig.suptitle("XGBoost SHAP dependence (line = binned mean)", y=1.0)
    fig.tight_layout()
    explore.save(fig, "fig11_shap_dependence")
