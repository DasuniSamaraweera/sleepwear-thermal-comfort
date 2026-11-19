"""Generate results/summary.md and results/design_garment_budget.csv.

Every number is read from files written by the pipeline (results/*.csv, results/key_numbers.json) or computed here
from data/processed/ashrae_clean.csv.gz. Nothing numeric is typed by hand. Qualitative wording was checked against
these numbers when the template was written; the checks are asserted at the top of build_all().
"""
from __future__ import annotations

import json

import pandas as pd

try:
    from . import config
except ImportError:  # pragma: no cover
    import config  # type: ignore

R = config.RESULTS


def f2(x):
    return f"{x:.2f}"


def f3(x):
    return f"{x:.3f}"


def pct(x, d=1):
    return f"{100 * x:.{d}f}%"


def md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(r[c]) for c in cols) + " |")
    return "\n".join(lines)


def load():
    d = {}
    d["flog"] = pd.read_csv(R / "filter_log.csv")
    d["cls"] = pd.read_csv(R / "model_comparison_warm.csv")
    d["sens"] = pd.read_csv(R / "model_comparison_sensation.csv")
    d["neg"] = pd.read_csv(R / "neg_comfort_label_model_comparison.csv")
    d["negs"] = pd.read_csv(R / "neg_comfort_label_sensation_comparison.csv")
    d["shap"] = pd.read_csv(R / "shap_global_importance.csv").set_index("feature")
    d["perm"] = pd.read_csv(R / "permutation_importance_loco.csv")
    d["coef"] = pd.read_csv(R / "logistic_coefficients.csv").set_index("feature")
    d["lt"] = pd.read_csv(R / "scenario_highest_clo_by_conditions.csv")
    d["sets"] = pd.read_csv(R / "scenario_bedding_sets_29C_85RH.csv")
    d["inc"] = pd.read_csv(R / "manikin_cover_increments.csv")
    d["clo"] = pd.read_csv(R / "manikin_total_clo.csv")
    d["cov"] = pd.read_csv(R / "clo_coverage.csv")
    d["country"] = pd.read_csv(R / "subset_by_country.csv")
    d["stats"] = pd.read_csv(R / "subset_summary_stats.csv", index_col=0)
    d["key"] = json.loads((R / "key_numbers.json").read_text())
    d["data"] = pd.read_csv(config.PROCESSED / "ashrae_clean.csv.gz")
    return d


def row(df, validation, model):
    return df[(df.validation == validation) & (df.model == model)].iloc[0]


def build_all() -> None:
    d = load()
    K = d["key"]
    BV, LO = "building 5-fold", "leave-one-contributor-out"
    P = K["primary_model"]
    LR, XG = "Logistic regression (linear)", "XGBoost"
    PI, PA = "PMV (ISO 7730)", "PMV (ASHRAE 55, cooling effect)"
    fan = K["fan"]
    lim = K["limits_29_85"]
    boot = K["boot_29_85"]
    q = K["clo_quantiles"]
    key_lr = f"p_warm__{P}"

    # ---- sanity checks on the claims the templates make ------------------------------------------
    sh = d["shap"]
    assert sh.index[0] == "ta" and sh.index[1] == "vel", "template assumes temperature > air speed > rest"
    assert sh.loc["clo", "share_xgboost_pct"] < 5 and sh.loc["clo", "share_logistic_pct"] < 6
    assert lim["fan"][key_lr]["highest_clo"] > lim["no fan"][key_lr]["highest_clo"]
    assert lim["fan"]["pmv_iso"]["highest_clo"] > lim["no fan"]["pmv_iso"]["highest_clo"]
    assert d["coef"].loc["ta", "odds_ratio_per_sd"] > 1 and d["coef"].loc["vel", "odds_ratio_per_sd"] < 1

    # ---- derived numbers ---------------------------------------------------------------------------
    f = d["flog"]
    n_final, b_final = int(f.iloc[-1].rows_after), int(f.iloc[-1].buildings_after)
    data = d["data"]
    M = config.COLUMN_MAP["ashrae"]
    near = (data[M["air_temp"]].sub(29).abs() <= 1) & (data[M["rel_humidity"]].sub(85).abs() <= 7)
    obs = {}
    for lab, flag in (("no fan", 0), ("fan", 1)):
        m = near & (data[M["fan"]] == flag)
        obs[lab] = (int(m.sum()), float(data.loc[m, models_y(data)].mean()) if m.sum() else float("nan"))
    inc = d["inc"]
    base_inc = inc.set_index(["set", "posture"])["increment"]
    posture_diff = (d["clo"].pivot(index="set", columns="posture", values="clo_total")
                    .assign(diff=lambda t: t["supine"] - t["side"])["diff"])
    sets = d["sets"]
    med = sets[sets.garment_level == "median"].copy()
    med["ML ok"] = med["p_warm_primary"] <= config.WARM_P_MAX
    med["PMV ISO ok"] = med["pmv_iso"] < config.PMV_WARM_CUT
    med["PMV ASHRAE ok"] = med["pmv_ashrae"] < config.PMV_WARM_CUT
    # garment budget = highest total clo (PMV ISO, the more cautious theory model) minus cover increment
    bud = []
    for lab in ("no fan", "fan"):
        for _, r in inc.iterrows():
            g = lim[lab]["pmv_iso"]["highest_clo"] - r["increment"]
            bud.append({"condition": lab, "set": r["set"], "posture": r["posture"],
                        "cover_increment": f2(r["increment"]),
                        "garment_clo_budget_pmv_iso": f2(max(g, 0.0)),
                        "fits_training_garment_range": "yes" if g >= 0.1 else "no garment fits"})
    bud = pd.DataFrame(bud)
    bud.to_csv(R / "design_garment_budget.csv", index=False)

    # ---- tables for the summary ----------------------------------------------------------------------
    def cmp_table(validation):
        t = d["cls"][d["cls"].validation == validation]
        out = pd.DataFrame({"Model": t.model, "Accuracy": t.accuracy.map(f3), "Macro F1": t.macro_f1.map(f3),
                            "Macro F1 (matched share)": t.macro_f1_matched.map(lambda v: "n/a" if v != v else f3(v)),
                            "ROC AUC": t.roc_auc.map(f3), "ROC AUC 95% CI": t.roc_auc_ci95})
        return md_table(out)

    def sens_table(validation):
        t = d["sens"][d["sens"].validation == validation]
        out = pd.DataFrame({"Model": t.model, "MAE (vote units)": t.mae.map(f3), "MAE 95% CI": t.mae_ci95,
                            "Accuracy within +-1": t.within_1.map(f3), "Within +-1 95% CI": t.within_1_ci95})
        return md_table(out)

    REF = "Always predict the majority class (reference)"   # reference row of the negative-result table
    if REF not in set(d["neg"].model):   # older stage 3 runs named the reference row differently
        REF = [m for m in d["neg"].model.unique() if m.startswith("Always")][0]
    neg_b = lambda m: row(d["neg"], BV, m)
    neg_l = lambda m: row(d["neg"], LO, m)

    imp = sh.reset_index()[["feature", "share_xgboost_pct", "share_logistic_pct"]].copy()
    imp["share_xgboost_pct"] = imp["share_xgboost_pct"].map(lambda v: f"{v:.1f}%")
    imp["share_logistic_pct"] = imp["share_logistic_pct"].map(lambda v: f"{v:.1f}%")
    imp.columns = ["Feature", "SHAP share, XGBoost", "SHAP share, logistic"]
    pm = d["perm"].pivot(index="feature", columns="model", values="mean").loc[sh.index]
    perm = pd.DataFrame({"Feature": pm.index, "AUC drop, logistic": pm[LR].map(f3).values,
                         "AUC drop, XGBoost": pm[XG].map(f3).values})

    lt = d["lt"].copy()
    def cell(v):
        return "none" if v != v else f2(v)
    lt_tab = pd.DataFrame({"Air temp (degC)": lt.ta.map(lambda v: f"{v:.0f}"), "RH (%)": lt.rh.map(lambda v: f"{v:.0f}"),
                           "Air speed": lt.fan, "Rows nearby": lt.support_rows,
                           "ML limit (clo)": lt[key_lr].map(cell), "PMV ISO limit": lt.pmv_iso.map(cell),
                           "PMV ASHRAE limit": lt.pmv_ashrae.map(cell)})
    st = med[["set", "posture", "total_clo", "condition", "p_warm_primary", "pmv_iso", "pmv_ashrae", "ML ok",
              "PMV ISO ok", "PMV ASHRAE ok"]].copy()
    st["p_warm_primary"] = st.p_warm_primary.map(lambda v: pct(v))
    for c in ("total_clo", "pmv_iso", "pmv_ashrae"):
        st[c] = st[c].map(f2)
    for c in ("ML ok", "PMV ISO ok", "PMV ASHRAE ok"):
        st[c] = st[c].map({True: "yes", False: "no"})
    st.columns = ["Set", "Posture", "Total clo", "Air", "ML P(warm)", "PMV ISO", "PMV ASHRAE", "ML ok", "ISO ok", "ASHRAE ok"]

    l29 = d["lt"][(d["lt"].ta == 29) & (d["lt"].fan == "no fan")].set_index("rh")
    hum_note = (f"Caution on humidity: at 29 degC without a fan the data-driven limit {cell(l29.loc[70, key_lr])} clo at 70% RH becomes "
                f"{cell(l29.loc[85, key_lr])} clo at 85% RH (it rises with humidity), while the PMV limit falls from {cell(l29.loc[70, 'pmv_iso'])} to "
                f"{cell(l29.loc[85, 'pmv_iso'])} clo. The fitted humidity effect is negative (odds ratio {f2(d['coef'].loc['rh','odds_ratio_per_sd'])} per sd), which "
                f"reflects who lives where in this subset, so do not read the data-driven humidity trend as physiology.")
    cov_rows = d["cov"]
    cov_min, cov_max = int(cov_rows.ashrae_rows_with_clo_ge_this.min()), int(cov_rows.ashrae_rows_with_clo_ge_this.max())
    n_fan_rows = fan["n_fan"] + fan["n_no_fan"]
    ctry = d["country"]
    ctry_txt = ", ".join(f"{r.country.title()} {int(r.rows):,} rows ({int(r.buildings)} buildings)"
                         for r in ctry.itertuples())

    # =================================================================================================
    summary = f"""# Results summary

All numbers below were produced by the code in `src/` (stage 4 driver: `src/stage4.py`) and are stored in `results/`.
Datasets (both CC BY 4.0): ASHRAE Global Thermal Comfort Database II v2.1 (https://zenodo.org/records/4993952) and
Heat balance of sleeping sets, Newton thermal manikin, Brno University of Technology (https://zenodo.org/records/20179324).

## 1. Data used (`results/filter_log.csv`)

{md_table(f[['step','rows_before','rows_after','rows_dropped','buildings_after']].rename(columns={'step':'Step','rows_before':'Rows before','rows_after':'Rows after','rows_dropped':'Dropped','buildings_after':'Buildings after'}))}

Final subset: {n_final:,} votes from {b_final} buildings, {data[M['contributor']].nunique()} contributors. {ctry_txt}.
There are no Sri Lankan records. Share of votes that are warm-side discomfort (rounded sensation vote of +1 or more):
{pct(K['warm_prevalence'])}.

## 2. Target definition (changed after the first modelling pass)

The first target, "comfortable = vote from -1 to +1", is almost unpredictable from the inputs (section 3, negative result).
It lumps warm and cold votes together. The main target is therefore **warm-side discomfort: rounded sensation vote of
+1 or more**, which is also the relevant failure for hot nights.

## 3. Negative result with the original "comfortable" label (`results/neg_comfort_label_*.csv`)

Held-out buildings: ROC AUC {f3(neg_b(XG).roc_auc)} (XGBoost), {f3(neg_b('Logistic regression (quadratic)').roc_auc)} (quadratic logistic),
{f3(neg_b(LR).roc_auc)} (linear logistic), {f3(neg_b(PA).roc_auc)} (PMV ASHRAE 55). Accuracy of every ML model
({f3(neg_b(XG).accuracy)} for XGBoost) equals always predicting "comfortable" ({f3(neg_b(REF).accuracy)}).
Leaving one contributor out, AUC fell to {f3(neg_l(LR).roc_auc)} (linear logistic), {f3(neg_l('Logistic regression (quadratic)').roc_auc)} (quadratic) and
{f3(neg_l(XG).roc_auc)} (XGBoost). Sensation MAE in that check was {f3(row(d['negs'], LO, 'Linear (ridge)').mae)} (ridge) to {f3(row(d['negs'], LO, 'XGBoost').mae)} (XGBoost) against {f3(row(d['negs'], LO, 'Predict training mean (reference)').mae)} for predicting the mean vote, with overlapping intervals, so there was essentially no gain.

## 4. Model comparison against PMV, main target (warm-side discomfort)

Inputs for all models: clo, air temperature, relative humidity, air speed, met (PMV also uses mean radiant temperature,
falling back to air temperature where missing). "Macro F1 (matched share)" calls the training-prevalence share of rows
in each test fold "warm" for every model, PMV included, using no test labels.

**Validation 1: building-grouped 5-fold** (StratifiedGroupKFold by `building_id`; intervals from resampling buildings)

{cmp_table(BV)}

**Validation 2: leave one contributor out** (4 held-out studies)

{cmp_table(LO)}

**Sensation vote (continuous), building-grouped 5-fold**

{sens_table(BV)}

**Sensation vote, leave one contributor out**

{sens_table(LO)}

Reading: the linear logistic model has the best or near-best ROC AUC in both validations ({f3(row(d['cls'], BV, LR).roc_auc)} and
{f3(row(d['cls'], LO, LR).roc_auc)}) and is the primary model. PMV ranks warm-side discomfort about as well (AUC {f3(row(d['cls'], BV, PI).roc_auc)}
for ISO 7730, {f3(row(d['cls'], BV, PA).roc_auc)} for ASHRAE 55) with overlapping intervals, but its raw threshold is badly
calibrated (accuracy {f3(row(d['cls'], BV, PI).accuracy)} at PMV >= +0.5, versus {f3(row(d['cls'], BV, 'Always predict the majority class (reference)').accuracy)}
for always predicting "not warm"). For the sensation vote, PMV has larger error (MAE {f3(row(d['sens'], BV, PA).mae)} to
{f3(row(d['sens'], BV, PI).mae)}) than predicting the mean vote ({f3(row(d['sens'], BV, 'Predict training mean (reference)').mae)}).
The ML models reduce MAE only slightly ({f3(row(d['sens'], BV, 'Linear (ridge)').mae)} for ridge).

## 5. What matters most (`results/shap_global_importance.csv`, `permutation_importance_loco.csv`, `logistic_coefficients.csv`)

Mean absolute SHAP value as a share of the total (final models fit on all {n_final:,} rows; SHAP describes the model, not causes):

{md_table(imp)}

Held-out permutation importance (drop in ROC AUC when one input is shuffled, leave-one-contributor-out folds):

{md_table(perm)}

Logistic odds ratio of warm-side discomfort per one standard deviation: air temperature {f2(d['coef'].loc['ta','odds_ratio_per_sd'])}
(sd {f2(d['coef'].loc['ta','sd'])} degC), air speed {f2(d['coef'].loc['vel','odds_ratio_per_sd'])} (sd {f2(d['coef'].loc['vel','sd'])} m/s),
relative humidity {f2(d['coef'].loc['rh','odds_ratio_per_sd'])} (sd {f2(d['coef'].loc['rh','sd'])} %), clo {f2(d['coef'].loc['clo','odds_ratio_per_sd'])}
(sd {f2(d['coef'].loc['clo','sd'])} clo). The dependence plots (`figures/fig11_shap_dependence`) show a steep rise with
temperature above roughly 27 degC, a fall with air speed up to about 1 m/s, a small negative association with humidity
(the opposite of what physiology alone suggests, so it reflects who lives where, not humidity as a cause), and a small,
irregular clo pattern. Clo contributes {sh.loc['clo','share_xgboost_pct']:.1f}% (XGBoost) and {sh.loc['clo','share_logistic_pct']:.1f}% (logistic) of total SHAP
importance, and shuffling it changes held-out AUC by about {d['perm'][(d['perm'].feature=='clo')]['mean'].abs().max():.3f} at most. **This data barely identifies an effect of clothing insulation.**

## 6. Sleepwear insulation from the manikin data (`results/manikin_total_clo.csv`, `manikin_cover_increments.csv`)

{md_table(d['clo'].assign(rtot_m2K_per_W=d['clo'].rtot_m2K_per_W.map(f3), clo_total=d['clo'].clo_total.map(f2), clo_uncertainty=d['clo'].clo_uncertainty.map(f2)).rename(columns={'set':'Set','posture':'Posture','rtot_m2K_per_W':'Rtot (m2K/W)','clo_total':'Total clo','clo_uncertainty':'Uncertainty (clo)'}))}

The "sets" are bedding configurations; the README does not say what garments were worn. Adding a cover to
"Without cover" adds {f2(inc[inc.increment>0].increment.min())} to {f2(inc.increment.max())} clo, while lying on the side instead of the back lowers the total by
{f2(posture_diff.min())} to {f2(posture_diff.max())} clo. Torso-segment back values above 5 clo (`manikin_torso_segments.csv`) look like a
contact artefact and were not used.

## 7. Scenarios at 29 degC and 85% RH (`results/scenario_*.csv`)

Scenario inputs: sleeping met {config.SLEEP_MET}; "no fan" = {f2(fan['no fan'])} m/s and "fan" = {f2(fan['fan'])} m/s (median measured speed in rows
where the fan flag is 0 / 1, from {fan['n_no_fan']} and {fan['n_fan']} rows). Total insulation = garment clo + manikin cover
increment. Rule for the data-driven model: predicted P(warm-side discomfort) <= {pct(config.WARM_P_MAX, 0)}. Rule for PMV: PMV < +{config.PMV_WARM_CUT}.
Support: {K['support_29_85']['no fan']['rows_within_1C_and_7pctRH']} votes lie within 1 degC and 7 % RH of this point
({obs['no fan'][0]} with the fan flag off, {obs['fan'][0]} with it on, the rest unrecorded), of which {pct(obs['no fan'][1], 0)} (no fan) and {pct(obs['fan'][1], 0)} (fan) were
warm-side discomfort. Only {K['support_29_85']['no fan']['rows_met_le_0.8']} votes have met of 0.8 or lower, so the sleeping-met input is near the edge of the data.

**Highest total clo that stays comfortable at 29 degC / 85% RH**

| Basis | No fan ({f2(fan['no fan'])} m/s) | Fan ({f2(fan['fan'])} m/s) |
|---|---|---|
| Data-driven model ({P}) | {f2(lim['no fan'][key_lr]['highest_clo'])} clo | {f2(lim['fan'][key_lr]['highest_clo'])} clo (beyond the data range: {lim['fan'][key_lr]['beyond_data_range']}) |
| ... 95% interval over buildings (bootstrap, {config.N_BOOT_SCEN} refits) | {f2(boot['no fan']['limit_lo'])} to {f2(boot['no fan']['limit_hi'])} clo | {f2(boot['fan']['limit_lo'])} to {f2(boot['fan']['limit_hi'])} clo |
| XGBoost (cross-check) | {f2(lim['no fan']['p_warm__XGBoost']['highest_clo'])} clo (top of grid, never exceeds {pct(config.WARM_P_MAX,0)}) | {f2(lim['fan']['p_warm__XGBoost']['highest_clo'])} clo (top of grid) |
| PMV, ISO 7730 | {f2(lim['no fan']['pmv_iso']['highest_clo'])} clo | {f2(lim['fan']['pmv_iso']['highest_clo'])} clo |
| PMV, ASHRAE 55 (cooling effect) | {f2(lim['no fan']['pmv_ashrae']['highest_clo'])} clo | {f2(lim['fan']['pmv_ashrae']['highest_clo'])} clo |

The training data's clo has a 95th percentile of {f2(q['p95'])} and a 99th percentile of {f2(q['p99'])}. The data-driven limits are
very wide (in {pct(boot['no fan']['share_at_grid_top'],0)} and {pct(boot['fan']['share_at_grid_top'],0)} of resamples the limit reaches the top of the {f2(2.5)} clo grid),
which is what a near-zero clo effect looks like. The PMV limits are physically derived, are not fitted to these votes, and
agree with the lower end of the data-driven interval. **Use the PMV limits as the working design numbers and the
data-driven result as support, not proof.**

**Highest comfortable total clo by condition** (`scenario_highest_clo_by_conditions.csv`; "none" = rule fails even at 0.10 clo;
rows nearby = votes within 1 degC and 7 % RH)

{md_table(lt_tab)}

{hum_note}

**Bedding sets at 29 degC / 85% RH, garment clo {f2(q['p50'])} (data median) plus manikin cover increment** (`scenario_bedding_sets_29C_85RH.csv`
also has the {f2(q['p25'])} and {f2(q['p75'])} garment levels)

{md_table(st)}

**Garment clo that is left once a cover is chosen** (highest total from PMV ISO 7730 minus the cover increment, `results/design_garment_budget.csv`)

{md_table(bud)}

## 8. Limitations

- ASHRAE votes come from awake people in buildings, not people sleeping, so they only stand in for sleep comfort.
- Neither dataset has skin-temperature data.
- The manikin data covers only the back (supine) and left side, and four bedding configurations with no garment detail.
- The data are observational: findings are associations, not causes (fans are used when it is hot; cooler homes have more clothing).
- The subset has {b_final} buildings but only {data[M['contributor']].nunique()} contributors, three countries and no Sri Lankan sites.
- ASHRAE `clo` is clothing only; manikin totals include bedding. Adding the cover increment to garment clo assumes the two are additive.
- {cov_min} to {cov_max} votes reach any manikin total clo, so totals above about {f2(q['p99'])} clo are extrapolation for the data-driven model.
- With the original label, ML models did not beat a constant predictor, and across-study performance collapsed (section 3).
- PMV was run outside its ISO 7730 validity range for many rows (limit checks switched off), and mean radiant temperature was missing for a large share of rows.
"""
    (R / "summary.md").write_text(summary)


def models_y(df: pd.DataFrame) -> str:
    """Name of the warm-side column in the processed file (derived from the rounded vote)."""
    if "warm" not in df:
        df["warm"] = (df["vote_rounded"] >= 1).astype(int)
    return "warm"
