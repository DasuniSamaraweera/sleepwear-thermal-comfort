"""Build notebooks/analysis.ipynb from src/*.py (single source of truth).

Each src file is embedded in a %%writefile cell so the notebook is
self-contained on Colab. Run:  python tools/build_notebook.py
"""
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
SRC_FILES = ["config.py", "download.py", "inspect_data.py", "clean.py", "manikin.py", "explore.py", "models.py", "explain.py", "scenarios.py", "stage4.py", "report.py"]


def md(text):
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text):
    return nbf.v4.new_code_cell(text.strip("\n"))


cells = [
    md("""
# Modelling Nocturnal Thermal Comfort of Sleepwear in Hot-Humid Climates

Staged run. **Stage 1** downloads and inspects both datasets (reviewed). **Stage 2** filters and cleans ASHRAE with a full row-count log, builds the manikin clo tables and exploratory figures (reviewed). **Stage 3** adds the PMV/PPD baseline and the models with the original comfort label (a documented negative result). **Stage 4** (this version) uses the warm-side discomfort target and adds SHAP, scenarios and the written reports. If pip asks you to restart the runtime after the first cell, restart and Run all again.

How to run on Colab: *Runtime > Run all* (a CPU runtime is enough). The last cell zips the
final results and downloads `stage4_outputs.zip`. Send that zip back so I can check it matches.

Data (both CC BY 4.0):
- ASHRAE Global Thermal Comfort Database II v2.1, https://zenodo.org/records/4993952
- Heat balance of sleeping sets (Newton thermal manikin), https://zenodo.org/records/20179324
"""),
    code("""
import os, sys
from pathlib import Path

IN_COLAB = "google.colab" in sys.modules
USE_DRIVE = False   # set True to keep everything in Google Drive so it survives runtime resets

if USE_DRIVE and IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    BASE = Path("/content/drive/MyDrive")
elif IN_COLAB:
    BASE = Path("/content")
else:
    BASE = Path(os.environ.get("SLEEPWEAR_BASE", Path.cwd()))

PROJECT = BASE / "sleepwear-comfort"
for d in ["src", "data/raw", "results", "figures", "notebooks"]:
    (PROJECT / d).mkdir(parents=True, exist_ok=True)
os.chdir(PROJECT)
sys.path.insert(0, str(PROJECT / "src"))
(PROJECT / ".gitignore").write_text("data/raw/\\n__pycache__/\\n*.zip\\n.ipynb_checkpoints/\\n")
print("Project folder:", PROJECT)
"""),
    code("%pip install -q openpyxl requests pythermalcomfort==4.8.1 xgboost==3.4.1 shap==0.54.1 scikit-learn"),
    md("## Source files\nThese cells write `src/` so the notebook is self-contained."),
]

for name in SRC_FILES:
    body = (ROOT / "src" / name).read_text()
    cells.append(code(f"%%writefile src/{name}\n{body}"))

cells += [
    md("## 1. Download both datasets into `data/raw/`\nFile names are discovered from the Zenodo API, "
       "checksums are verified, zips are extracted."),
    code("""
import importlib, config, download
importlib.reload(config); importlib.reload(download)
download.main()
"""),
    md("## 2. Inspect: ASHRAE Global Thermal Comfort Database II"),
    code("""
import inspect_data
importlib.reload(inspect_data)
tables = {}
tables["ashrae"] = inspect_data.inspect_dataset("ashrae")
"""),
    md("## 3. Inspect: manikin heat-balance data"),
    code("""
tables["manikin"] = inspect_data.inspect_dataset("manikin")
"""),
    md("## 4. Candidate columns (hints only, to be confirmed by a human)"),
    code("""
cand = inspect_data.write_candidates(tables)
print(cand.to_string(index=False) if len(cand) else "No candidates found")
"""),
    md("## 5. Package the inspection output\nRaw data is **not** included (it is large and public); "
       "only the reports and Zenodo record metadata are zipped."),
    code("""
import zipfile
out_zip = PROJECT / "stage1_outputs.zip"
with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as z:
    for p in (PROJECT / "results" / "inspection").rglob("*"):
        if p.is_file():
            z.write(p, p.relative_to(PROJECT))
    for p in (PROJECT / "data" / "raw").glob("*/record_metadata.json"):
        z.write(p, p.relative_to(PROJECT))
    listing = "\\n".join(f"{p.relative_to(PROJECT)}\\t{p.stat().st_size}" for p in sorted((PROJECT / "data" / "raw").rglob("*")) if p.is_file())
    z.writestr("data_raw_file_listing.tsv", listing)
print("Wrote", out_zip, f"({out_zip.stat().st_size/1024:.0f} kB)")
if IN_COLAB:
    from google.colab import files
"""),
]


cells += [
    md("---\n# Stage 2: filtering, cleaning, manikin clo, exploration\n"
       "Decisions in force (set in `src/config.py`): hot-humid = climate in the explicit class list in `config.HOT_HUMID_CLASSES`; "
       "residential = `multifamily housing`; low activity = met <= 1.3; comfortable = rounded sensation vote in [-1, +1]."),
    md("## 6. ASHRAE: climate classes, filters and cleaning log"),
    code("""
import pandas as pd
import clean
importlib.reload(config); importlib.reload(clean)
pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_rows", 100)

raw = clean.load_ashrae()
print("\\nClimate classes (all rows) and whether the hot-humid rule matches them:")
clim = clean.climate_report(raw)
print(clim.to_string(index=False))
print("\\nBuilding types:"); print(raw[config.COLUMN_MAP["ashrae"]["building_type"]].value_counts(dropna=False).to_string())
"""),
    code("""
d, flog = clean.build_subset(raw)
print(flog.to_string(index=False))
sens = clean.met_sensitivity(raw)
print("\\nSensitivity: rows after QA + hot-humid + residential, by met cutoff"); print(sens.to_string(index=False))

(config.RESULTS).mkdir(exist_ok=True)
flog.to_csv(config.RESULTS / "filter_log.csv", index=False)
clim.to_csv(config.RESULTS / "climate_classes.csv", index=False)
sens.to_csv(config.RESULTS / "met_sensitivity.csv", index=False)
print("\\nSaved", clean.save_processed(d))
"""),
    code("""
M = config.COLUMN_MAP["ashrae"]
num = [M["air_temp"], M["rel_humidity"], M["air_speed"], M["clo"], M["met"], M["thermal_sensation"]]
summ = d[num].describe().T
print("Final subset:", f"{len(d):,} rows,", d[M["building_id"]].nunique(), "buildings,",
      d[M["contributor"]].nunique(), "contributors,", d[M["country"]].nunique(), "countries")
print(summ.to_string())
print("\\nRounded vote counts:"); print(d["vote_rounded"].value_counts().sort_index().to_string())
print("\\nShare comfortable (vote in [-1,+1]): %.3f" % d["comfortable"].mean())
print("\\nRows per country / climate / building:")
print(d[M["country"]].value_counts().to_string()); print(d[M["climate"]].value_counts().to_string())
print(d.groupby(M["building_id"]).size().describe().to_string())
print("\\nAcceptability available for", int(d[M["thermal_acceptability"]].notna().sum()), "rows;",
      "fan recorded for", int(d[M["fan"]].notna().sum()), "rows")
summ.to_csv(config.RESULTS / "subset_summary_stats.csv")
d.groupby(M["country"]).agg(rows=("comfortable", "size"), buildings=(M["building_id"], "nunique"),
                            share_comfortable=("comfortable", "mean")).to_csv(config.RESULTS / "subset_by_country.csv")
"""),
    md("## 7. Manikin data: total clo by sleeping set and posture\n"
       "The sets are bedding configurations, not garments. Torso-segment values above 5 clo are flagged as suspect."),
    code("""
import manikin
importlib.reload(manikin)
clo_tbl, reps, seg = manikin.build_clo_table()
print(clo_tbl.to_string(index=False))
print("\\nTorso segments (suspect_artifact = clo > 5):"); print(seg.to_string(index=False))
clo_tbl.to_csv(config.RESULTS / "manikin_total_clo.csv", index=False)
reps.to_csv(config.RESULTS / "manikin_replicates.csv", index=False)
seg.to_csv(config.RESULTS / "manikin_torso_segments.csv", index=False)

# How much of the ASHRAE training data reaches the manikin clo range?
rows = []
for _, r in clo_tbl.iterrows():
    rows.append({"set": r["set"], "posture": r["posture"], "clo_total": r["clo_total"],
                 "ashrae_rows_with_clo_ge_this": int((d[M["clo"]] >= r["clo_total"]).sum()),
                 "share": round(float((d[M["clo"]] >= r["clo_total"]).mean()), 4)})
cov = pd.DataFrame(rows); print("\\nTraining-data coverage of manikin clo values:"); print(cov.to_string(index=False))
cov.to_csv(config.RESULTS / "clo_coverage.csv", index=False)
print("ASHRAE subset clo: max = %.2f, 95th pct = %.2f, 99th pct = %.2f" % (d[M["clo"]].max(), d[M["clo"]].quantile(.95), d[M["clo"]].quantile(.99)))
"""),
    md("## 8. Exploratory figures (PNG 300 dpi + SVG in `figures/`)"),
    code("""
import explore
importlib.reload(explore)
explore.run_all(d, clo_tbl)
from IPython.display import Image, display
for f in sorted(config.FIGURES.glob("fig0*.png")):
    print(f.name); display(Image(filename=str(f), width=520))
"""),
    md("---\n# Stage 3: PMV baseline and ML models\n"
       "Same inputs for every model: clo, air temperature, RH, air speed, met (PMV also uses mean radiant temperature, "
       "falling back to air temperature where missing). Validation is StratifiedGroupKFold by `building_id` "
       "(never random rows), plus leave-one-contributor-out as a robustness check. Intervals are cluster bootstrap over buildings."),
    code("""
import models
importlib.reload(models)
df = models.set_target(models.load_clean(), "comfortable")   # ORIGINAL label (negative result)
pmv_chk = models.pmv_baseline(df)
print("rows:", len(df), "| buildings:", df[M["building_id"]].nunique(), "| rows with tr imputed from air temp:", pmv_chk.attrs["n_tr_imputed"])
print("ISO PMV rows with air speed > 1.0 m/s or air temp > 30 C (outside the ISO 7730 validity range):",
      int(((df[M["air_speed"]] > 1.0) | (df[M["air_temp"]] > 30)).sum()))
splits = models.building_splits(df)
print("Fold sizes (train, test):", [(len(a), len(b)) for a, b in splits])
"""),
    md("## 10. Building-grouped 5-fold validation (main result)"),
    code("""
cls_b, sens_b, fold_b, oof_b, best_b = models.run_experiment(df, splits, "building 5-fold")
show_c = ["model", "accuracy", "macro_f1", "macro_f1_matched", "roc_auc", "roc_auc_ci95", "macro_f1_matched_ci95"]
print("COMFORT (vote in [-1,+1])"); print(cls_b[show_c].round(3).to_string(index=False))
print("\\nSENSATION VOTE"); print(sens_b[["model", "mae", "mae_ci95", "within_1", "within_1_ci95"]].round(3).to_string(index=False))
print("\\nChosen hyperparameters per fold (XGBoost classifier):", best_b["classifiers"]["XGBoost"])
print("Chosen hyperparameters per fold (quadratic logistic C):", [b["logisticregression__C"] for b in best_b["classifiers"]["Logistic regression (quadratic)"]])
"""),
    md("## 11. Robustness: leave one contributor out (4 folds)"),
    code("""
csplits = models.contributor_splits(df)
print("Held-out contributors:", [(df[M["contributor"]].iloc[te[0]], len(te)) for _, te in csplits])
cls_c, sens_c, fold_c, oof_c, best_c = models.run_experiment(df, csplits, "leave-one-contributor-out")
print(cls_c[show_c].round(3).to_string(index=False))
print(sens_c[["model", "mae", "mae_ci95", "within_1", "within_1_ci95"]].round(3).to_string(index=False))
print("\\nPer-fold (held-out contributor) ROC AUC:")
print(fold_c[fold_c.task == "comfort"].pivot(index="model", columns="fold", values="roc_auc").round(3).to_string())
"""),
    md("## 12. Sensitivity: label = thermal acceptability (instead of sensation vote)"),
    code("""
acc = df[M["thermal_acceptability"]].map({"acceptable": 1, "unacceptable": 0})
print("Rows with acceptability:", int(acc.notna().sum()), "| share acceptable: %.3f" % acc.mean())
d_acc = df[acc.notna()].copy(); d_acc[models.Y] = acc[acc.notna()].astype(int).to_numpy()
cls_a, _, _, _, _ = models.run_experiment(d_acc, models.building_splits(d_acc), "acceptability label")
print(cls_a[show_c].round(3).to_string(index=False))
"""),
    md("## 13. Figures and saved results"),
    code("""
for nm, t in (("neg_comfort_label_model_comparison", pd.concat([cls_b, cls_c, cls_a])), ("neg_comfort_label_sensation_comparison", pd.concat([sens_b, sens_c])),
              ("neg_comfort_label_fold_metrics", pd.concat([fold_b, fold_c]))):
    t.to_csv(config.RESULTS / f"{nm}.csv", index=False)
oof_b.to_csv(config.RESULTS / "neg_comfort_label_oof_predictions.csv.gz", index=False)
explore.fig_roc(oof_b); explore.fig_calibration(oof_b)
for f in ("fig07_roc_curves.png", "fig08_calibration.png"):
    print(f); display(Image(filename=str(config.FIGURES / f), width=480))
"""),
    md("---\\n# Stage 4: warm-side discomfort target, SHAP, scenarios, reports\\n"
       "Main target = rounded sensation vote of +1 or more (decided after the stage 3 negative result). "
       "Runs the models against PMV, SHAP and permutation importance, the 29 degC / 85% RH scenarios with and without a fan "
       "(data-driven model and PMV side by side), then writes `results/summary.md` and `results/design_garment_budget.csv` from the saved numbers."),
    code("""
import explain, scenarios, stage4, report
for m in (explain, scenarios, stage4, report): importlib.reload(m)
out = stage4.run()
print("Primary model:", out["primary_model"])
print(out["auc_by_validation"].round(3).to_string())
"""),
    code("""
show_c = ["validation", "model", "accuracy", "macro_f1", "macro_f1_matched", "roc_auc", "roc_auc_ci95"]
print(out["cls"][show_c].round(3).to_string(index=False))
print(out["sens"][["validation", "model", "mae", "mae_ci95", "within_1", "within_1_ci95"]].round(3).to_string(index=False))
print(out["shap"].round(3).to_string(index=False)); print(out["perm"].round(4).to_string(index=False)); print(out["coef"].round(3).to_string(index=False))
print("Fan levels:", out["fan"]); print("Highest comfortable total clo at 29 C / 85% RH:", out["limits_29_85"])
print("Bootstrap:", out["boot_29_85"])
print(out["limits_table"].round(2).to_string(index=False))
"""),
    code("""
report.build_all()
for f in ("fig09_shap_global_importance", "fig11_shap_dependence", "fig12_scenario_curves_29C_85RH", "fig13_scenario_heatmaps_temp_rh", "fig14_scenario_bedding_sets"):
    print(f); display(Image(filename=str(config.FIGURES / f"{f}.png"), width=620))
print(open(config.RESULTS / "summary.md").read()[:3000])
"""),
    md("## Package everything"),
    code("""
import zipfile
out4 = PROJECT / "stage4_outputs.zip"
with zipfile.ZipFile(out4, "w", zipfile.ZIP_DEFLATED) as z:
    for sub in ("results", "figures", "data/processed", "src", "notebooks"):
        for p in (PROJECT / sub).rglob("*"):
            if p.is_file() and "inspection" not in p.parts and "__pycache__" not in p.parts:
                z.write(p, p.relative_to(PROJECT))
    for name in ("README.md", "requirements.txt"):
        if (PROJECT / name).exists(): z.write(PROJECT / name, name)
print("Wrote", out4, f"({out4.stat().st_size/1024:.0f} kB)")
if IN_COLAB:
    files.download(str(out4))
"""),
]

nb = nbf.v4.new_notebook()
nb.cells = cells
nb.metadata = {
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"},
    "colab": {"provenance": []},
}
out = ROOT / "notebooks" / "analysis.ipynb"
nbf.write(nb, out)
print("wrote", out)
