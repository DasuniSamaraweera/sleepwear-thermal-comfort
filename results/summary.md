# Results summary

All numbers below were produced by the code in `src/` (stage 4 driver: `src/stage4.py`) and are stored in `results/`.
Datasets (both CC BY 4.0): ASHRAE Global Thermal Comfort Database II v2.1 (https://zenodo.org/records/4993952) and
Heat balance of sleeping sets, Newton thermal manikin, Brno University of Technology (https://zenodo.org/records/20179324).

## 1. Data used (`results/filter_log.csv`)

| Step | Rows before | Rows after | Dropped | Buildings after |
|---|---|---|---|---|
| 0. raw measurements joined with metadata | 109033 | 109033 | 0 | 809 |
| 1. quality assurance | 109033 | 108235 | 798 | 805 |
| 2. hot-humid climate | 108235 | 38397 | 69838 | 350 |
| 3. residential | 38397 | 5211 | 33186 | 199 |
| 4. low metabolic rate | 5211 | 3356 | 1855 | 180 |
| 5. complete cases: thermal_sensation | 3356 | 3355 | 1 | 180 |
| 5. complete cases: ta | 3355 | 3355 | 0 | 180 |
| 5. complete cases: rh | 3355 | 3020 | 335 | 175 |
| 5. complete cases: vel | 3020 | 2379 | 641 | 165 |
| 5. complete cases: clo | 2379 | 2372 | 7 | 165 |
| 6. plausible range: ta | 2372 | 2369 | 3 | 165 |
| 6. plausible range: rh | 2369 | 2369 | 0 | 165 |
| 6. plausible range: vel | 2369 | 2334 | 35 | 164 |
| 6. plausible range: clo | 2334 | 2307 | 27 | 164 |

Final subset: 2,307 votes from 164 buildings, 4 contributors. China 127 rows (2 buildings), India 1,341 rows (154 buildings), Malaysia 839 rows (8 buildings).
There are no Sri Lankan records. Share of votes that are warm-side discomfort (rounded sensation vote of +1 or more):
27.1%.

## 2. Target definition (changed after the first modelling pass)

The first target, "comfortable = vote from -1 to +1", is almost unpredictable from the inputs (section 3, negative result).
It lumps warm and cold votes together. The main target is therefore **warm-side discomfort: rounded sensation vote of
+1 or more**, which is also the relevant failure for hot nights.

## 3. Negative result with the original "comfortable" label (`results/neg_comfort_label_*.csv`)

Held-out buildings: ROC AUC 0.634 (XGBoost), 0.628 (quadratic logistic),
0.545 (linear logistic), 0.582 (PMV ASHRAE 55). Accuracy of every ML model
(0.820 for XGBoost) equals always predicting "comfortable" (0.819).
Leaving one contributor out, AUC fell to 0.356 (linear logistic), 0.502 (quadratic) and
0.444 (XGBoost). Sensation MAE in that check was 0.770 (ridge) to 0.791 (XGBoost) against 0.791 for predicting the mean vote, with overlapping intervals, so there was essentially no gain.

## 4. Model comparison against PMV, main target (warm-side discomfort)

Inputs for all models: clo, air temperature, relative humidity, air speed, met (PMV also uses mean radiant temperature,
falling back to air temperature where missing). "Macro F1 (matched share)" calls the training-prevalence share of rows
in each test fold "warm" for every model, PMV included, using no test labels.

**Validation 1: building-grouped 5-fold** (StratifiedGroupKFold by `building_id`; intervals from resampling buildings)

| Model | Accuracy | Macro F1 | Macro F1 (matched share) | ROC AUC | ROC AUC 95% CI |
|---|---|---|---|---|---|
| PMV (ISO 7730) | 0.504 | 0.504 | 0.671 | 0.761 | [0.724, 0.793] |
| PMV (ASHRAE 55, cooling effect) | 0.582 | 0.576 | 0.671 | 0.763 | [0.724, 0.800] |
| Logistic regression (linear) | 0.773 | 0.653 | 0.686 | 0.780 | [0.742, 0.817] |
| Logistic regression (quadratic) | 0.769 | 0.649 | 0.682 | 0.772 | [0.733, 0.810] |
| XGBoost | 0.765 | 0.654 | 0.678 | 0.774 | [0.731, 0.813] |
| Always predict the majority class (reference) | 0.729 | 0.422 | n/a | 0.500 | [0.500, 0.500] |

**Validation 2: leave one contributor out** (4 held-out studies)

| Model | Accuracy | Macro F1 | Macro F1 (matched share) | ROC AUC | ROC AUC 95% CI |
|---|---|---|---|---|---|
| PMV (ISO 7730) | 0.504 | 0.504 | 0.636 | 0.761 | [0.724, 0.793] |
| PMV (ASHRAE 55, cooling effect) | 0.582 | 0.576 | 0.628 | 0.763 | [0.724, 0.800] |
| Logistic regression (linear) | 0.774 | 0.659 | 0.645 | 0.777 | [0.737, 0.813] |
| Logistic regression (quadratic) | 0.771 | 0.650 | 0.628 | 0.738 | [0.691, 0.784] |
| XGBoost | 0.750 | 0.606 | 0.624 | 0.745 | [0.701, 0.786] |
| Always predict the majority class (reference) | 0.729 | 0.422 | n/a | 0.500 | [0.500, 0.500] |

**Sensation vote (continuous), building-grouped 5-fold**

| Model | MAE (vote units) | MAE 95% CI | Accuracy within +-1 | Within +-1 95% CI |
|---|---|---|---|---|
| PMV (ISO 7730) | 1.304 | [1.166, 1.421] | 0.624 | [0.572, 0.678] |
| PMV (ASHRAE 55, cooling effect) | 1.082 | [0.934, 1.207] | 0.744 | [0.690, 0.806] |
| Linear (ridge) | 0.760 | [0.651, 0.853] | 0.862 | [0.815, 0.919] |
| Quadratic (ridge) | 0.760 | [0.656, 0.851] | 0.863 | [0.817, 0.918] |
| XGBoost | 0.761 | [0.654, 0.852] | 0.863 | [0.814, 0.922] |
| Predict training mean (reference) | 0.791 | [0.680, 0.898] | 0.819 | [0.767, 0.877] |

**Sensation vote, leave one contributor out**

| Model | MAE (vote units) | MAE 95% CI | Accuracy within +-1 | Within +-1 95% CI |
|---|---|---|---|---|
| PMV (ISO 7730) | 1.304 | [1.166, 1.421] | 0.624 | [0.572, 0.678] |
| PMV (ASHRAE 55, cooling effect) | 1.082 | [0.934, 1.207] | 0.744 | [0.690, 0.806] |
| Linear (ridge) | 0.770 | [0.663, 0.866] | 0.863 | [0.814, 0.921] |
| Quadratic (ridge) | 0.777 | [0.664, 0.872] | 0.857 | [0.813, 0.910] |
| XGBoost | 0.791 | [0.680, 0.890] | 0.854 | [0.809, 0.907] |
| Predict training mean (reference) | 0.791 | [0.680, 0.898] | 0.819 | [0.767, 0.877] |

Reading: the linear logistic model has the best or near-best ROC AUC in both validations (0.780 and
0.777) and is the primary model. PMV ranks warm-side discomfort about as well (AUC 0.761
for ISO 7730, 0.763 for ASHRAE 55) with overlapping intervals, but its raw threshold is badly
calibrated (accuracy 0.504 at PMV >= +0.5, versus 0.729
for always predicting "not warm"). For the sensation vote, PMV has larger error (MAE 1.082 to
1.304) than predicting the mean vote (0.791).
The ML models reduce MAE only slightly (0.760 for ridge).

## 5. What matters most (`results/shap_global_importance.csv`, `permutation_importance_loco.csv`, `logistic_coefficients.csv`)

Mean absolute SHAP value as a share of the total (final models fit on all 2,307 rows; SHAP describes the model, not causes):

| Feature | SHAP share, XGBoost | SHAP share, logistic |
|---|---|---|
| ta | 64.2% | 59.2% |
| vel | 22.7% | 22.2% |
| rh | 7.4% | 8.7% |
| clo | 3.4% | 3.9% |
| met | 2.4% | 6.0% |

Held-out permutation importance (drop in ROC AUC when one input is shuffled, leave-one-contributor-out folds):

| Feature | AUC drop, logistic | AUC drop, XGBoost |
|---|---|---|
| ta | 0.214 | 0.203 |
| vel | 0.018 | 0.005 |
| rh | -0.001 | 0.003 |
| clo | 0.000 | 0.003 |
| met | 0.003 | -0.000 |

Logistic odds ratio of warm-side discomfort per one standard deviation: air temperature 3.09
(sd 2.61 degC), air speed 0.66 (sd 0.42 m/s),
relative humidity 0.85 (sd 10.53 %), clo 1.09
(sd 0.23 clo). The dependence plots (`figures/fig11_shap_dependence`) show a steep rise with
temperature above roughly 27 degC, a fall with air speed up to about 1 m/s, a small negative association with humidity
(the opposite of what physiology alone suggests, so it reflects who lives where, not humidity as a cause), and a small,
irregular clo pattern. Clo contributes 3.4% (XGBoost) and 3.9% (logistic) of total SHAP
importance, and shuffling it changes held-out AUC by about 0.003 at most. **This data barely identifies an effect of clothing insulation.**

## 6. Sleepwear insulation from the manikin data (`results/manikin_total_clo.csv`, `manikin_cover_increments.csv`)

| Set | Posture | Rtot (m2K/W) | Total clo | Uncertainty (clo) |
|---|---|---|---|---|
| Without cover | side | 0.173 | 1.12 | 0.02 |
| Cotton cover | side | 0.243 | 1.57 | 0.06 |
| Blanket | side | 0.257 | 1.66 | 0.06 |
| Summer duvet | side | 0.297 | 1.92 | 0.07 |
| Without cover | supine | 0.181 | 1.17 | 0.03 |
| Cotton cover | supine | 0.254 | 1.64 | 0.10 |
| Blanket | supine | 0.268 | 1.73 | 0.10 |
| Summer duvet | supine | 0.318 | 2.05 | 0.11 |

The "sets" are bedding configurations; the README does not say what garments were worn. Adding a cover to
"Without cover" adds 0.45 to 0.88 clo, while lying on the side instead of the back lowers the total by
0.05 to 0.13 clo. Torso-segment back values above 5 clo (`manikin_torso_segments.csv`) look like a
contact artefact and were not used.

## 7. Scenarios at 29 degC and 85% RH (`results/scenario_*.csv`)

Scenario inputs: sleeping met 0.7; "no fan" = 0.09 m/s and "fan" = 0.40 m/s (median measured speed in rows
where the fan flag is 0 / 1, from 509 and 1669 rows). Total insulation = garment clo + manikin cover
increment. Rule for the data-driven model: predicted P(warm-side discomfort) <= 20%. Rule for PMV: PMV < +0.5.
Support: 202 votes lie within 1 degC and 7 % RH of this point
(20 with the fan flag off, 172 with it on, the rest unrecorded), of which 25% (no fan) and 16% (fan) were
warm-side discomfort. Only 142 votes have met of 0.8 or lower, so the sleeping-met input is near the edge of the data.

**Highest total clo that stays comfortable at 29 degC / 85% RH**

| Basis | No fan (0.09 m/s) | Fan (0.40 m/s) |
|---|---|---|
| Data-driven model (Logistic regression (linear)) | 1.12 clo | 1.94 clo (beyond the data range: True) |
| ... 95% interval over buildings (bootstrap, 200 refits) | 0.37 to 2.50 clo | 1.00 to 2.50 clo |
| XGBoost (cross-check) | 2.50 clo (top of grid, never exceeds 20%) | 2.50 clo (top of grid) |
| PMV, ISO 7730 | 0.58 clo | 0.83 clo |
| PMV, ASHRAE 55 (cooling effect) | 0.58 clo | 0.94 clo |

The training data's clo has a 95th percentile of 0.78 and a 99th percentile of 1.29. The data-driven limits are
very wide (in 15% and 32% of resamples the limit reaches the top of the 2.50 clo grid),
which is what a near-zero clo effect looks like. The PMV limits are physically derived, are not fitted to these votes, and
agree with the lower end of the data-driven interval. **Use the PMV limits as the working design numbers and the
data-driven result as support, not proof.**

**Highest comfortable total clo by condition** (`scenario_highest_clo_by_conditions.csv`; "none" = rule fails even at 0.10 clo;
rows nearby = votes within 1 degC and 7 % RH)

| Air temp (degC) | RH (%) | Air speed | Rows nearby | ML limit (clo) | PMV ISO limit | PMV ASHRAE limit |
|---|---|---|---|---|---|---|
| 27 | 85 | no fan | 133 | 2.50 | 1.18 | 1.18 |
| 27 | 85 | fan | 133 | 2.50 | 1.42 | 1.53 |
| 28 | 85 | no fan | 229 | 2.31 | 0.87 | 0.87 |
| 28 | 85 | fan | 229 | 2.50 | 1.12 | 1.23 |
| 29 | 85 | no fan | 202 | 1.12 | 0.58 | 0.58 |
| 29 | 85 | fan | 202 | 1.94 | 0.83 | 0.94 |
| 30 | 85 | no fan | 110 | none | 0.28 | 0.28 |
| 30 | 85 | fan | 110 | 0.75 | 0.55 | 0.66 |
| 31 | 85 | no fan | 48 | none | none | none |
| 31 | 85 | fan | 48 | none | 0.28 | 0.39 |
| 29 | 60 | no fan | 103 | none | 0.80 | 0.80 |
| 29 | 60 | fan | 103 | 0.91 | 1.04 | 1.23 |
| 29 | 70 | no fan | 326 | 0.50 | 0.70 | 0.70 |
| 29 | 70 | fan | 326 | 1.32 | 0.95 | 1.10 |
| 29 | 80 | no fan | 422 | 0.91 | 0.62 | 0.62 |
| 29 | 80 | fan | 422 | 1.73 | 0.87 | 0.99 |

Caution on humidity: at 29 degC without a fan the data-driven limit 0.50 clo at 70% RH becomes 1.12 clo at 85% RH (it rises with humidity), while the PMV limit falls from 0.70 to 0.58 clo. The fitted humidity effect is negative (odds ratio 0.85 per sd), which reflects who lives where in this subset, so do not read the data-driven humidity trend as physiology.

**Bedding sets at 29 degC / 85% RH, garment clo 0.37 (data median) plus manikin cover increment** (`scenario_bedding_sets_29C_85RH.csv`
also has the 0.27 and 0.51 garment levels)

| Set | Posture | Total clo | Air | ML P(warm) | PMV ISO | PMV ASHRAE | ML ok | ISO ok | ASHRAE ok |
|---|---|---|---|---|---|---|---|---|---|
| Without cover | side | 0.37 | no fan | 16.0% | 0.03 | 0.03 | yes | yes | yes |
| Without cover | side | 0.37 | fan | 12.3% | -0.87 | -0.81 | yes | yes | yes |
| Cotton cover | side | 0.82 | no fan | 18.3% | 0.88 | 0.88 | yes | no | no |
| Cotton cover | side | 0.82 | fan | 14.2% | 0.47 | 0.28 | yes | yes | yes |
| Blanket | side | 0.91 | no fan | 18.8% | 1.00 | 1.00 | yes | no | no |
| Blanket | side | 0.91 | fan | 14.6% | 0.64 | 0.45 | yes | no | yes |
| Summer duvet | side | 1.17 | no fan | 20.3% | 1.29 | 1.29 | no | no | no |
| Summer duvet | side | 1.17 | fan | 15.9% | 1.03 | 0.83 | yes | no | no |
| Without cover | supine | 0.37 | no fan | 16.0% | 0.03 | 0.03 | yes | yes | yes |
| Without cover | supine | 0.37 | fan | 12.3% | -0.87 | -0.81 | yes | yes | yes |
| Cotton cover | supine | 0.84 | no fan | 18.4% | 0.91 | 0.91 | yes | no | no |
| Cotton cover | supine | 0.84 | fan | 14.3% | 0.51 | 0.32 | yes | no | yes |
| Blanket | supine | 0.93 | no fan | 18.9% | 1.03 | 1.03 | yes | no | no |
| Blanket | supine | 0.93 | fan | 14.7% | 0.67 | 0.48 | yes | no | yes |
| Summer duvet | supine | 1.25 | no fan | 20.7% | 1.37 | 1.37 | no | no | no |
| Summer duvet | supine | 1.25 | fan | 16.2% | 1.13 | 0.92 | yes | no | no |

**Garment clo that is left once a cover is chosen** (highest total from PMV ISO 7730 minus the cover increment, `results/design_garment_budget.csv`)

| condition | set | posture | cover_increment | garment_clo_budget_pmv_iso | fits_training_garment_range |
|---|---|---|---|---|---|
| no fan | Without cover | side | 0.00 | 0.58 | yes |
| no fan | Cotton cover | side | 0.45 | 0.13 | yes |
| no fan | Blanket | side | 0.54 | 0.04 | no garment fits |
| no fan | Summer duvet | side | 0.80 | 0.00 | no garment fits |
| no fan | Without cover | supine | 0.00 | 0.58 | yes |
| no fan | Cotton cover | supine | 0.47 | 0.11 | yes |
| no fan | Blanket | supine | 0.56 | 0.02 | no garment fits |
| no fan | Summer duvet | supine | 0.88 | 0.00 | no garment fits |
| fan | Without cover | side | 0.00 | 0.83 | yes |
| fan | Cotton cover | side | 0.45 | 0.38 | yes |
| fan | Blanket | side | 0.54 | 0.29 | yes |
| fan | Summer duvet | side | 0.80 | 0.03 | no garment fits |
| fan | Without cover | supine | 0.00 | 0.83 | yes |
| fan | Cotton cover | supine | 0.47 | 0.36 | yes |
| fan | Blanket | supine | 0.56 | 0.27 | yes |
| fan | Summer duvet | supine | 0.88 | 0.00 | no garment fits |

## 8. Limitations

- ASHRAE votes come from awake people in buildings, not people sleeping, so they only stand in for sleep comfort.
- Neither dataset has skin-temperature data.
- The manikin data covers only the back (supine) and left side, and four bedding configurations with no garment detail.
- The data are observational: findings are associations, not causes (fans are used when it is hot; cooler homes have more clothing).
- The subset has 164 buildings but only 4 contributors, three countries and no Sri Lankan sites.
- ASHRAE `clo` is clothing only; manikin totals include bedding. Adding the cover increment to garment clo assumes the two are additive.
- 11 to 29 votes reach any manikin total clo, so totals above about 1.29 clo are extrapolation for the data-driven model.
- With the original label, ML models did not beat a constant predictor, and across-study performance collapsed (section 3).
- PMV was run outside its ISO 7730 validity range for many rows (limit checks switched off), and mean radiant temperature was missing for a large share of rows.
