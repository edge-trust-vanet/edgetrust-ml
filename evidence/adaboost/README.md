# Model Selection Evidence: Why AdaBoost for Size and Speed

AdaBoost and Random Forest **tie** on the dashboard holdout split (95.71% weighted F1, 0 missed attacks). This folder is the edge-deployment tie-break: AdaBoost is the model we take because it keeps that detection quality at a fraction of the memory and a lower inference cost than the remaining high-accuracy models.

Plots are generated from the 20-split means in [`results/model_comparison_report.json`](../../results/model_comparison_report.json) (17 classifiers).

## Files

- [`generate_adaboost_evidence.py`](generate_adaboost_evidence.py): Builds the size/speed figures from the saved benchmark.
- [`adaboost_size_speed.csv`](adaboost_size_speed.csv): Per-model F1, size, latency, and multiples vs AdaBoost.
- [`adaboost_size_speed.json`](adaboost_size_speed.json): Selection rule and the RF / Extra Trees / Voting comparisons.
- [`plots/adaboost_size_speed.png`](plots/adaboost_size_speed.png): Main 4-panel figure (Pareto, size, latency, F1).
- [`plots/adaboost_size_speed_ratios.png`](plots/adaboost_size_speed_ratios.png): How many times larger or slower every other model is.

Copies also land in `results/` next to the other paper charts.

## Selection Rule

Among models with **20-split weighted F1 ≥ 95.4%**, choose the **smallest** serialized model that still infers in **under 10 µs/sample**.

Only AdaBoost sits in that compact + fast + high-F1 box.

## Headline Numbers (20 stratified 80/20 splits)

| Model | Mean F1 | Mean FN / 1000 | Size | Latency | vs AdaBoost size | vs AdaBoost latency |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: |
| **AdaBoost (chosen)** | **95.57%** | **0.30** | **28.2 KB** | 3.89 µs | 1.0× | 1.0× |
| Voting Ensemble | 95.53% | 0.70 | 2,718.5 KB | 11.59 µs | 96× larger | 3.0× slower |
| Stacking Ensemble | 95.49% | 1.20 | 2,184.7 KB | 27.65 µs | 77× larger | 7.1× slower |
| Random Forest | 95.48% | 1.20 | 2,530.2 KB | 5.87 µs | 90× larger | 1.5× slower |
| CatBoost | 95.44% | 1.50 | 125.2 KB | 0.61 µs | 4.4× larger | 6.4× faster |
| XGBoost | 95.42% | 1.80 | 243.3 KB | 0.79 µs | 8.6× larger | 4.9× faster |
| Extra Trees | 94.49% | 11.30 | 9,519.7 KB | 8.43 µs | 337× larger | 2.2× slower |
| Decision Tree | 94.77% | 10.50 | 6.8 KB | 0.12 µs | smaller | faster, but misses attacks |
| Logistic Regression | 92.24% | 37.80 | 0.8 KB | 0.14 µs | smaller | faster, but misses attacks |

## Why AdaBoost, Not The Rest

**Random Forest / Extra Trees / Voting / Stacking.** Same or slightly worse F1, but they store deep bags of trees (or several full ensembles). Random Forest is ~90× larger; Extra Trees is ~337× larger. Too heavy for an RSU flash/RAM budget.

**XGBoost / CatBoost / LightGBM / Gradient Boosting.** Competitive F1 and often faster inference, but 4–9× larger than AdaBoost and they miss more attacks (1.5–3.5 FN vs AdaBoost’s 0.30). AdaBoost wins the size-constrained high-recall slot.

**Decision Tree / Logistic Regression / Naive Bayes.** Genuinely tiny and fast, but they fall out of the high-F1 band. DT averages 10.5 missed attacks; LR averages 37.8. Size cannot come at the cost of letting malicious nodes through.

**SVM / KNN.** Kernel distances and lazy nearest-neighbor search are 8–13× slower than AdaBoost and weaker on F1. Not viable for per-beacon RSU scoring.

**Why the pickle is 28 KB.** AdaBoost uses 50 depth-1 stumps. Each stump is one feature index, one threshold, and a weight — not a fully grown tree. Random Forest grows 100 deep trees until leaves are pure, so the serialized object is megabytes.

## Paper Wording

> Although Random Forest and AdaBoost achieved identical metrics on the initial holdout split (95.71% weighted F1, zero false negatives), repeated stratified splits showed AdaBoost obtaining the highest mean F1 (95.57%) and the lowest missed-attack count (0.30). AdaBoost requires 3.89 µs per sample and a 28.2 KB model — a 90× size reduction versus Random Forest (2.53 MB) and a 337× reduction versus Extra Trees (9.52 MB). Tiny linear and single-tree baselines are smaller still, but they drop below 95% F1 and miss far more attacks. Therefore AdaBoost is selected as the edge classifier for resource-constrained roadside units.

## Re-running

```bash
python scripts/compare_all_models.py          # only if the JSON benchmark is missing
python evidence/adaboost/generate_adaboost_evidence.py
```
