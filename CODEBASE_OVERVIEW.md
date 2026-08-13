# EdgeTrust-VANET Codebase Overview

This repository is a Python/Flask machine learning demo for detecting malicious behavior in Vehicular Ad-Hoc Networks (VANETs). The core idea is to simulate a Roadside Unit (RSU) that combines trust-based heuristics with supervised ML classifiers to decide whether a vehicle should be accepted, warned, or blocked.

The project currently contains three main layers:

1. Data and offline ML training in `data/`, `scripts/`, and `results/`.
2. A Flask API in `dashboard/app.py`.
3. A browser dashboard in `dashboard/templates/` and `dashboard/static/`.

## Repository Layout

```text
.
├── README.md
├── CODEBASE_OVERVIEW.md
├── requirements.txt
├── data/
│   ├── vanet_malicious_nodes.csv
│   ├── X_train.npy / X_test.npy / y_train.npy / y_test.npy
│   └── vanet_X_train.npy / vanet_X_test.npy / vanet_y_train.npy / vanet_y_test.npy
├── scripts/
│   ├── analyze_vanet_nodes.py
│   ├── preprocess.py
│   ├── train_model.py
│   ├── trust_score.py
│   ├── combined_decision.py
│   └── visualize_results.py
├── dashboard/
│   ├── app.py
│   ├── templates/index.html
│   └── static/
│       ├── app.js
│       └── style.css
├── results/
│   ├── model_metrics.json
│   ├── vanet_model_metrics.json
│   └── generated charts
├── survey/
│   └── survey HTML, CSS, extracted text, and PDF
└── ppt and report/
    └── presentation/report PDFs
```

There is no committed `models/` directory. The training scripts create it locally and write `.pkl` model/scaler artifacts into it. `.pkl` files are ignored by `.gitignore`.

## What The Project Is Doing

The project trains binary classifiers to predict whether a vehicle/node is malicious. The main dataset is `data/vanet_malicious_nodes.csv`, which has 5,000 rows and columns for:

- Mobility signals: position, speed, direction, acceleration.
- Network signals: sent/received packets, drop ratio, latency, retransmissions, signal strength.
- Trust signals: current trust score, neighbor trust average, historical trust score.
- Post-hoc attack counters: false packet injection, blackhole attempts, Sybil attempts, denial of service.
- Target label: `is_malicious`.

The intended architecture is:

```text
Vehicle telemetry + network behavior
        |
        v
RSU trust / plausibility logic
        |
        v
Feature vector for ML classifier
        |
        v
Prediction + trust score
        |
        v
Final decision: ACCEPT, WARN, or BLOCK
```

## Main ML Workflows

### Rich VANET Workflow

`scripts/analyze_vanet_nodes.py` is the primary training script for the richer VANET dataset.

It performs these steps:

1. Loads `data/vanet_malicious_nodes.csv`.
2. Defines 14 input features.
3. Drops missing rows.
4. Standard-scales the feature matrix.
5. Splits into train/test sets with stratification.
6. Saves the split arrays as `data/vanet_*.npy`.
7. Trains nine classifiers:
   - Random Forest
   - Gradient Boosting
   - Extra Trees
   - AdaBoost
   - Decision Tree
   - SVM (RBF)
   - K-Nearest Neighbors
   - Logistic Regression
   - Gaussian Naive Bayes
8. Selects the best model by weighted F1 score.
9. Writes metrics to `results/vanet_model_metrics.json`.
10. Writes charts to `results/vanet_comparison.png` and `results/vanet_feature_importance.png`.
11. Saves generated model files into `models/` using names such as `vanet_Random_Forest.pkl`.

The current saved metrics show Random Forest as the best VANET model:

- Accuracy: 95.60%
- F1 score: 95.71%
- Precision: 96.25%
- Recall: 95.60%

The strongest features in the saved feature-importance output are trust-related:

- `trust_score`
- `neighbor_trust_score_avg`
- `historical_trust_score`

### Older VeReMi-Style Workflow

`scripts/preprocess.py` and `scripts/train_model.py` form an older or simpler pipeline.

`preprocess.py`:

1. Loads `data/vanet_malicious_nodes.csv`.
2. Uses 8 features: speed, acceleration, position, direction, packet drop ratio, latency, and signal strength.
3. Saves `data/X_train.npy`, `data/X_test.npy`, `data/y_train.npy`, and `data/y_test.npy`.
4. Saves `models/scaler.pkl`.

`train_model.py`:

1. Loads the `.npy` files created by `preprocess.py`.
2. Trains the same general set of nine classifiers.
3. Writes `results/model_metrics.json`.
4. Saves `.pkl` models into `models/`.

The current saved `results/model_metrics.json` shows Decision Tree as the best model for this older workflow, with a weighted F1 score of 65.39%.

### Chart Generation

`scripts/visualize_results.py` reads `results/model_metrics.json` and generates comparison charts for the older workflow.

The richer VANET workflow already generates its own charts directly inside `analyze_vanet_nodes.py`.

## Trust Logic

`scripts/trust_score.py` currently defines `HybridTrustEngine`, which maintains vehicle trust over time using an exponential moving average:

```text
new_trust = alpha * previous_trust + (1 - alpha) * evidence_score
```

It classifies trust scores as:

- `Trusted` for scores >= 0.70
- `Suspicious` for scores >= 0.40
- `Blocked` below 0.40

`scripts/combined_decision.py` demonstrates how a model prediction can feed into this trust engine. It loads a trained model and scaler, predicts the probability that a packet is honest, updates the vehicle's historical trust, and returns a final verdict.

## Dashboard Backend

`dashboard/app.py` is a Flask app running on port `5000`.

It serves:

- `GET /`: dashboard HTML.
- `POST /api/evaluate`: evaluates a manually entered vehicle.
- `POST /api/simulate`: creates a random normal or malicious-looking vehicle and evaluates it.
- `GET /api/log`: returns the recent in-memory activity log.
- `GET /api/stats`: returns accepted/warned/blocked counters.
- `GET /api/models`: returns older VeReMi-style model metrics from `results/model_metrics.json`.
- `GET /api/vanet_models`: returns rich VANET model metrics from `results/vanet_model_metrics.json`.

The backend keeps activity state in memory, so logs and counters reset when the Flask process restarts.

## Dashboard Frontend

The frontend is plain HTML, CSS, and JavaScript:

- `dashboard/templates/index.html` defines the dashboard layout.
- `dashboard/static/app.js` handles API calls, charts, logs, navigation, simulation, and rendering.
- `dashboard/static/style.css` controls the visual design.

The UI has three main sections:

1. `Evaluator`: manually submit/simulate a vehicle and see `ACCEPT`, `WARN`, or `BLOCK`.
2. `VeReMi Arena`: compare the older workflow's model metrics.
3. `VANET Nodes`: compare the richer VANET dataset models, feature importance, attack types, and best model.

Chart.js is loaded from a CDN in `index.html`, so the dashboard needs internet access for charts unless Chart.js is vendored locally.

## Generated And Ignored Artifacts

Tracked/generated data and results currently include:

- `data/*.npy`
- `results/*.json`
- `results/*.png`

Generated but ignored local artifacts include:

- `models/*.pkl`
- `models/scaler.pkl`
- `models/vanet_scaler.pkl`

Because `.pkl` files are ignored, a fresh clone needs the training scripts to be run before model-backed evaluation can work.

## How To Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the richer VANET models:

```bash
python scripts/analyze_vanet_nodes.py
```

Optionally run the older preprocessing/training flow:

```bash
python scripts/preprocess.py
python scripts/train_model.py
python scripts/visualize_results.py
```

Start the dashboard:

```bash
python dashboard/app.py
```

Open:

```text
http://127.0.0.1:5000
```

## Current Implementation Gaps

These are important when trying to run the app from the current code as committed:

1. `dashboard/app.py` imports `calculate_trust_score` and `classify_vehicle` from `scripts/trust_score.py`, but that file currently defines `HybridTrustEngine` and does not define those two functions. As written, the dashboard import path is inconsistent.

2. The dashboard's real-time evaluator builds a 7-feature vector, while `preprocess.py` trains on 8 features and `analyze_vanet_nodes.py` trains on 14 features. A trained model will expect the same feature count and scaling used during training.

3. `dashboard/app.py` loads older model filenames such as `Random_Forest.pkl`, but `train_model.py` saves the grid-search model as `Random_Forest_GridSearch.pkl`, and `analyze_vanet_nodes.py` saves rich VANET models with a `vanet_` prefix.

4. The richer VANET model metrics are displayed in the dashboard, but the evaluator does not currently load and scale the richer `vanet_*.pkl` models for live prediction.

5. Chart.js is external, so dashboard charts depend on CDN availability.

## Practical Mental Model

Think of this repo as a panel/demo ML system:

- `scripts/analyze_vanet_nodes.py` proves the ML results and generates the strongest metrics/charts.
- `results/vanet_model_metrics.json` is what powers the dashboard's strongest model-comparison story.
- `dashboard/` is the presentation layer for showing the evaluation concept.
- `scripts/trust_score.py` and `combined_decision.py` express the intended hybrid trust architecture.
- The live evaluator needs alignment work before it is fully consistent with the trained model artifacts.
