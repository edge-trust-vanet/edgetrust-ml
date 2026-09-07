# EdgeTrust-VANET Codebase Overview 📂

This repository contains the machine learning, data engineering, and web demonstration pipeline for **EdgeTrust-VANET: Hierarchical Trust-Adaptive Secure Communication Framework**. The system simulates a Roadside Unit (RSU) that combines dynamic trust tracking with 17 machine-learning classifiers to evaluate Basic Safety Messages (BSMs) and decide whether a vehicle should be **ACCEPTED**, **WARNED**, or **BLOCKED**.

---

## 🏗️ Repository Layout

```text
.
├── README.md                            # Master repository guide
├── DASHBOARD_GUIDE.md                   # Complete user and visual guide to the web UI
├── DATASET_AND_PIPELINE_GUIDE.md        # Mathematical, data engineering & leakage audit guide
├── PAPER_JUSTIFICATION_POINTS.md        # Academic justification points & evidence
├── CODEBASE_OVERVIEW.md                 # Technical code architecture (this file)
├── demo.txt                             # 5-step quick demonstration script
├── requirements.txt                     # Python packages (pandas, scikit-learn, xgboost, etc.)
├── data/
│   ├── unified_edgetrust_dataset.csv    # 23,970 unified records (72.9% VeReMi)
│   ├── veremi_extracted.csv             # 17,470 records extracted from VeReMi logs
│   ├── vanet_malicious_nodes.csv        # 5,000 legacy records
│   ├── raw_veremi/                      # 9 official VeReMi simulation .tgz archives
│   ├── unified_X_train.npy / y_train    # 16,885 training samples (70%)
│   ├── unified_X_val.npy / y_val        # 3,546 validation samples (15%)
│   └── unified_X_test.npy / y_test      # 3,539 holdout test samples (15%)
├── scripts/
│   ├── extract_veremi.py                # Extracts 9 VeReMi simulation archives into CSV
│   ├── build_unified_dataset.py         # Unifies VeReMi, EdgeTrust, and V-RADD
│   ├── leakage_audit.py                 # Proves zero leakage (counters & spatiotemporal splitting)
│   ├── train_unified_pipeline.py        # 17-model trainer & Group CV benchmark
│   ├── trust_score.py                   # EdgeTrust EMA hybrid trust engine
│   ├── combined_decision.py             # Hybrid ML + Trust simulation script
│   └── analyze_vanet_nodes.py           # Legacy 5,000 nodes analysis script
├── dashboard/
│   ├── app.py                           # Flask backend with 14-feature real-time scaling & prediction
│   ├── templates/index.html             # HTML layout (Evaluator, Arena, VANET Nodes)
│   └── static/
│       ├── app.js                       # Frontend logic, Chart.js charts, and live simulation
│       └── style.css                    # Dark cybersecurity theme styling
├── models/
│   ├── vanet_scaler.pkl                 # StandardScaler fitted on train split only
│   ├── vanet_Random_Forest.pkl          # Champion model (97.74% F1, 97.77% Acc)
│   └── vanet_*.pkl                      # All 17 trained model artifacts
└── results/
    ├── unified_model_metrics.json       # Benchmark metrics for all 17 models
    ├── leakage_audit_report.json        # Correlation & leakage audit proof
    ├── unified_model_comparison.png     # Publication comparison bar chart
    ├── unified_feature_importance.png   # 14-feature importance visualization
    └── leakage_audit.png                # Leakage experiment comparison chart
```

---

## 🔄 Core ML Workflows

### 1. Data Engineering & Extraction Workflow
- **`scripts/extract_veremi.py`**:
  - Ingests raw `.tgz` archives of official VeReMi simulations (Types 1, 2, 4, 8, 16 across low, medium, and high densities).
  - Matches BSM message IDs with ground-truth logs.
  - Derives kinematic vectors (`speed`, `direction`, `acceleration`), transmission reception rates, `packet_drop_ratio`, `latency`, and `signal_strength` in dBm.
  - Dynamically calculates EdgeTrust trust scores using the $\alpha = 0.70$ Exponential Moving Average equation.
  - Computes spatial neighborhood trust averages via KD-tree within $300\text{ m}$.
  - Exports `data/veremi_extracted.csv`.

- **`scripts/build_unified_dataset.py`**:
  - Combines VeReMi (17,470 records, **72.88%**), EdgeTrust-VANET (5,000 records, **20.86%**), and V-RADD routing augmentations (1,500 records, **6.26%**).
  - Unifies all records under the common EdgeTrust trust engine.
  - Standardizes attack taxonomy: `normal`, `false_data_injection`, `grayhole`, `replay`, `blackhole`, `dos`, `sybil`.
  - Re-indexes metadata: `source_dataset`, `scenario_id`, `simulation_id`, `node_id`, `timestamp`.
  - Exports `data/unified_edgetrust_dataset.csv`.

### 2. Leakage Audit Workflow
- **`scripts/leakage_audit.py`**:
  - Computes Pearson correlation, Spearman rank correlation, and Mutual Information across all 14 observable features and the 4 post-hoc attack counters.
  - Demonstrates that cheating on post-hoc attack counters yields an artificial 100% score on static tests, but collapses to 86.05% F1 in live deployment when counters are initially zero.
  - Quantifies the 1.39% generalization gap between naive random row splitting and scientifically valid group-based vehicle splitting (`node_id`).

### 3. Model Training & Cross-Validation Workflow
- **`scripts/train_unified_pipeline.py`**:
  - Splits data into 70% Train, 15% Validation, and 15% Holdout Test grouped strictly by `node_id` (zero vehicle overlap).
  - Fits `StandardScaler` strictly on the training partition.
  - Performs 5-Fold Stratified Group Cross-Validation on the training set (**96.95% ± 1.12% Weighted F1**).
  - Trains all 17 classification models across 5 distinct paradigm families.
  - Evaluates models on 3,539 completely unseen vehicle trajectories.
  - Measures single-query RSU inference latency in microseconds and model storage footprint in KB.
  - Exports trained `.pkl` artifacts to `models/`, metrics to `results/`, and publication-quality plots.

### 4. Interactive RSU Dashboard
- **`dashboard/app.py`**:
  - Flask web server running on port 5000.
  - Loads `models/vanet_scaler.pkl` and `models/vanet_Random_Forest.pkl`.
  - Reconstructs 14 observable features from vehicle inputs and trust sliders.
  - Transforms features via the scaler and runs genuine model inference.
  - Implements the hybrid decision matrix:
    - Trust $\ge 0.70$ and ML Normal $\rightarrow$ **ACCEPT**
    - Trust $< 0.40$ or ML Malicious $\rightarrow$ **WARN**
    - Trust $< 0.40$ and ML Malicious $\rightarrow$ **BLOCK**
  - Serves live simulation, activity logging, and model benchmark statistics.
