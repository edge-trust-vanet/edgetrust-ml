# EdgeTrust-VANET 🛡️🚗
**Hierarchical Trust-Adaptive Secure Communication Framework for Connected Vehicles**

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Web Framework](https://img.shields.io/badge/Flask-Web_Framework-green.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-orange.svg)](https://scikit-learn.org/)
[![GBDT Suite](https://img.shields.io/badge/GBDT-XGBoost%20%7C%20LightGBM%20%7C%20CatBoost-purple.svg)](https://xgboost.readthedocs.io/)
[![Primary Dataset](https://img.shields.io/badge/Dataset-VeReMi%20(72.9%25)-red.svg)](https://veremi-dataset.github.io/veremi)

EdgeTrust-VANET is a high-performance, edge-deployable cybersecurity and trust management system designed for **Roadside Units (RSUs) and connected autonomous vehicles**. It detects malicious network intrusions, spoofed GPS telemetry, blackhole packet drops, Sybil spoofing, and Denial of Service (DoS) attacks in **802.11p DSRC and C-V2X** vehicular environments.

---

## 📚 Documentation Index

To make this project easy to understand for everyone (whether you are an evaluator, researcher, or developer), the documentation is organized into clear guides:

* 🚗 **[DASHBOARD_GUIDE.md](file:///Users/mouniksai/Documents/edgetrust-ml/DASHBOARD_GUIDE.md)**: **Start here if you want to use or present the web dashboard!** Explains every button, slider, chart, and how the RSU makes `ACCEPT`, `WARN`, and `BLOCK` decisions.
* 🔬 **[DATASET_AND_PIPELINE_GUIDE.md](file:///Users/mouniksai/Documents/edgetrust-ml/DATASET_AND_PIPELINE_GUIDE.md)**: Mathematical formulations, data extraction from VeReMi, feature engineering, and leakage audit proofs.
* 📝 **[PAPER_JUSTIFICATION_POINTS.md](file:///Users/mouniksai/Documents/edgetrust-ml/PAPER_JUSTIFICATION_POINTS.md)**: Defense and justification points for research presentations and papers (why Random Forest, why $\alpha=0.7$, why weighted F1).
* 📁 **[CODEBASE_OVERVIEW.md](file:///Users/mouniksai/Documents/edgetrust-ml/CODEBASE_OVERVIEW.md)**: Technical directory structure, script interactions, and software design.
* 🎬 **[demo.txt](file:///Users/mouniksai/Documents/edgetrust-ml/demo.txt)**: Simple 5-step quick script for live demonstration.

---

## 📖 The Problem

Modern connected vehicles broadcast **Basic Safety Messages (BSMs)** 10 times per second to announce their position, speed, and heading. Traditional security relies on Public Key Infrastructure (PKI) to verify the sender's *identity* using digital certificates.

However, **PKI does not verify whether the data inside the message is truthful**. If an attacker compromises a legitimate vehicle's private key or an onboard sensor malfunctions, the vehicle can broadcast falsified GPS coordinates, forge non-existent emergency brake events to cause ghost jams, drop safety warnings, or flood the communication channel.

---

## 🚀 Two-Tiered Edge Architecture

EdgeTrust-VANET operates directly at the network edge (RSU) using a two-tier hybrid architecture:

```text
Vehicle Telemetry (GPS, Speed, Accel) + Radio Telemetry (Drop Ratio, Latency, RSSI)
                                       │
                                       ▼
    ┌────────────────────────────────────────────────────────────────────────┐
    │  TIER 1: Hybrid Trust Management Engine (Real-Time Heuristics)         │
    │  - Exponential Moving Average (EMA) trust update with α = 0.70         │
    │  - Kinematic plausibility (speed ≤ 45 m/s, acceleration bounds)        │
    │  - 300m spatial neighbor multi-angle consensus                         │
    │  - Generates: trust_score, neighbor_trust_avg, historical_trust        │
    └────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
    ┌────────────────────────────────────────────────────────────────────────┐
    │  TIER 2: Advanced Machine Learning Classifier Suite                    │
    │  - 14-dimensional leakage-free feature vector (RSU-observable)         │
    │  - Evaluated on 17 ML algorithms across 5 distinct paradigm families   │
    │  - Sub-millisecond inference latency (89 µs - 329 µs) for edge RSUs    │
    └────────────────────────────────────────────────────────────────────────┘
                                       │
                                       ▼
    ┌────────────────────────────────────────────────────────────────────────┐
    │  HYBRID DECISION MATRIX:                                               │
    │  - High Trust (≥0.70) AND ML Normal     -->  [ ✅ ACCEPT VEHICLE ]     │
    │  - Low Trust (<0.40)  OR  ML Malicious  -->  [ ⚠️ WARN VEHICLE ]       │
    │  - Low Trust (<0.40)  AND ML Malicious  -->  [ 🚫 BLOCK & QUARANTINE ] │
    └────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 The Unified Research Dataset

The dataset (`data/unified_edgetrust_dataset.csv`) contains **23,970 records** across **26 columns**. As required for defensible scientific research, **VeReMi forms the majority (72.88%)**:

| Source | Role | Records | Share | Attacks Covered |
| :--- | :--- | :---: | :---: | :--- |
| **[VeReMi](https://veremi-dataset.github.io/veremi)** | Primary Reference Source | **17,470** | **72.88%** | Types 1, 2, 4, 8, 16 across low-, medium-, and high-density LuST Luxembourg scenarios |
| **EdgeTrust-VANET** | Secondary Telemetry Source | **5,000** | **20.86%** | Normal, Blackhole, Sybil, Denial of Service (DoS), False Packet Injection |
| **V-RADD** | Routing Augmentation | **1,500** | **6.26%** | Grayhole (Selective Forwarding), Data Replay |
| **TOTAL** | Unified Research Dataset | **23,970** | **100.0%** | **7 Standardized Taxonomy Classes** |

- **Class Distribution**: 16,917 Benign (70.58%) vs. 7,053 Malicious (29.42%).

---

## 🛡️ The 14 Leakage-Free Observable Features

To prevent circular reasoning and ensure the model works in live deployment, the classifier uses strictly **14 features** observable by an RSU:

1. **Mobility Kinematics (5)**: `position_x`, `position_y`, `speed`, `direction`, `acceleration`
2. **Network & Radio Telemetry (6)**: `packet_sent`, `packet_received`, `packet_drop_ratio`, `latency`, `retransmission_count`, `signal_strength`
3. **EdgeTrust Reputation (3)**: `trust_score`, `neighbor_trust_score_avg`, `historical_trust_score`

### 🚫 Why Post-Hoc Attack Counters are Excluded
Legacy datasets contain 4 retrospective attack counters: `false_packet_injection`, `blackhole_attack_attempts`, `sybil_attack_attempts`, `denial_of_service`.
- **The Cheat**: Including them produces an artificial **100.00% F1 score** on static tests.
- **The Proof**: When an unknown vehicle first enters an RSU, its retrospective counters are 0. A model relying on these counters **collapses to 86.05% F1** in live deployment.
- **Our Honest Model**: Trained on the 14 observable features, it maintains a genuine **97.74% F1 score** on unseen vehicles!

---

## 🏆 17-Model Benchmark Suite (Unseen Vehicle Evaluation)

Evaluated on **3,539 completely unseen vehicle trajectories** after **5-Fold Stratified Group Cross-Validation** ($96.95\% \pm 1.12\%$ on Train):

| Model Name | Accuracy | Weighted F1 | Recall | Precision | ROC-AUC | Latency (µs) | Model Size |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Champion)** | **97.77%** | **97.74%** | **97.77%** | **97.74%** | **99.82%** | 13,441 µs | 6.4 MB |
| **LightGBM (Top Edge GBDT)** | **97.74%** | **97.71%** | **97.74%** | **97.71%** | **99.78%** | **329 µs** | **669 KB** |
| **Stacking Ensemble** | 97.51% | 97.48% | 97.51% | 97.48% | 99.79% | 14,522 µs | 2.2 MB |
| **Hist Gradient Boosting** | 97.40% | 97.37% | 97.40% | 97.37% | 99.79% | 7,374 µs | 653 KB |
| **CatBoost** | 97.29% | 97.24% | 97.29% | 97.24% | 99.79% | **89 µs** | **248 KB** |
| **XGBoost** | 97.29% | 97.24% | 97.29% | 97.23% | 99.77% | 177 µs | 464 KB |
| **Voting Ensemble** | 97.20% | 97.15% | 97.20% | 97.15% | 99.80% | 14,174 µs | 3.9 MB |
| **Gradient Boosting** | 97.17% | 97.13% | 97.17% | 97.11% | 99.71% | 129 µs | 535 KB |
| **Decision Tree (CART)** | 96.84% | 96.79% | 96.84% | 96.78% | 96.53% | **49 µs** | **18 KB** |
| **MLP Neural Network** | 96.13% | 96.11% | 96.13% | 96.11% | 99.16% | **76 µs** | 392 KB |
| **Extra Trees** | 92.43% | 91.96% | 92.43% | 92.20% | 98.71% | 13,469 µs | 11.3 MB |
| **K-Nearest Neighbors** | 90.96% | 90.65% | 90.96% | 90.49% | 95.84% | 15,329 µs | 5.0 MB |
| **SVM (RBF Kernel)** | 90.03% | 89.19% | 90.03% | 90.52% | 94.75% | 242 µs | 742 KB |
| **AdaBoost** | 88.25% | 87.13% | 88.25% | 88.08% | 94.67% | 3,220 µs | 38 KB |
| **Logistic Regression** | 88.08% | 86.78% | 88.08% | 88.35% | 91.80% | **65 µs** | **1 KB** |
| **Linear Discriminant (LDA)** | 87.93% | 86.58% | 87.93% | 88.29% | 91.90% | **68 µs** | **1 KB** |
| **Gaussian Naive Bayes** | 82.17% | 81.12% | 82.17% | 82.02% | 86.83% | **65 µs** | **1 KB** |

---

## ⚡ Quick Start: How to Run the Project

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the Full Data Pipeline (Optional, already pre-built)
```bash
# Extract raw VeReMi simulation logs
python scripts/extract_veremi.py

# Unify datasets and compute trust features
python scripts/build_unified_dataset.py

# Run leakage verification audit
python scripts/leakage_audit.py

# Train all 17 ML models and save benchmark charts
python scripts/train_unified_pipeline.py
```

### 3. Launch the Dashboard
```bash
python dashboard/app.py
```

Open your browser and navigate to:
```
http://127.0.0.1:5000
```
*(See **[DASHBOARD_GUIDE.md](file:///Users/mouniksai/Documents/edgetrust-ml/DASHBOARD_GUIDE.md)** for a full interactive walkthrough!)*

---

## 📂 Repository Structure

```text
.
├── README.md                            # Primary project documentation
├── DASHBOARD_GUIDE.md                   # Visual guide to using the web dashboard
├── DATASET_AND_PIPELINE_GUIDE.md        # Mathematical & data engineering guide
├── PAPER_JUSTIFICATION_POINTS.md        # Academic defense & justification points
├── CODEBASE_OVERVIEW.md                 # Technical code layout
├── demo.txt                             # 5-step quick panel demonstration script
├── requirements.txt                     # Dependencies
├── data/
│   ├── unified_edgetrust_dataset.csv    # 23,970 records (72.9% VeReMi)
│   ├── veremi_extracted.csv             # 17,470 records from VeReMi logs
│   ├── vanet_malicious_nodes.csv        # 5,000 legacy records
│   ├── raw_veremi/                      # Downloaded official VeReMi .tgz archives
│   ├── unified_X_train.npy / y_train    # 16,885 training samples (70%)
│   ├── unified_X_val.npy / y_val        # 3,546 validation samples (15%)
│   └── unified_X_test.npy / y_test      # 3,539 holdout test samples (15%)
├── scripts/
│   ├── extract_veremi.py                # Extracts 9 VeReMi simulation archives
│   ├── build_unified_dataset.py         # Unifies VeReMi, EdgeTrust, and V-RADD
│   ├── leakage_audit.py                 # Mathematical proof of zero leakage
│   ├── train_unified_pipeline.py        # 17-model trainer & Group CV benchmark
│   └── trust_score.py                   # EdgeTrust EMA hybrid trust engine
├── dashboard/
│   ├── app.py                           # Flask backend with real-time 14-feature scaling
│   ├── templates/index.html             # Dashboard UI layout
│   └── static/
│       ├── app.js                       # Chart.js, live simulator, and logs
│       └── style.css                    # Modern cybersecurity dark theme
├── models/
│   ├── vanet_scaler.pkl                 # StandardScaler fitted on train split
│   ├── vanet_Random_Forest.pkl          # Champion model (97.74% F1)
│   └── vanet_*.pkl                      # All 17 trained model artifacts
└── results/
    ├── unified_model_metrics.json       # Benchmark metrics for all 17 models
    ├── leakage_audit_report.json        # Correlation & leakage audit proof
    ├── unified_model_comparison.png     # Publication comparison bar chart
    ├── unified_feature_importance.png   # 14-feature importance visualization
    └── leakage_audit.png                # Leakage experiment comparison chart
```

---

## 📜 Citation & References
1. **VeReMi**: R. van der Heijden et al., *"VeReMi: A Dataset for Comparable Evaluation of Misbehavior Detection in VANETs"*, ACM WiSec.
2. **VeReMi Extension**: J. Kamel et al., *"VeReMi Extension: A Dataset for Comparable Evaluation of Misbehavior Detection in VANETs"*, IEEE ICC.
3. **LuST**: L. Codecá et al., *"Luxembourg SUMO Traffic (LuST) Scenario"*, IEEE VNC.
