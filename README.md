# EdgeTrust-VANET Machine Learning Pipeline

**14-Dimensional Leakage-Free Vehicular Misbehavior Detection and 17-Model Benchmark Suite**

[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3%2B-orange.svg)](https://scikit-learn.org/)
[![GBDT Frameworks](https://img.shields.io/badge/GBDT-XGBoost%20%7C%20LightGBM%20%7C%20CatBoost-purple.svg)](https://xgboost.readthedocs.io/)
[![Primary Dataset](https://img.shields.io/badge/Dataset-VeReMi%20(72.9%25)-red.svg)](https://veremi-dataset.github.io/veremi)
[![Kaggle Dataset](https://img.shields.io/badge/Kaggle-EdgeTrust--VANET-blue.svg)](https://www.kaggle.com/datasets/mouniksai/vehicular-misbehavior-dataset-edgetrust-vanet/data)
[![License](https://img.shields.io/badge/License-Academic%20Research-lightgrey.svg)]()

---

## Technical Overview

`edgetrust-ml` contains the machine learning feature engineering, statistical leakage verification, model training, and embedded C++ inference export pipelines for the **EdgeTrust-VANET** framework.

In connected vehicular networks operating over IEEE 802.11p and Cellular-V2X (C-V2X), vehicles broadcast Basic Safety Messages (BSMs) at 10 Hz. While cryptographic Public Key Infrastructure (PKI) authenticates sender identity, it cannot verify the truthfulness of the kinematic telemetry inside the payload. An authenticated insider with compromised sensors or hijacked firmware can broadcast physically fabricated telemetry, leading to hazardous automated braking or phantom traffic jams.

This repository provides:
1. A unified multi-source research dataset of **23,970 records** integrating VeReMi simulation traces (72.88%), EdgeTrust physical telemetry (20.86%), and V-RADD routing logs (6.26%).
2. A strictly **14-dimensional, leakage-free feature space** containing exclusively quantities observable by a Roadside Unit (RSU).
3. Mathematical proof of target leakage in legacy benchmarks that rely on retrospective post-hoc attack counters.
4. A comprehensive benchmark of **17 machine learning algorithms across 5 distinct algorithmic families**, evaluated on 3,539 completely unseen vehicular trajectories via Stratified GroupKFold validation.
5. Automated C++ code generators exporting trained classifiers into zero-dependency C++ header files for sub-microsecond edge execution.

---

## Documentation Index

For in-depth explanations and supplementary technical guides:

- **[DASHBOARD_GUIDE.md](./DASHBOARD_GUIDE.md):** Architectural walkthrough of the real-time Flask edge simulator, telemetry injectors, and decision threshold controls.
- **[DATASET_AND_PIPELINE_GUIDE.md](./DATASET_AND_PIPELINE_GUIDE.md):** Mathematical formulations, VeReMi raw log extraction routines, feature engineering pipelines, and cryptographic leakage audits.
- **[PAPER_JUSTIFICATION_POINTS.md](./PAPER_JUSTIFICATION_POINTS.md):** Defense arguments, hyperparameter justifications (why $\alpha=0.70$, why Random Forest vs. LightGBM, why weighted F1).
- **[CODEBASE_OVERVIEW.md](./CODEBASE_OVERVIEW.md):** Software architecture, module interactions, and file dependencies.
- **[Kaggle Dataset Hub](https://www.kaggle.com/datasets/mouniksai/vehicular-misbehavior-dataset-edgetrust-vanet/data):** Public repository for the unified dataset and raw simulation traces.

---

## The 14 Leakage-Free Observable Features

To guarantee that classifiers remain functional when deployed on physical Roadside Units (RSUs), the feature representation $\mathbf{x} \in \mathbb{R}^{14}$ is strictly restricted to spatiotemporal, network, and reputation metrics observable in real time:

| Category | Feature Identifier | Physical / Metric Dimension | Measurement Source |
| :--- | :--- | :--- | :--- |
| **Mobility Kinematics (5)** | `position_x` | Longitudinal coordinate ($x$) | RSU Local Cartesian Frame (m) |
| | `position_y` | Lateral coordinate ($y$) | RSU Local Cartesian Frame (m) |
| | `speed` | Vehicle instantaneous velocity ($v$) | BSM Payload / Doppler Radar (m/s) |
| | `direction` | Heading angle ($\theta$) | Kinematic Orientation ($^\circ$) |
| | `acceleration` | Longitudinal acceleration ($a$) | IMU Sensor / Velocity Delta ($\text{m/s}^2$) |
| **Radio & Network (6)** | `packet_sent` | Cumulative transmission counter | MAC Layer Sequence Tracking |
| | `packet_received` | Packets successfully acknowledged | RSU Interface Receiver Queue |
| | `packet_drop_ratio`| Packet loss ratio ($\text{PDR}$) | Channel Error / Drop Rate $[0, 1]$ |
| | `latency` | End-to-end communication delay | Timestamp Delta ($t_{\text{rx}} - t_{\text{tx}}$) in ms |
| | `retransmission_count` | 802.11p MAC retransmissions | Frame Retry Counter |
| | `signal_strength` | Received Signal Strength Indication | Physical PHY Layer RSSI (dBm) |
| **EdgeTrust Reputation (3)** | `trust_score` | Instantaneous EMA trust score | Tier 2 Exponential Moving Average |
| | `neighbor_trust_score_avg`| Multi-angle consensus reputation | 300 m Spatial Neighbor Average |
| | `historical_trust_score` | Long-term accumulated reputation | Regional Fog Ledger Rolling Mean |

### Mathematical Elimination of Post-Hoc Data Leakage

Legacy vehicular intrusion datasets frequently include retrospective post-hoc attack counters:
- `false_packet_injection`
- `blackhole_attack_attempts`
- `sybil_attack_attempts`
- `denial_of_service`

**Why these counters constitute invalid data leakage:**
1. **The Semantic Trap:** These fields record the ground-truth number of cumulative malicious actions an attacker has already executed. If an ML model is trained with these features, the tree split places 99% of its decision weight on `if attack_attempts > 0 then Malicious`, producing an artificial **100.00% F1-score** on static train-test splits.
2. **Failure in Live Deployment:** When an unknown vehicle enters an RSU's 300 m coverage zone, its post-hoc attack history is strictly $0$. A model trained on retrospective counters immediately collapses to **86.05% F1-score** when tested on unseen trajectories.
3. **Audit Verification:** By mathematically stripping all post-hoc counters (`scripts/leakage_audit.py`), our model trains exclusively on observable physical and communicative telemetry, retaining an authentic **97.74% F1-score** on completely unseen vehicle IDs.

---

## Unified Multi-Source Research Dataset

The dataset (`data/unified_edgetrust_dataset.csv`) contains **23,970 records** across 26 attributes, adhering to defensible scientific methodology where VeReMi represents the primary benchmark component:

| Source | Role | Records | Proportion | Threat Coverage |
| :--- | :--- | :---: | :---: | :--- |
| **VeReMi** | Primary Benchmark Source | **17,470** | **72.88%** | Types 1, 2, 4, 8, 16 across low, medium, and high traffic densities |
| **EdgeTrust Telemetry** | Secondary Telemetry Source | **5,000** | **20.86%** | Normal, Blackhole, Sybil, DoS, False Packet Injection |
| **V-RADD** | Routing Augmentation | **1,500** | **6.26%** | Grayhole (Selective Forwarding), Data Replay |
| **Total** | Unified Research Dataset | **23,970** | **100.0%** | **7 Standardized Taxonomy Classes** |

- **Class Balance:** 16,917 Benign instances (70.58%) vs. 7,053 Malicious instances (29.42%).
- **Partitioning Protocol:** 5-Fold Stratified GroupKFold grouped strictly by `vehicle_id` (16,885 train / 3,546 validation / 3,539 holdout test). Zero sample or vehicle trajectory overlap across splits.

---

## 17-Model Benchmark Suite (Unseen Vehicle Evaluation)

Evaluated on **3,539 holdout test records from completely unseen vehicle trajectories** after 5-Fold Stratified Group Cross-Validation ($96.95\% \pm 1.12\%$ on training folds):

| Algorithm Name | Model Family | Accuracy | Weighted F1 | Precision | Recall | ROC-AUC | Inference (us) | Footprint |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | Bagging Ensemble | **97.77%** | **97.74%** | **97.74%** | **97.77%** | **99.82%** | 13,441 us | 6.4 MB |
| **LightGBM** | Gradient Boosting | **97.74%** | **97.71%** | **97.71%** | **97.74%** | **99.78%** | **329 us** | **669 KB** |
| **Stacking Ensemble** | Hybrid Ensemble | 97.51% | 97.48% | 97.48% | 97.51% | 99.79% | 14,522 us | 2.2 MB |
| **HistGradientBoosting** | Histogram GBDT | 97.40% | 97.37% | 97.37% | 97.40% | 99.79% | 7,374 us | 653 KB |
| **CatBoost** | Oblivious Trees | 97.29% | 97.24% | 97.24% | 97.29% | 99.79% | **89 us** | **248 KB** |
| **XGBoost** | Exact GBDT | 97.29% | 97.24% | 97.23% | 97.29% | 99.77% | 177 us | 464 KB |
| **Voting Ensemble** | Soft Voting | 97.20% | 97.15% | 97.15% | 97.20% | 99.80% | 14,174 us | 3.9 MB |
| **Gradient Boosting** | Sequential Trees | 97.17% | 97.13% | 97.11% | 97.17% | 99.71% | 129 us | 535 KB |
| **Decision Tree (CART)**| Single Tree | 96.84% | 96.79% | 96.78% | 96.84% | 96.53% | **49 us** | **18 KB** |
| **MLP Neural Network** | Deep Learning | 96.13% | 96.11% | 96.11% | 96.13% | 99.16% | **76 us** | 392 KB |
| **Extra Trees** | Randomized Forests| 92.43% | 91.96% | 92.20% | 92.43% | 98.71% | 13,469 us | 11.3 MB |
| **K-Nearest Neighbors** | Instance-Based | 90.96% | 90.65% | 90.49% | 90.96% | 95.84% | 15,329 us | 5.0 MB |
| **SVM (RBF Kernel)** | Kernel Classifier | 90.03% | 89.19% | 90.52% | 90.03% | 94.75% | 242 us | 742 KB |
| **AdaBoost** | Adaptive Boosting | 88.25% | 87.13% | 88.08% | 88.25% | 94.67% | 3,220 us | 38 KB |
| **Logistic Regression** | Generalized Linear| 88.08% | 86.78% | 88.35% | 88.08% | 91.80% | **65 us** | **1 KB** |
| **Linear Discriminant** | Generative Linear | 87.93% | 86.58% | 88.29% | 87.93% | 91.90% | **68 us** | **1 KB** |
| **Gaussian Naive Bayes**| Probabilistic | 82.17% | 81.12% | 82.02% | 82.17% | 86.83% | **65 us** | **1 KB** |

---

## Embedded C++ Export Architecture

To achieve deterministic sub-microsecond latency on edge Roadside Units, trained Python models are automatically parsed and translated into standalone C++ header files compiled directly into `edgetrust-core`:

```
+---------------------------+       scripts/train_unified_pipeline.py
| scikit-learn / LightGBM   | ------------------------------------------+
| Trained Python Model .pkl |                                           |
+---------------------------+                                           v
                                                       +----------------------------------+
                                                       | Code Generator Parser            |
                                                       | - Tree split traversal           |
                                                       | - Fixed-point weight unrolling   |
                                                       | - Zero external dependencies     |
                                                       +----------------------------------+
                                                                        |
                                                                        v
+-----------------------------------------------------------------------------------------+
| Target: edgetrust-core/simulator/omnetpp/EdgeTrustSim/                                  |
|   - LogisticRegressionPredictor.h  -->  Dot product vectorization (0.38 us / msg)       |
|   - RandomForestPredictor.h        -->  Nested branch decision forest (13.4 ms / msg)   |
|   - LightGBMPredictor.h            -->  Compact leaf index lookups (329 us / msg)       |
+-----------------------------------------------------------------------------------------+
```

---

## Pipeline Execution Guide

### 1. Installation
Clone the repository and install required Python packages:

```bash
git clone https://github.com/edge-trust-vanet/edgetrust-ml.git
cd edgetrust-ml
pip install -r requirements.txt
```

### 2. Extracting Raw VeReMi Traces
Extract and parse JSON-based simulation logs from VeReMi simulation archives:

```bash
python scripts/extract_veremi.py
```

### 3. Dataset Unification and Trust Computation
Fuse VeReMi logs with physical telemetry and compute EMA trust metrics:

```bash
python scripts/build_unified_dataset.py
```

### 4. Running the Leakage Audit
Execute the mathematical zero-leakage verification audit:

```bash
python scripts/leakage_audit.py
```

### 5. Training the 17-Model Benchmark Suite
Train, validate, and evaluate all 17 models using Stratified GroupKFold:

```bash
python scripts/train_unified_pipeline.py
```

### 6. Launching the Flask Edge Simulation Dashboard
Run the interactive RSU edge simulator locally:

```bash
python dashboard/app.py
```
Access the dashboard at `http://127.0.0.1:5000`.

---

## Repository Structure

```
edgetrust-ml/
├── README.md                            # Comprehensive machine learning documentation
├── DASHBOARD_GUIDE.md                   # Real-time Flask dashboard walkthrough
├── DATASET_AND_PIPELINE_GUIDE.md        # Mathematical derivations and data pipeline guide
├── PAPER_JUSTIFICATION_POINTS.md        # Academic defense and parameter justifications
├── CODEBASE_OVERVIEW.md                 # Technical code architecture overview
├── demo.txt                             # Quick panel demonstration script
├── requirements.txt                     # Python dependencies
├── data/                                # Unified datasets and partition splits
│   ├── unified_edgetrust_dataset.csv    # 23,970 records (72.88% VeReMi)
│   ├── veremi_extracted.csv             # Extracted VeReMi telemetry
│   ├── vanet_malicious_nodes.csv        # Physical edge telemetry
│   ├── unified_X_train.npy / y_train    # 16,885 training records (70%)
│   ├── unified_X_val.npy / y_val        # 3,546 validation records (15%)
│   └── unified_X_test.npy / y_test      # 3,539 holdout test records (15%)
├── scripts/                             # Pipeline automation scripts
│   ├── extract_veremi.py                # Raw archive extractor
│   ├── build_unified_dataset.py         # Multi-source unification script
│   ├── leakage_audit.py                 # Mathematical correlation & leakage verifier
│   ├── train_unified_pipeline.py        # 17-model trainer & GroupKFold benchmark
│   └── trust_score.py                   # Exponential Moving Average trust calculator
├── dashboard/                           # Flask edge simulation runtime
│   ├── app.py                           # Backend REST API and live model inference
│   ├── templates/index.html             # Dashboard frontend template
│   └── static/                          # Frontend CSS and JavaScript
├── models/                              # Serialized model artifacts (.pkl)
│   ├── vanet_scaler.pkl                 # StandardScaler fitted on train split
│   ├── vanet_Random_Forest.pkl          # Champion model artifact
│   └── vanet_*.pkl                      # Remaining 16 trained model files
└── results/                             # Benchmark metrics and publication charts
    ├── unified_model_metrics.json       # Exact numerical metrics for all 17 models
    ├── leakage_audit_report.json        # Correlation matrices and leakage audit report
    ├── unified_model_comparison.png     # Publication performance bar chart
    └── unified_feature_importance.png   # 14-feature Gini importance diagram
```

---

## Academic Citation

```bibtex
@inproceedings{edgetrust_vanet_2026,
  author    = {Neethu, M. R. and Mounik Sai, M. and Aashiq, E. and Vivek Vardhan, C. and Nishanth, K. and Abhinav, J.},
  title     = {EdgeTrust-VANET: A Decentralized Four-Tier Hierarchical Trust Management and Kinematic Filtering Framework for Vehicular Ad-Hoc Networks},
  booktitle = {Proceedings of the IEEE International Conference on Communications (ICC) / VTC},
  year      = {2026},
  pages     = {1--8},
  address   = {Coimbatore, India}
}
```
