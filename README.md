# EdgeTrust-VANET 🛡️🚗
**Machine Learning-Driven Misbehavior Detection for Vehicular Ad-Hoc Networks (VANETs)**

[![Made with Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Built with Flask](https://img.shields.io/badge/Flask-Web_Framework-green.svg)](https://flask.palletsprojects.com/)
[![Powered by Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine_Learning-orange.svg)](https://scikit-learn.org/)

EdgeTrust-VANET is a sophisticated, edge-deployed cybersecurity system designed to protect Roadside Units (RSUs) from malicious vehicles transmitting falsified telemetry, performing Blackhole attacks, Sybil spoofing, or Network Denial of Service (DoS) in V2X communication environments.

---

## 📖 The Problem
Vehicular Ad-Hoc Networks allow autonomous and connected vehicles to share critical traffic data (speed, location) to prevent collisions. However, traditional security relies strictly on verifying the *identity* of the sender (PKI/Cryptography), not the *truthfulness* or *intent* of the data being sent. 

If a legitimate node is compromised, it can inject false traffic data to cause accidents or drop critical warning packets.

## 🚀 Our Solution: Two-Tiered Architecture
EdgeTrust-VANET solves this by operating directly at the network edge (RSU) using a highly efficient two-stage pipeline:

1. **Tier 1 - Trust Management Module (Heuristics)**
   - Computes physical plausibility checks (e.g., "Is acceleration physically possible?").
   - Analyzes message frequency, behavior history, and neighbor validation.
   - Outputs a continuous `trust_score` (0.0 to 1.0).
   
2. **Tier 2 - Machine Learning Evaluation Engine (AI)**
   - Ingests physical mobility features (GPS, Speed, Heading), network metrics (Packet Drops, Latency, RTT), alongside the RSU-computed Trust Score.
   - Passes the `1x14` vector into a pre-loaded ensemble Machine Learning model to output an immediate `ACCEPT`, `WARN`, or `BLOCK` decision in milliseconds.

---

## 📊 Dataset & Feature Selection

We trained and evaluated the system against the realistic **`vanet_malicious_nodes.csv`** dataset containing 5,000 uniquely behaving vehicles (25.2% Attack Rate). The raw dataset contains 18 features in total, but our final model uses only the **14 proposed features**.

### 🔍 Data Leakage Prevention

The raw dataset contains **4 retrospective post-hoc attack counters**: `false_packet_injection`, `blackhole_attack_attempts`, `sybil_attack_attempts`, and `denial_of_service`.
*   **Are they used in our active model?** **No.** They are programmatically dropped from our feature training list (`FEATURES`). The active system runs entirely on the **Proposed 14-feature set** (Mobility + Network + Trust).
*   **Why are they dropped?** These counters represent retrospective logs recorded *after* an attack is already detected and confirmed. An active Roadside Unit (RSU) making real-time decision calculations does not have access to these summaries.
*   **Why do we benchmark them?** In our feature ablation study ([`evidence/feature_selection/`](file:///Users/vivekchitturi/Desktop/edgetrust-ml/evidence/feature_selection/README.md)), combining these 4 leakage features with our proposed features (`Proposed + Leakage` - 18 features total) yields an artificial, perfect **100.0% F1 score**. We benchmark this configuration solely as a control group to demonstrate the data leakage trap and warn against circular reasoning in V2X machine learning research.


### Best Model: Random Forest 🏆
Out of 9 state-of-the-art algorithms evaluated (SVM, KNN, Extra Trees, AdaBoost, etc.), our Random Forest ensemble achieved the best balance for imbalanced security datasets:

*   **Accuracy:** 95.60%
*   **F1 Score:** 95.71%
*   **Precision:** 96.25% (Extremely low false-positive rate for legitimate drivers)

*Feature Importance:* Our RSU-computed generic `trust_score` proved to be the most critical feature (37.1% importance), validating the hybrid Heuristics + AI architectural approach.

---

## 🛠️ How to Run the Dashboard Locally

This repository includes a lightweight, glassmorphic web dashboard (powered by Flask and Vanilla JS) that simulates the intelligence of a Roadside Unit.

### 1. Requirements
*   Python 3.8+
*   `pip install -r requirements.txt` *(pandas, scikit-learn, flask, flask-cors, matplotlib, seaborn)*

### 2. Retrain the Models (Optional)
To retrain the AI models dynamically and repopulate the `results/` folder:
```bash
python scripts/analyze_vanet_nodes.py
```

### 3. Start the Web Dashboard
Boot up the Flask API and frontend server:
```bash
python dashboard/app.py
```

### 4. Open Application
Navigate to [http://127.0.0.1:5000](http://127.0.0.1:5000) in your web browser. 
*(Tip: If you see cached test data, perform a hard refresh with `Ctrl + Shift + R`)*

---

## 📂 Repository Structure

```text
EdgeTrust-VANET/
├── data/
│   └── vanet_malicious_nodes.csv      # V2X telemetry & attack dataset
├── dashboard/
│   ├── app.py                         # Flask Backend API Router
│   ├── static/                        # Frontend UI (JS logic, CSS Glassmorphism)
│   └── templates/                     # HTML Views
├── models/                            # Serialized weights (*.pkl)
├── results/                           # Evaluation metrics & Confusion Matrices (JSON/PNG)
└── scripts/
    ├── analyze_vanet_nodes.py         # Primary ML training loop & Data Prep
    ├── trust_score.py                 # Mathematical trust calculation engine
    └── preprocess.py                  # Initial data cleaning utils
```

---

## 🎓 Research Justification Evidence

This codebase includes reproducibility evidence packs to justify key design decisions and model selections for academic publication:

*   **Model Selection Justification ([`evidence/randomforest/`](file:///Users/vivekchitturi/Desktop/edgetrust-ml/evidence/randomforest/README.md))**: Compares Random Forest against all other 8 classifiers across holdout metrics, 20-fold repeated split stability, and edge computational efficiency (predict latency, size).
*   **Feature Selection Ablation Study ([`evidence/feature_selection/`](file:///Users/vivekchitturi/Desktop/edgetrust-ml/evidence/feature_selection/README.md))**: Ablation study across mobility, network, trust, and leakage features, justifying the 14-feature proposed set and proving the data leakage trap.
*   **Weighted F1 vs Accuracy ([`evidence/weighted_f1/`](file:///Users/vivekchitturi/Desktop/edgetrust-ml/evidence/weighted_f1/README.md))**: Justification for F1 ranking over accuracy under class imbalance.
*   **Trust Threshold Selection ([`evidence/trust_thresholds/`](file:///Users/vivekchitturi/Desktop/edgetrust-ml/evidence/trust_thresholds/README.md))**: Justification for the 0.70/0.40 warning/blocking boundaries.
*   **Trust Alpha Selection ([`evidence/trust_alpha/`](file:///Users/vivekchitturi/Desktop/edgetrust-ml/evidence/trust_alpha/README.md))**: Sensitivity of the EMA update \(T_t = \alpha T_{t-1} + (1-\alpha) E_t\) showing why α = 0.7.


