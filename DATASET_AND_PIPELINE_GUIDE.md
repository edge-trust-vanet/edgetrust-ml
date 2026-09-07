# EdgeTrust-VANET: Dataset Engineering, Trust Engine & ML Pipeline Guide 🔬📊

This document provides a detailed, mathematically rigorous explanation of the **EdgeTrust-VANET** data engineering pipeline, trust computation engine, leakage audit protocol, and machine-learning benchmarking suite.

---

## 📌 1. Dataset Architecture & Sourcing

The unified research dataset (`data/unified_edgetrust_dataset.csv`) contains **23,970 records** across **26 columns**, scientifically unifying three complementary data sources:

```text
                                  +---------------------------------------+
                                  |    RAW VEHICULAR DATA SOURCES        |
                                  +---------------------------------------+
                                                      |
                    +---------------------------------+---------------------------------+
                    |                                 |                                 |
                    v                                 v                                 v
    +------------------------------+  +------------------------------+  +------------------------------+
    |  VeReMi Reference (Primary)   |  |  EdgeTrust-VANET (Secondary) |  |   V-RADD (Network Routing)   |
    |  - 17,470 records (72.88%)   |  |  - 5,000 records (20.86%)    |  |  - 1,500 records (6.26%)     |
    |  - Types 1, 2, 4, 8, 16      |  |  - Blackhole, Sybil, DoS, FDI|  |  - Grayhole & Replay Attacks|
    |  - LuST Luxembourg Scenario  |  |  - Radio Signal & Retrans.   |  |  - Forwarding Anomalies      |
    +------------------------------+  +------------------------------+  +------------------------------+
                    |                                 |                                 |
                    +---------------------------------+---------------------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |   COMMON EDGETRUST TRUST ENGINE (α)   |
                                  |   - Real-time Evidence E_t in [0, 1]  |
                                  |   - EMA Trust Update: T_t = 0.7*T_t-1 |
                                  |   - 300m KD-Tree Spatial Neighbor Avg |
                                  +---------------------------------------+
                                                      |
                                                      v
                                  +---------------------------------------+
                                  |       UNIFIED RESEARCH DATASET        |
                                  |       23,970 Rows × 26 Columns        |
                                  +---------------------------------------+
```

### Source Distribution Table

| Source | Role | Records | Percentage | Attack Vectors |
| :--- | :--- | :---: | :---: | :--- |
| **[VeReMi](https://veremi-dataset.github.io/veremi)** | Primary Reference Source | **17,470** | **72.88%** | Normal, Type 1 (Constant Position), Type 2 (Constant Offset), Type 4 (Random Position), Type 8 (Random Offset), Type 16 (Eventual Stop) |
| **EdgeTrust-VANET** | Secondary Telemetry Source | **5,000** | **20.86%** | Normal, Blackhole, Sybil, Denial of Service (DoS), False Packet Injection |
| **V-RADD** | Routing Augmentation | **1,500** | **6.26%** | Grayhole (Selective Forwarding), Data Replay |
| **TOTAL** | Unified Research Dataset | **23,970** | **100.0%** | **7 Standardized Taxonomy Classes** |

---

## 📡 2. The 14 Leakage-Free Observable Features

In an operational VANET, a Roadside Unit (RSU) or Edge server receives Basic Safety Messages (BSMs) over the IEEE 802.11p / C-V2X radio layer. The classifier operates strictly on **14 observable features**:

$$\mathbf{X} = [p_x, p_y, v, \theta, a, N_{\text{sent}}, N_{\text{rcv}}, \rho_{\text{drop}}, L, N_{\text{retx}}, P_{\text{RSSI}}, T_t, \bar{T}_{\text{neighbor}}, T_{t-1}]$$

$$\mathbf{y} = \text{is\_malicious} \in \{0, 1\}$$

```
+----------------------------------------------------------------------------------------------------+
|                                    14 LEAKAGE-FREE ML FEATURES                                     |
+--------------------------+---------------------+-------+-------------------------------------------+
| Feature Name             | Category            | Unit  | Physical & Network Definition             |
+--------------------------+---------------------+-------+-------------------------------------------+
| position_x               | Mobility Kinematics | m     | UTM/Cartesian X coordinate reported in BSM|
| position_y               | Mobility Kinematics | m     | UTM/Cartesian Y coordinate reported in BSM|
| speed                    | Mobility Kinematics | m/s   | Velocity magnitude sqrt(vx² + vy²)        |
| direction                | Mobility Kinematics | deg   | Heading angle atan2(vy, vx) in [0, 360)   |
| acceleration             | Mobility Kinematics | m/s²  | Rate of speed change: Δv / Δt             |
| packet_sent              | Network Telemetry   | count | Cumulative periodic beacons transmitted   |
| packet_received          | Network Telemetry   | count | Beacons successfully received by observer |
| packet_drop_ratio        | Network Telemetry   | [0,1] | Fraction of expected beacons lost/dropped |
| latency                  | Network Telemetry   | ms    | Air transmission + processing delay       |
| retransmission_count     | Network Telemetry   | count | 802.11p MAC contention retry count        |
| signal_strength          | Network Telemetry   | dBm   | Received Signal Strength Indicator (RSSI) |
| trust_score              | EdgeTrust Engine    | [0,1] | Current dynamic reputation score (T_t)    |
| neighbor_trust_score_avg | EdgeTrust Engine    | [0,1] | Average trust of 300m spatial neighbors   |
| historical_trust_score   | EdgeTrust Engine    | [0,1] | Prior trust score in RSU cache (T_t-1)    |
+--------------------------+---------------------+-------+-------------------------------------------+
```

---

## 🧠 3. EdgeTrust Trust Engine Mathematical Formulation

The trust engine executes at the RSU **prior** to ML classification. It maps physical and radio observations to a normalized evidence score $E_t \in [0, 1]$ and updates vehicle trust using an Exponential Moving Average (EMA).

### 1. Dynamic Trust Update Equation
$$T_t = \alpha \cdot T_{t-1} + (1 - \alpha) \cdot E_t$$
where:
- $\alpha = 0.70$ is the **memory retention / forgetting factor**.
- $T_{t-1}$ is the vehicle's historical trust stored in the RSU database.
- When a vehicle first enters the RSU coverage zone, it is initialized with prior trust $T_0 = 1.0$.

### 2. Observable Real-Time Evidence ($E_t \in [0, 1]$)
$$E_t = 0.35 \cdot E_{\text{mobility}} + 0.30 \cdot E_{\text{delivery}} + 0.20 \cdot E_{\text{latency}} + 0.15 \cdot E_{\text{channel}}$$

#### A. Kinematic & Mobility Plausibility ($E_{\text{mobility}}$):
$$E_{\text{mobility}} = 0.40 \cdot E_{\text{pos}} + 0.30 \cdot E_{\text{spd}} + 0.30 \cdot E_{\text{accel}}$$
- **Speed Consistency**:
  $$E_{\text{spd}} = \begin{cases} 1.0, & v \le 45\text{ m/s} \\ \max\left(0.0, 1.0 - \frac{v - 45}{25}\right), & v > 45\text{ m/s} \end{cases}$$
- **Acceleration Consistency**:
  $$E_{\text{accel}} = \begin{cases} 1.0, & -9.0 \le a \le 6.0\text{ m/s}^2 \\ \max\left(0.0, 1.0 - \frac{|a - a_{\text{bound}}|}{12}\right), & \text{otherwise} \end{cases}$$
- **Kinematic Displacement Jump Check**:
  For successive reports separated by $\Delta t$, the observed displacement is $d = \|\mathbf{p}_t - \mathbf{p}_{t-1}\|$. If $d > (v + 10)\Delta t + 5$ (non-physical teleportation in VeReMi position falsification attacks):
  $$E_{\text{pos}} = \exp\left(-\frac{(d - d_{\max})^2}{2\sigma_d^2}\right)$$

#### B. Packet Delivery Plausibility ($E_{\text{delivery}}$):
$$E_{\text{delivery}} = \max(0.0, 1.0 - \text{packet\_drop\_ratio})$$
Normal vehicles show drop ratios $< 0.15$. Blackhole and Grayhole nodes intentionally drop $50\% - 100\%$ of forwarded packets, causing $E_{\text{delivery}}$ to plummet.

#### C. Transmission Latency Plausibility ($E_{\text{latency}}$):
$$E_{\text{latency}} = \begin{cases} 1.0, & L \le 25\text{ ms} \\ \max\left(0.05, \exp\left(-\frac{L - 25}{45}\right)\right), & L > 25\text{ ms} \end{cases}$$
Directly penalizes message replay attacks and queue-congested nodes.

#### D. Radio Channel Plausibility ($E_{\text{channel}}$):
Checks that received signal strength in dBm falls within the physical 802.11p range $[-90\text{ dBm}, -35\text{ dBm}]$.

### 3. Spatial Neighbor Trust Averaging
Using a spatial KD-tree over all active vehicles within direct radio communication radius $R \le 300\text{ m}$:
$$\bar{T}_{\text{neighbor}, i} = \frac{1}{|\mathcal{N}_i|} \sum_{j \in \mathcal{N}_i} T_j$$

---

## 🚫 4. Leakage Audit & Mathematical Proof

A crucial research contribution of EdgeTrust-VANET is eliminating **two forms of insidious data leakage** commonly found in naive VANET ML papers:

### Leakage Trap 1: Retrospective Attack Counters
Four fields exist in raw simulation logs: `false_packet_injection`, `blackhole_attack_attempts`, `sybil_attack_attempts`, `denial_of_service`.
- **The Cheat**: Including these counters gives an artificial **100.00% Accuracy and 100.00% F1 score** on a static test set.
- **The Deployment Proof**: When an unknown vehicle first encounters an RSU, its retrospective counters are unknown ($0$). When evaluated under realistic conditions, the cheating model **collapses from 100% to 86.05% F1**, missing malicious vehicles!
- **Our Honest Model**: Trained on the 14 observable features, it maintains a genuine **97.74% F1 and 97.77% Recall** on unseen vehicles.

### Leakage Trap 2: Random Row Splitting vs. Group Vehicle Splitting
- In spatiotemporal datasets, consecutive BSM messages from the same vehicle trajectory share strong autocorrelation.
- **Random Row Split**: Randomly splitting individual messages results in an inflated **99.29% F1 score** because the model memorizes vehicle-specific trajectory segments.
- **Group Vehicle Split (Our Protocol)**: Grouped strictly by `node_id` so that **no vehicle in the test set has ever been seen in training**. This reveals the true, publishable generalization score of **97.74% F1** (a 1.55% generalization gap).

---

## 🏆 5. 17-Model Benchmark Suite & Cross-Validation

The dataset was partitioned into:
- **70% Train** (16,885 records across 5,021 unique vehicles)
- **15% Validation** (3,546 records across 1,076 unique vehicles)
- **15% Holdout Test** (3,539 records across 1,077 unique vehicles)

All models were evaluated using **5-Fold Stratified Group Cross-Validation** on the training set (**Mean Weighted F1: 96.95% ± 1.12%**):

```text
==========================================================================================================
                         EDGETRUST-VANET 17-MODEL BENCHMARK LEADERBOARD
==========================================================================================================
Model Name                 Accuracy   Weighted F1   Recall    Precision  ROC-AUC    Latency (µs)   Size (KB)
----------------------------------------------------------------------------------------------------------
Random Forest (Champion)    97.77%      97.74%      97.77%     97.74%     99.82%     13,441 µs     6,441 KB
LightGBM (Top Edge GBDT)   97.74%      97.71%      97.74%     97.71%     99.78%        329 µs       669 KB
Stacking Ensemble          97.51%      97.48%      97.51%     97.48%     99.79%     14,522 µs     2,256 KB
Hist Gradient Boosting     97.40%      97.37%      97.40%     97.37%     99.79%      7,374 µs       653 KB
CatBoost                   97.29%      97.24%      97.29%     97.24%     99.79%         89 µs       248 KB
XGBoost                    97.29%      97.24%      97.29%     97.23%     99.77%        177 µs       464 KB
Voting Ensemble            97.20%      97.15%      97.20%     97.15%     99.80%     14,174 µs     3,905 KB
Gradient Boosting          97.17%      97.13%      97.17%     97.11%     99.71%        129 µs       535 KB
Decision Tree (CART)       96.84%      96.79%      96.84%     96.78%     96.53%         49 µs        18 KB
MLP Neural Network         96.13%      96.11%      96.13%     96.11%     99.16%         76 µs       392 KB
Extra Trees                92.43%      91.96%      92.43%     92.20%     98.71%     13,469 µs    11,314 KB
K-Nearest Neighbors        90.96%      90.65%      90.96%     90.49%     95.84%     15,329 µs     5,045 KB
SVM (RBF Kernel)           90.03%      89.19%      90.03%     90.52%     94.75%        242 µs       742 KB
AdaBoost                   88.25%      87.13%      88.25%     88.08%     94.67%      3,220 µs        38 KB
Logistic Regression        88.08%      86.78%      88.08%     88.35%     91.80%         65 µs         1 KB
Linear Discriminant (LDA)  87.93%      86.58%      87.93%     88.29%     91.90%         68 µs         1 KB
Gaussian Naive Bayes       82.17%      81.12%      82.17%     82.02%     86.83%         65 µs         1 KB
==========================================================================================================
```

### Key Scientific Takeaways
1. **Ensemble Champion**: **Random Forest** achieved the highest overall score (**97.74% F1, 97.77% Acc**) with an outstanding **99.82% ROC-AUC** and a tiny **1.77% False Alert Rate**.
2. **Edge Real-Time Champion**: **LightGBM** achieved essentially identical performance (**97.71% F1**) while executing an inference query in just **329 microseconds** ($0.33\text{ ms}$) with a compact 669 KB footprint, capable of evaluating over 3,000 vehicles per second on a single edge CPU core.
3. **Trust Meta-Feature Impact**: The three EdgeTrust features account for **55.86% of total feature importance** (`historical_trust`: $27.2\%$, `neighbor_trust`: $14.4\%$, `trust_score`: $14.3\%$), proving that dynamic trust provides a strong meta-feature layer for ML classifiers.

---

## 🛠️ 6. How to Run the Pipeline From Scratch

All pipeline scripts are modular and fully automated:

```bash
# 1. Extract raw VeReMi simulation logs & compute RSU features
python scripts/extract_veremi.py

# 2. Unify VeReMi (72.9%), EdgeTrust-VANET, and V-RADD into standardized dataset
python scripts/build_unified_dataset.py

# 3. Run the controlled Leakage Audit & generate proof report
python scripts/leakage_audit.py

# 4. Train the 17-model benchmark suite & export artifacts
python scripts/train_unified_pipeline.py

# 5. Launch the live interactive RSU dashboard
python dashboard/app.py
```
