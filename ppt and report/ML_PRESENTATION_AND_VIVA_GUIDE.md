# EdgeTrust-VANET: ML Presentation & Panel Defense Guide

---

## PART 1: Slide-by-Slide Presentation Script
*(Spoken in first person, simple conversational English, perfectly timed for 3–4 minutes total)*

---

### Slide 2: Dataset & Feature Engineering

> **"Good morning / afternoon respected panel members.**
> 
> Today, I will walk you through the Machine Learning and Data Engineering core of our EdgeTrust-VANET framework.
> 
> First, to ensure our research is credible and reproducible, we created and published our unified benchmark on **Kaggle** under the name **EdgeTrust-VANET**. 
> 
> Rather than relying on a single synthetic file, we built a **multi-source ETL pipeline** that standardized and harmonized **23,970 records** across three distinct simulation sources:
> * **72.9%** comes from the authentic **VeReMi** benchmark based on Luxembourg urban traffic.
> * **20.9%** comes from **EdgeTrust-VANET** attack profiles.
> * And **6.3%** comes from **V-RADD**, covering a total of 6 attack vectors with 70.6% normal and 29.4% malicious traffic.
> 
> From raw broadcast beacons, we engineered **14 real-time, leakage-free features** observable by an edge RSU:
> * **5 Mobility features**: position coordinates, speed, heading, and acceleration.
> * **6 Network features**: packet counts, drop ratio, latency, retransmissions, and signal strength.
> * And **3 Dynamic Trust features**: direct trust, historical trust, and 300-meter spatial neighbor trust.
> 
> Crucially, we audited and purged 4 retrospective counters—like 'attack attempts'—which leak the label and cause real-time models to collapse from 100% down to 86%. Furthermore, we enforced **Group-based vehicle splitting by Vehicle ID**, guaranteeing zero trajectory overlap across our 70% train, 15% validation, and 15% test splits.
> 
> We then performed a rigorous two-stage evaluation: first, **Validation on the split dataset**, achieving **97.74% F1-score**, and second, **Live Testing in the OMNeT++ / Veins simulation environment**, achieving **90.99% live accuracy**."

---

### Slide 3: Model Validation & Live RSU Testing Benchmark

> **"Moving to Slide 3: Model Validation and Live RSU Testing.**
> 
> We benchmarked **17 machine learning models** spanning Tree Ensembles, GBDTs, Linear models, and Neural Networks across two distinct stages:
> 
> 1. **Validation on the Split Dataset (3,539 holdout samples)**: Using 5-Fold Stratified Group Cross-Validation, our models achieved an average of 96.95%. Random Forest achieved **97.77% accuracy** and **97.74% F1-score**, closely followed by LightGBM at 97.74% and CatBoost at 97.29%.
> 
> 2. **Testing on the Live OMNeT++ / Veins RSU Environment**: We then deployed these models directly onto the simulated RSU edge and tested them on **2,331 live telemetry packets** containing real wireless channel noise and fading.
> 
> If you look at the **comparison graph on the right**:
> * The **left chart** compares **Validation Accuracy** in blue against **Live Test Accuracy** in green. As you can see, **Random Forest is our champion**, retaining a stellar **90.99% live test accuracy** and **90.97% live F1-score**, while maintaining the lowest missed attack rate at only **6.89%**.
> * For ultra-constrained micro-RSUs, **Logistic Regression and LDA** proved exceptional—delivering **89.28% live F1-score** with an ultra-low latency of **82.8 microseconds**, processing over **12,000 packets per second** with just a **1 kilobyte** memory footprint.
> * Looking at the **right chart** showing Live F1-scores, modern tree ensembles consistently outperform deep networks. Unregularized MLP Neural Networks and SVM collapsed down to 44% live F1 with over 97% false alarm rates due to raw radio jitter. This proves that tree bagging and regularized linear classifiers are the most reliable choices for physical VANET deployments."

---

### Slide 4: Feature Importance & Edge Trust Score Formula

> **"Moving to Slide 4: Feature Importance and the Edge Trust Formulation.**
> 
> Looking at the **top-left chart**, our feature importance analysis reveals that **Trust features dominate 55.86%** of all model decisions. 
> * Specifically, **Historical Trust** is the single most predictive signal at **27.2%**, followed by **Neighbor Trust Average** at **14.4%**, and **Direct Trust** at **14.3%**. 
> * Kinematic features like spatial position account for 28.5%, while speed, heading, packet drop ratio, and latency make up the remaining 44.1%. 
> 
> This proves mathematically that an RSU-computed dynamic trust score provides the decisive signal needed to catch sophisticated attacks that mimic legitimate vehicle speeds.
> 
> To compute this real-time trust efficiently, we developed an **Exponential Moving Average (EMA) Trust Update formula**:
> 
> $$T_t = \alpha \cdot T_{t-1} + (1 - \alpha) \cdot E_t$$
> 
> Where:
> * $T_t$ is the current trust score,
> * $T_{t-1}$ is the historical trust score,
> * $\alpha = 0.70$ is our memory retention factor,
> * And $E_t$ is observable real-time evidence combining mobility consistency (35%), packet delivery (30%), latency (20%), and channel signal strength (15%).
> 
> Neighbor trust is aggregated within a **300-meter radio range using an $O(\log N)$ KD-Tree**. 
> 
> We chose **$\alpha = 0.70$** because sensitivity testing shows it provides the sweet spot: it swiftly degrades an attacker's trust below the **0.40 blocking threshold within just 4 packets**, while having enough memory retention to absorb transient radio noise so that honest braking vehicles or ambulances are never falsely disconnected.
> 
> Thank you, and I am now ready for your questions."

---

*(Quick reference scripts if the panel asks you to flip to Slides 5, 6, 7, or 10)*:

* **Slide 5 (Alpha = 0.70 Sensitivity)**: *"This sensitivity curve proves why $\alpha=0.70$ is optimal. If $\alpha=0.30$, the system is too nervous and falsely penalizes cars during brief fading. If $\alpha=0.90$, it takes over 15 packets to detect an attack, which is too slow. At 0.70, an attacker drops below 0.40 by Packet 4."*
* **Slide 6 (Edge RSU Trade-Off)**: *"This Pareto bubble chart shows hardware feasibility. For multi-core RSUs, Random Forest gives peak 90.99% accuracy. For low-power micro-controllers, Logistic Regression provides 89.28% F1 at 82.8 microseconds and 1 KB RAM. CatBoost offers the best GBDT trade-off at 118 microseconds."*
* **Slide 7 (Error Rates: FAR vs. MAR)**: *"In VANETs, missed attacks cause collisions and false alarms cause traffic jams. Random Forest, Extra Trees, and Logistic Regression sit inside our green 'Secure Operating Zone' with False Alarms under 15% and Missed Attacks under 10%."*
* **Slide 10 (Technical Decisions)**: *"Our core architectural contribution is a two-tier safety policy: Tier 1 computes $O(1)$ heuristic trust instantly; Tier 2 runs the ML model. An RSU only BLOCKs a vehicle if BOTH the trust score is below 0.40 AND the ML model predicts malicious, guaranteeing maximum availability."*

---
---

## PART 2: Fundamental Concepts Explained (Ground Up)

### 1. What is "Validation" vs. "Testing"?
* **What is Validation?**
  * Validation is evaluating models on held-out data during the development phase to tune hyperparameters, compare different algorithms, and check if models are overfitting.
  * In our project: Validation was performed on the **split dataset** (3,539 holdout vehicle samples from `data/unified_edgetrust_dataset.csv`, reported in `results/unified_model_metrics.json`).
* **What is Testing?**
  * Testing is evaluating the finalized, trained model on completely independent, unseen real-world or simulated deployment data that was never touched during training.
  * In our project: Testing was performed on **2,331 live telemetry packets** extracted directly from a running **OMNeT++ / Veins + SUMO** vehicular network simulation.
* **Why the difference matters:**
  * High validation accuracy on a CSV file does not guarantee good performance under real radio packet collisions, multipath fading, and variable latency. Our two-stage approach proves that our models genuinely work in live simulation.

---

### 2. What is Cross-Fold Validation, and why Stratified Group K-Fold?
* **What is K-Fold Cross Validation?**
  * You divide your training data into $K$ equal parts (we used $K=5$). You train the model on 4 parts (80%) and validate on the remaining 1 part (20%). You repeat this 5 times so every sample is used for validation once, and compute the mean and standard deviation.
* **What does "Stratified" mean?**
  * It means every fold preserves the exact same class balance (70.6% benign, 29.4% malicious). This prevents a fold from having 0% malicious samples, which would bias the metrics.
* **What does "Group" mean (GroupKFold)?**
  * Vehicles broadcast multiple packets over time (a trajectory). If we split rows randomly, packet #1 of Car-A would be in training, and packet #2 of Car-A would be in validation. The model would "memorize" Car-A rather than learning general misbehavior!
  * **GroupKFold by `node_id` ensures that all packets from a given vehicle are strictly confined to either Train or Validation.**
* **Why did we do this?**
  * To guarantee zero temporal and trajectory leakage. Our 5-fold CV score of **$96.95\% \pm 1.12\%$** proves consistent, reliable learning across unseen vehicles.

---

### 3. What is Data Leakage, and how did we prevent it?
* **What is Data Leakage?**
  * Data leakage occurs when information from outside the training dataset or information unavailable at prediction time is inadvertently fed into the model during training, giving artificially inflated scores.
* **Type 1: Retrospective Feature Leakage (Post-Attack Counters)**:
  * In naive datasets, features like `blackhole_attack_attempts` or `false_packet_injection` exist. These are ground-truth counters logged *after* an attack is concluded. A real RSU listening to a radio beacon cannot know this.
  * **Our Action**: We ran a leakage audit script (`scripts/leakage_audit.py`). Models trained with those counters showed 100% fake F1-score, but collapsed down to 86% in live deployment. We stripped them out completely, keeping only 14 real-time observable features.
* **Type 2: Trajectory / Temporal Leakage**:
  * Splitting sequential rows randomly causes high correlation between adjacent time-steps.
  * **Our Action**: Enforced strict GroupKFold on `node_id`.

---

### 4. What is the Trust Engine formula, and why $\alpha = 0.70$?
* **The Formula**:
  $$T_t = \alpha \cdot T_{t-1} + (1 - \alpha) \cdot E_t$$
* **What are the terms?**
  * $T_t \in [0, 1]$: Current trust score of the vehicle at time $t$.
  * $T_{t-1} \in [0, 1]$: Previous trust score (initialized to 1.0 or historical reputation).
  * $\alpha = 0.70$: Forgetting / retention factor. 70% weight to history, 30% weight to new evidence.
  * $E_t \in [0, 1]$: Observable real-time evidence score:
    $$E_t = 0.35 E_{\text{mobility}} + 0.30 E_{\text{delivery}} + 0.20 E_{\text{latency}} + 0.15 E_{\text{channel}}$$
* **Why $\alpha = 0.70$?**
  * If $\alpha$ is small (e.g. 0.30): A single dropped beacon due to tall buildings drops trust immediately. Normal vehicles get blocked (high false alarms).
  * If $\alpha$ is large (e.g. 0.90): An attacker can transmit 15 malicious beacons before trust drops below the 0.40 threshold, poisoning routing tables.
  * **At $\alpha = 0.70$**: Trust gracefully tolerates 1–2 noisy beacons, but decisively drops an attacker below the 0.40 threshold by **Packet 4**.

---

### 5. What is the KD-Tree Neighbor Trust?
* **What is a KD-Tree?**
  * A k-dimensional tree is a space-partitioning data structure that allows fast nearest-neighbor lookups in $O(\log N)$ time rather than checking every vehicle in $O(N^2)$ time.
* **Why do we need it?**
  * In VANETs, vehicles within the 300-meter radio transmission range observe each other. If Car-A claims to be at $(x_1, y_1)$, but 5 neighbors around Car-A report packet drops or conflicting location data, Car-A's neighbor trust score ($\text{neighbor\_trust\_score\_avg}$) drops.
  * Using KD-Tree allows an RSU with 200 vehicles in range to compute neighbor spatial consensus in under 1 millisecond.

---

### 6. Why did Random Forest win, and why did MLP/SVM fail on Live Data?
* **Why Random Forest won (90.99% Live Acc, 90.97% Live F1, 6.89% MAR)**:
  * Vehicular safety data has hard, non-linear physical rules (e.g., packet drop ratio $> 0.40$ or trust $< 0.40$). Decision trees natively learn orthogonal step boundaries.
  * Bagging (aggregating 150 diverse trees) acts as a low-pass filter against non-Gaussian radio noise and packet jitter.
* **Why MLP Neural Net and SVM collapsed (FAR $> 97\%$)**:
  * Neural networks and SVMs construct smooth, continuous hyperplanes. When exposed to raw radio telemetry with transient latency spikes and signal fades, their decision boundaries get distorted, classifying normal noise as attacks. Without spatial trust smoothing, they are unsuitable for live edge deployment.

---
---

## PART 3: Comprehensive Panel Q&A Defense Cheat Sheet

### Category A: Dataset & Preprocessing

#### Q1: "Why did you create a unified dataset instead of just using VeReMi?"
* **Short Answer**: 
  *"VeReMi is the gold standard for BSM position falsification, but it lacks specific routing attacks like Blackhole and high-rate DoS flooding. By harmonizing VeReMi (72.9%) with EdgeTrust-VANET (20.9%) and V-RADD (6.3%), our unified benchmark of 23,970 records covers 6 distinct attack vectors under a single standardized 14-feature schema."*

#### Q2: "What is the class balance in your dataset, and how did you handle imbalance?"
* **Short Answer**:
  *"Our dataset is 70.6% benign (16,917 records) and 29.4% malicious (7,053 records), which reflects realistic urban attack densities. We handled this using Stratified Group K-Fold splitting, class-weighted training loss, and evaluating with Weighted F1-score and ROC-AUC rather than misleading raw accuracy."*

#### Q3: "What features did you eliminate to prevent data leakage?"
* **Short Answer**:
  *"We removed 4 retrospective attack counters: `blackhole_attack_attempts`, `false_packet_injection`, `sybil_node_count`, and `dos_frequency_ratio`. Our leakage audit proved that keeping these gives a deceptive 100% F1-score on paper, but drops to 86% in live testing because an RSU cannot observe future summary statistics."*

#### Q4: "What are the 14 features, and can a real RSU observe them in real time?"
* **Short Answer**:
  *"Yes, all 14 are strictly observable from 802.11p BSM headers and standard RSU network counters: 5 Mobility features ($x, y$, speed, heading, accel), 6 Network features (sent, received, drop ratio, latency, retransmissions, RSSI), and 3 Trust metrics computed by our RSU trust engine."*

---

### Category B: ML Benchmarks & Validation vs. Live Testing

#### Q5: "What is the exact difference between your Validation results and your Live Test results?"
* **Short Answer**:
  *"Validation was conducted on our split dataset of 3,539 holdout vehicle samples, where Random Forest achieved 97.77% accuracy and 97.74% F1. Testing was performed on 2,331 live telemetry packets generated inside an active OMNeT++ / Veins simulation with real radio propagation, where Random Forest achieved 90.99% accuracy and 90.97% F1."*

#### Q6: "Why did model accuracy drop from 97.77% in validation to 90.99% in live testing?"
* **Short Answer**:
  *"The 6.78% gap is the honest reality of physical wireless channels. Live OMNeT++ simulations introduce dynamic obstacles, Doppler shifts, packet collisions, and radio jitter that do not exist in static CSV datasets. Achieving 90.99% on live telemetry with under 7% missed attacks confirms genuine deployment robustness."*

#### Q7: "Why did you test 17 models? Isn't Random Forest enough?"
* **Short Answer**:
  *"RSUs have diverse hardware constraints. A high-end multi-core edge server can easily run Random Forest (90.99% accuracy, 6.4 MB). But for low-power solar or microcontroller RSUs, our benchmark proved that Logistic Regression and LDA execute in just 82.8 microseconds ($>12,000$ packets/sec) using only 1.0 KB of memory while delivering 89.28% F1-score."*

#### Q8: "What are False Alarm Rate (FAR) and Missed Attack Rate (MAR), and which is more dangerous?"
* **Short Answer**:
  *"FAR is the percentage of legitimate vehicles wrongly flagged as malicious; MAR is the percentage of attacks that slip through undetected. In VANETs, MAR is more safety-critical because a missed attack can cause routing collapse or physical crashes. Random Forest achieved our lowest MAR at only 6.89%."*

---

### Category C: Trust Formulation & Security Policy

#### Q9: "What if a normal ambulance brakes hard or speeds up? Will your trust engine falsely BLOCK it?"
* **Short Answer**:
  *"No, because of our two-tier safety policy. Tier 1 updates trust using $\alpha = 0.70$, which absorbs transient kinematic spikes. Crucially, an RSU never blocks based on trust alone—it requires BOTH trust below 0.40 AND an ML malicious classification to trigger a BLOCK. An ambulance broadcasting authentic BSMs with zero packet drop will maintain normal network features and will never be blocked."*

#### Q10: "How does the RSU calculate neighbor trust without getting overwhelmed?"
* **Short Answer**:
  *"The RSU uses a 2D KD-Tree indexed by vehicle $(x, y)$ coordinates. Whenever a vehicle broadcasts, neighbor queries within a 300-meter radius execute in $O(\log N)$ time, allowing instant consensus calculation even in dense traffic with hundreds of vehicles."*

#### Q11: "Why do trust features dominate 55.86% of the model's decisions?"
* **Short Answer**:
  *"Because sophisticated attackers spoof realistic kinematic values—like driving at 45 km/h—to fool simple speed checks. However, an attacker cannot easily fake historical consistency and multi-vehicle spatial consensus. The trust score captures temporal behavior over time, making it the hardest feature for an adversary to bypass."*

---
---

## Summary Cheat Sheet Table

| Item | What to Remember |
| :--- | :--- |
| **Kaggle Link** | Published under `EdgeTrust-VANET` for open-science reproducibility. |
| **Dataset Size** | 23,970 records: VeReMi (72.9%), EdgeTrust (20.9%), V-RADD (6.3%). |
| **Features** | 14 RSU-observable features (5 Mobility, 6 Network, 3 Trust). Zero leakage. |
| **Validation Score** | 97.74% F1, 97.77% Accuracy on 3,539 holdout split (5-Fold Group CV: $96.95\% \pm 1.12\%$). |
| **Live Test Score** | 90.99% Accuracy, 90.97% F1 on 2,331 OMNeT++/Veins simulation packets. |
| **Top Champion** | Random Forest (lowest missed attack rate: 6.89%). |
| **Edge Champions** | Logistic Regression & LDA (89.28% Live F1, 82.8 $\mu$s latency, 1.0 KB RAM). |
| **Trust Weight** | $\alpha = 0.70$ (quarantines attackers by Packet 4, absorbs transient fading). |
| **Safety Policy** | Requires **both** Trust $< 0.40$ **and** ML Malicious prediction to BLOCK. |
