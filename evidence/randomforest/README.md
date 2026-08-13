# Model Selection Evidence: Random Forest vs All Models

This directory contains empirical evidence and plots comparing **Random Forest** (the final chosen classifier) against all other 8 machine learning models implemented in the EdgeTrust-VANET system.

These comparative results serve as the primary empirical justification for the choice of classifier in the paper.

## Files

- [generate_randomforest_evidence.py](file:///Users/vivekchitturi/Desktop/edgetrust-ml/evidence/randomforest/generate_randomforest_evidence.py): Reproducible evidence-generation script.
- [all_models_metrics.json](file:///Users/vivekchitturi/Desktop/edgetrust-ml/evidence/randomforest/all_models_metrics.json): Main numeric evidence containing exact seed-42 metrics and repeated-split statistics.
- [all_model_runs.csv](file:///Users/vivekchitturi/Desktop/edgetrust-ml/evidence/randomforest/all_model_runs.csv): Raw metrics for all nine classifiers across 20 stratified holdout runs.
- [plots/all_models_metrics.png](file:///Users/vivekchitturi/Desktop/edgetrust-ml/evidence/randomforest/plots/all_models_metrics.png): Exact seed-42 holdout metric comparisons.
- [plots/all_models_stability.png](file:///Users/vivekchitturi/Desktop/edgetrust-ml/evidence/randomforest/plots/all_models_stability.png): F1 boxplot stability and average error counts (FP & FN) across 20 repeated splits.
- [plots/all_models_confusion.png](file:///Users/vivekchitturi/Desktop/edgetrust-ml/evidence/randomforest/plots/all_models_confusion.png): Confusion matrices for all 9 models on seed 42.
- [plots/all_models_feature_importance.png](file:///Users/vivekchitturi/Desktop/edgetrust-ml/evidence/randomforest/plots/all_models_feature_importance.png): Feature importances across all tree-based ensemble models.

---

## Dataset & Feature Setup

The script evaluates the models using the rich VANET dataset (`data/vanet_malicious_nodes.csv`) with the standard **14-feature** input vector:

- **Class distribution**: 3,738 Normal (74.76%), 1,262 Malicious (25.24%)
- **Stratified holdout**: 80% training set (4,000 nodes), 20% test set (1,000 nodes)
- **Features (14)**: GPS coordinates (`position_x`, `position_y`), `speed`, `direction`, `acceleration`, network packets (`packet_sent`, `packet_received`), packet drop ratio (`packet_drop_ratio`), `latency`, message retransmissions (`message_retransmission_count`), signal strength (`signal_strength`), and RSU trust score telemetry (`trust_score`, `neighbor_trust_score_avg`, `historical_trust_score`).

---

## 1. Exact Seed-42 Holdout Results

This split corresponds to the exact evaluation split used by `scripts/analyze_vanet_nodes.py` to pick the best model for the dashboard. 

| Model | Accuracy | Weighted F1 | Precision | Recall | False Positives | False Negatives |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest** | 95.60% | **95.71%** | 96.25% | 95.60% | 44 | 0 |
| **AdaBoost** | 95.60% | **95.71%** | 96.25% | 95.60% | 44 | 0 |
| **Gradient Boosting** | 95.30% | 95.41% | 95.85% | 95.30% | 43 | 4 |
| **Decision Tree** | 94.60% | 94.70% | 95.04% | 94.60% | 43 | 11 |
| **Extra Trees** | 94.40% | 94.52% | 94.92% | 94.40% | 46 | 10 |
| **Gaussian Naive Bayes** | 93.60% | 93.80% | 94.66% | 93.60% | 60 | 4 |
| **SVM (RBF)** | 93.40% | 93.50% | 93.76% | 93.40% | 46 | 20 |
| **Logistic Regression** | 92.20% | 92.22% | 92.24% | 92.20% | 41 | 37 |
| **K-Nearest Neighbors** | 89.00% | 89.03% | 89.06% | 89.00% | 57 | 53 |

### 🔍 Analysis of Holdout Performance (Why & What is Happening)

*   **Ensemble Tree Dominance (Random Forest & AdaBoost)**: 
    *   **What is happening**: Both classifiers achieve a tied peak F1 of **95.71%** and **zero false negatives** (catching all 252 malicious nodes).
    *   **Why**: The VANET dataset contains highly discriminative features like RSU-computed `trust_score` and network `packet_drop_ratio`. Tree-based models are excellent at finding axis-aligned decision boundaries (e.g., `trust_score < 0.4` or `packet_drop_ratio > 0.3`). By grouping these splits into ensembles (bootstrap bagging for Random Forest and adaptive boosting for AdaBoost), they eliminate individual tree overfitting and isolate attackers perfectly.
*   **Linear & Kernel Limitations (Logistic Regression & SVM)**:
    *   **What is happening**: SVM (RBF) misses 20 attacks and Logistic Regression misses 37 attacks (high False Negatives).
    *   **Why**: Logistic Regression attempts to fit a single flat linear hyperplane through a 14-dimensional space. Because network anomalies (like latency spikes) interact non-linearly with physical mobility coordinates, a linear boundary is too simplistic and misclassifies subtle attackers. SVM with RBF kernel maps features to a high-dimensional space but relies on distance metrics. Extreme outlier values in packets or latency during DOS attacks skew the distance calculations, blurring the boundary and letting malicious nodes slip through.
*   **Violated Assumptions (Gaussian Naive Bayes)**:
    *   **What is happening**: GNB produces the highest false alarm rate with **60 false positives**.
    *   **Why**: Naive Bayes assumes all features are conditionally independent given the class. In a real-world VANET, this assumption is heavily violated: `packet_sent` and `packet_received` are highly correlated; `packet_drop_ratio` is derived from them; and `trust_score` is directly computed from `historical_trust_score`. Multiplying these dependent feature probabilities causes GNB to overstate class confidence, leading to frequent false alarms.

---

## 2. Repeated Split Stability (20 Stratified Runs)

To ensure the seed-42 tie and ranking are not statistical anomalies, we evaluated all models across 20 distinct stratified 80/20 splits using seeds `0` through `19`. Ranks are computed for each split (1 = highest F1, 9 = lowest F1).

| Model | Mean Accuracy | Mean Weighted F1 | Mean Precision | Mean Recall | Avg. False Positives | Avg. False Negatives | Mean Rank |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **AdaBoost** | 95.455% | **95.572%** | 96.145% | 95.455% | 45.15 | **0.30** | **1.05** |
| **Random Forest** | 95.365% | **95.482%** | 96.038% | 95.365% | 45.15 | 1.20 | 1.65 |
| **Gradient Boosting** | 95.200% | 95.315% | 95.821% | 95.200% | 44.70 | 3.30 | 2.95 |
| **Decision Tree** | 94.670% | 94.772% | 95.123% | 94.670% | 42.80 | 10.50 | 4.30 |
| **Extra Trees** | 94.380% | 94.492% | 94.867% | 94.380% | 44.90 | 11.30 | 4.55 |
| **Gaussian Naive Bayes** | 93.185% | 93.393% | 94.259% | 93.185% | 61.80 | 6.35 | 6.10 |
| **SVM (RBF)** | 92.755% | 92.847% | 93.048% | 92.755% | 46.40 | 26.05 | 7.00 |
| **Logistic Regression** | 92.230% | 92.239% | 92.275% | 92.230% | 39.90 | 37.80 | 7.85 |
| **K-Nearest Neighbors** | 89.610% | 89.626% | 89.677% | 89.610% | 53.50 | 50.40 | 9.00 |

### 🔍 Analysis of Stability & Error Patterns (Why & What is Happening)

*   **AdaBoost vs Random Forest Stability (Rank 1.05 vs 1.65)**:
    *   **What is happening**: AdaBoost consistently achieves the highest Weighted F1 across 19 out of 20 splits, while Random Forest is slightly more variable and averages 1.20 missed attacks per split compared to AdaBoost's 0.30.
    *   **Why**: AdaBoost is a sequential boosting algorithm that trains estimators iteratively, dynamically increasing the weights of samples that were misclassified in previous rounds. Since malicious behavior profiles contain consistent boundary-line cases, AdaBoost's focused error targeting allows it to refine its boundary splits with extreme precision. 
    *   Random Forest relies on bagging and *random feature selection* at each node split (evaluating only a subset of $\sqrt{N} \approx 3$ features). If a node split in a tree is restricted to weak mobility features (like `direction` or `acceleration`) without access to the highly informative `trust_score` or `packet_drop_ratio`, that tree's predictive power is diminished. While averaging 100 trees cancels out most of this variance, this feature restriction introduces a minor performance penalty across random data splits, resulting in a slightly lower mean rank.
*   **Ensemble Error Reductions**:
    *   **What is happening**: Both AdaBoost and Random Forest drastically outperform a single Decision Tree (which averages 10.50 missed attacks).
    *   **Why**: A single Decision Tree creates a highly detailed, rigid boundary that is susceptible to variance and overfitting. Ensemble averaging (bagging or boosting) smooths out these step-like decision boundaries, reducing generalization error and preventing legitimate nodes from being warned or blocked due to temporary network noise.

---

## 3. Edge-Deployment Computational Efficiency

For edge deployment on Roadside Units (RSUs) or On-Board Units (OBUs), CPU time and memory footprints are critical.

| Model | Mean Fit Time | Mean Predict Latency (per sample) | Mean Pickled Model Size |
| :--- | :---: | :---: | :---: |
| **AdaBoost** | 0.1814 s | 2.846 us | **27.26 KB** |
| **Random Forest** | 0.2711 s | 5.323 us | 2,527.69 KB |
| **Gradient Boosting** | 0.9435 s | **0.788 us** | 190.78 KB |
| **Decision Tree** | 0.0097 s | 0.101 us | 6.82 KB |
| **Extra Trees** | 0.1224 s | 7.753 us | 9,517.13 KB |
| **Gaussian Naive Bayes** | 0.0006 s | 0.129 us | **0.98 KB** |
| **SVM (RBF)** | 0.3165 s | 24.248 us | 121.28 KB |
| **Logistic Regression** | 0.0038 s | 0.095 us | **0.76 KB** |
| **K-Nearest Neighbors** | 0.0010 s | 35.307 us | 564.83 KB |

### 🔍 Analysis of Computational Efficiency (Why & What is Happening)

*   **Pickled Model Size Differences**:
    *   **What is happening**: Extra Trees (**9.5 MB**) and Random Forest (**2.5 MB**) are massive, while AdaBoost is tiny (**27 KB**).
    *   **Why**: Serialized model size is directly proportional to the total number of decision splits/nodes stored in memory. Random Forest trains 100 deep estimators. Each tree is allowed to grow until nodes are pure, resulting in deep trees containing thousands of branches. Extra Trees selects split thresholds completely at random rather than choosing the optimal split point. This randomized partitioning requires much deeper trees to achieve leaf purity, producing a massive quantity of nodes. AdaBoost, by default, uses **Decision Stumps** (`max_depth=1`) as base estimators. Storing 50 decision stumps requires saving only 50 single-split rules (a single threshold and feature index per stump), making it exceptionally lightweight.
*   **Predict Latency Trade-offs**:
    *   **What is happening**: Gradient Boosting (**0.79 us**) is the fastest, Random Forest (**5.32 us**) is moderate, while SVM (**24.25 us**) and KNN (**35.31 us**) are extremely slow.
    *   **Why**: 
        *   Gradient Boosting uses relatively shallow trees (default depth of 3 or 4) and sums their predictions. Traversing 100 trees of depth 4 requires executing at most 400 branch comparisons. Random Forest requires traversing 100 deep trees (depth > 15), executing up to 2,000 branch checks.
        *   SVM (RBF) is kernel-based and must compute expensive exponential radial basis function distance calculations against support vectors at runtime.
        *   K-Nearest Neighbors is a lazy learner. It stores the entire training dataset and, for every single prediction, must compute the Euclidean distance to all 4,000 training samples in memory, making it the least viable candidate for real-time edge V2X communication.

---

## 4. Selecting Random Forest vs Alternatives: Paper Rationale

Based on the empirical evidence, the paper can justify the selection of **Random Forest** using two different strategies:

### Strategy A: Retain Random Forest (Focus on Interpretability & Robustness)
> "Random Forest and AdaBoost achieved identical predictive performance (95.71% weighted F1) on our baseline holdout. Random Forest was selected as the final classifier for EdgeTrust-VANET because its bag-of-trees ensemble architecture provides a stable, multi-split average feature-importance estimation across a wide range of features. This allows the RSU to report explainable trust factors to neighboring units. Since AdaBoost achieved statistically competitive performance (95.57% mean F1) with lower inference latency and a smaller model size, it is documented as a key lightweight alternative for resource-constrained edge systems."

### Strategy B: Switch to AdaBoost (Focus on Edge Resource Constraints)
> "Although Random Forest and AdaBoost achieved identical metrics on the initial holdout split, repeated stratified splits showed AdaBoost obtaining a slightly superior average rank (1.05 vs 1.65) and lower average false negatives (0.30 vs 1.20). Furthermore, AdaBoost requires only 2.85 us of prediction latency per sample and a pickled model size of just 27.26 KB (a 98.9% reduction compared to Random Forest). Therefore, AdaBoost is selected as the optimal model for resource-constrained edge roadside units."

---

## Re-running the Analysis
To regenerate all results and plots from the project root:
```bash
.venv/bin/python evidence/randomforest/generate_randomforest_evidence.py
```
