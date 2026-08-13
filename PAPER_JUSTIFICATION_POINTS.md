# Paper Justification Points

This file lists the places in the codebase where the paper should explain "why this choice instead of alternatives?" These are the spots that need either a citation, an experiment, an ablation table, a sensitivity analysis, or a clear design rationale.

## Highest Priority For The Paper

### 1. Why Random Forest Is The Final Model

Code locations:

- `scripts/analyze_vanet_nodes.py`: trains nine classifiers and selects the best by weighted F1 score.
- `results/vanet_model_metrics.json`: stores the current comparison results.
- `README.md`: states that Random Forest is the best model.

Current choice:

- Random Forest is presented as the best VANET model.
- Current metrics: 95.60% accuracy, 95.71% F1, 96.25% precision, 95.60% recall.
- AdaBoost has the same saved accuracy/F1/precision/recall in `results/vanet_model_metrics.json`, but Random Forest is marked as best because it appears first in the model dictionary and ties are not explicitly handled.

What the paper must justify:

- Why Random Forest is preferred over AdaBoost if their metrics tie.
- Why weighted F1 is the primary ranking metric.
- Whether inference time, interpretability, robustness, or feature importance support the Random Forest choice.

Recommended evidence:

- Add a model-comparison table with all nine models.
- Add tie-breaking criteria: lower false negatives, lower inference latency, stability across folds, interpretability, or feature-importance usefulness.
- Run repeated stratified cross-validation instead of relying only on one 80/20 split.

### 2. Why Weighted F1 Is Used Instead Of Accuracy

Code locations:

- `scripts/analyze_vanet_nodes.py`: uses weighted F1 to select `best_model`.
- `scripts/train_model.py`: also chooses the best model by weighted F1.
- `dashboard/app.py`: dashboard ranks models by F1 score.
- `dashboard/templates/index.html`: labels the arena as ranked by F1 score.

Current choice:

- Weighted F1 is the primary metric.
- Accuracy, precision, recall, false alert rate, and confusion matrix are also recorded.

What the paper must justify:

- VANET attack detection is imbalanced: normal nodes are the majority and malicious nodes are the minority.
- Accuracy can look high even when the classifier misses attacks.
- Weighted F1 balances precision and recall while accounting for class imbalance.

Recommended evidence:

- Include class distribution: 3,738 normal and 1,262 malicious nodes, about 25.2% attack rate.
- Include confusion matrices and false negative counts.
- Consider also reporting macro F1, ROC-AUC, PR-AUC, and false positive/false negative rates.

Generated evidence: `evidence/weighted_f1/` contains reproducible metrics, CSV output, and plots for the majority-class baseline and all nine classifiers. The current evidence shows that always predicting Normal gives 74.8% accuracy but 0 malicious-class F1 and 252 missed attacks on the test split.

### 3. Why Alpha Equals 0.7 In The Trust Engine

Code locations:

- `scripts/trust_score.py`: default `HybridTrustEngine(alpha=0.7)`.
- `scripts/combined_decision.py`: initializes `HybridTrustEngine(alpha=0.7)` with comment "as proven".

Current choice:

- `alpha = 0.7`.
- Formula: `new_trust = alpha * previous_trust + (1 - alpha) * evidence_score`.
- Higher alpha means more weight on history; lower alpha means faster reaction to new evidence.

What the paper must justify:

- Why 0.7 is the best tradeoff between stability and responsiveness.
- Why not 0.5, 0.6, 0.8, or 0.9.
- How quickly the trust score drops under attack and recovers after normal behavior.

Recommended evidence:

- Add a sensitivity analysis table for alpha values such as 0.3, 0.5, 0.7, and 0.9.
- Measure detection delay: how many malicious packets until a vehicle becomes Suspicious or Blocked.
- Measure false punishment: how many benign noisy packets incorrectly lower trust too much.
- Plot trust decay/recovery curves for different alpha values.

### 4. Why Trust Thresholds Are 0.70 And 0.40

Code locations:

- `scripts/trust_score.py`: classifies `Trusted`, `Suspicious`, and `Blocked` at 0.70 and 0.40.
- `dashboard/app.py`: uses `trust < 0.4` for `WARN` or `BLOCK`.
- `dashboard/templates/index.html`: trust sliders use a 0.0 to 1.0 scale.

Current choice:

- `trust >= 0.70`: Trusted.
- `0.40 <= trust < 0.70`: Suspicious.
- `trust < 0.40`: Blocked.
- Dashboard decision logic blocks only when both trust is low and ML predicts malicious.

What the paper must justify:

- Why 0.70 and 0.40 are the operational boundaries.
- Why low trust alone creates a warning, not always a block.
- Whether these thresholds minimize false positives or reduce missed attacks.

Recommended evidence:

- Threshold sweep over `(trusted_threshold, blocked_threshold)`.
- Table showing precision, recall, false positives, and false negatives at different thresholds.
- Explain RSU safety policy: blocking should require stronger evidence than warning.

Generated evidence: `evidence/trust_thresholds/` contains reproducible trust decay/recovery trajectories, alternative threshold comparisons, noisy-benign sensitivity counts, and the implemented hybrid `WARN`/`BLOCK` policy. It supports a staged-response rationale, but does not by itself prove that 0.70 and 0.40 are globally optimal.

### 5. Why These Features Are Included Or Excluded

Code locations:

- `scripts/analyze_vanet_nodes.py`: defines the 14-feature richer VANET feature set.
- `scripts/preprocess.py`: defines the older 8-feature feature set.
- `README.md`: says retrospective attack labels are excluded to prevent leakage.

Current choice:

- Rich workflow includes mobility, network, and trust features.
- Rich workflow excludes confirmed attack counters: `false_packet_injection`, `blackhole_attack_attempts`, `sybil_attack_attempts`, and `denial_of_service`.
- There is a code comment conflict: one comment says trust scores are excluded to prevent leakage, but the actual `FEATURES` list includes `trust_score`, `neighbor_trust_score_avg`, and `historical_trust_score`.

What the paper must justify:

- Which signals are available before classification at an RSU.
- Why post-hoc attack counters must be excluded.
- Whether trust-score features are legitimate real-time RSU-computed inputs or leakage-prone labels.

Recommended evidence:

- Add an ablation table:
  - Mobility only
  - Network only
  - Trust only
  - Mobility + network
  - Mobility + network + trust
  - With post-hoc attack counters, clearly labeled as leakage/unfair
- Fix the code comments so the feature-selection story is internally consistent.

## Medium Priority Methodology Choices

### 6. Why 80/20 Train/Test Split

Code locations:

- `scripts/analyze_vanet_nodes.py`: `test_size=0.2`.
- `scripts/preprocess.py`: `test_size=0.2`.

Current choice:

- 80% train, 20% test.
- Uses `stratify=y` to preserve class distribution.
- Uses `random_state=42` for reproducibility.

What the paper must justify:

- Why one holdout split is enough, or whether cross-validation is needed.
- Why stratification is required for an imbalanced security dataset.

Recommended evidence:

- Use stratified k-fold cross-validation and report mean plus standard deviation.
- Keep the 80/20 split as the dashboard/demo split, but use cross-validation in the paper.

### 7. Why StandardScaler Is Used

Code locations:

- `scripts/analyze_vanet_nodes.py`: scales all rich workflow features.
- `scripts/preprocess.py`: scales older 8-feature workflow.
- `scripts/combined_decision.py`: uses saved scaler before prediction.

Current choice:

- Uses `StandardScaler`.

What the paper must justify:

- Features have very different ranges, such as packets in thousands and trust scores from 0 to 1.
- Scaling is important for SVM, KNN, and Logistic Regression.
- Tree models are less sensitive to scaling, but scaling keeps the comparison consistent across algorithms.

Recommended evidence:

- Mention this as preprocessing rationale.
- Optional: add a with-scaling vs without-scaling comparison for SVM/KNN/LR.

### 8. Why These Nine Models Were Compared

Code locations:

- `scripts/analyze_vanet_nodes.py`: model dictionary.
- `scripts/train_model.py`: model dictionary.
- `dashboard/app.py`: `build_why_best` model descriptions.

Current choice:

- Random Forest, Gradient Boosting, Extra Trees, AdaBoost, Decision Tree, SVM, KNN, Logistic Regression, Gaussian Naive Bayes.

What the paper must justify:

- These represent different model families: ensemble trees, boosting, kernel method, instance-based, linear, interpretable tree, and probabilistic baseline.
- This gives a fair comparison between accuracy-oriented and edge-deployable lightweight methods.

Recommended evidence:

- Include a paragraph describing why each family was selected.
- Include model size and inference latency if claiming edge deployment.

### 9. Why The Hyperparameters Were Chosen

Code locations:

- `scripts/analyze_vanet_nodes.py`: hardcoded hyperparameters.
- `scripts/train_model.py`: Random Forest GridSearch for older workflow and hardcoded hyperparameters for others.

Current choices:

- Random Forest: `n_estimators=100`.
- Gradient Boosting: `n_estimators=100`, `max_depth=4`.
- Extra Trees: `n_estimators=100`.
- AdaBoost: `n_estimators=50`.
- Decision Tree: `max_depth=10`.
- SVM: RBF kernel with probability enabled.
- KNN: `n_neighbors=5`.
- Logistic Regression: `max_iter=1000`.
- Random state: `42`.

What the paper must justify:

- Whether these are default/baseline settings or tuned settings.
- If they are tuned, what search space was used.
- If not tuned, the paper should avoid overclaiming optimality.

Recommended evidence:

- Add grid/randomized search for the rich VANET workflow.
- Report the search space and best parameters.
- Or state these are standard baseline hyperparameters used for comparative evaluation.

## Lower Priority Demo/UI Choices

### 10. Why Dashboard Simulation Uses These Ranges

Code locations:

- `dashboard/app.py`: simulation probability and random ranges.

Current choices:

- `is_mal = random.random() < 0.45`.
- Malicious simulated speed range: 80 to 200.
- Normal simulated speed range: 5 to 60.
- Malicious trust factors: 0.0 to 0.35.
- Normal trust factors: 0.7 to 1.0.

What the paper must justify:

- These are demo assumptions, not experimental dataset generation.
- If used in experiments, they need a source or dataset-derived distribution.

Recommended evidence:

- Keep dashboard simulation out of core paper results.
- If mentioned, label it as a demonstration tool.

### 11. Why Final Decision Requires Both ML And Trust

Code locations:

- `dashboard/app.py`: `make_decision`.
- `scripts/combined_decision.py`: hybrid ML + trust simulation.

Current choice:

- `BLOCK` when trust is low and ML predicts malicious.
- `WARN` when either trust is low or ML predicts malicious.
- `ACCEPT` otherwise.

What the paper must justify:

- This is a conservative safety policy that reduces false blocking.
- Blocking requires agreement between two evidence sources.
- Warning allows lower-confidence anomaly handling.

Recommended evidence:

- Compare ML-only, trust-only, and hybrid decision policies.
- Report false positives and false negatives for each policy.

### 12. Why Dashboard Shows Baseline Numbers

Code locations:

- `dashboard/static/app.js`: hardcoded baseline performance chart values.

Current choice:

- The dashboard shows PKI only, ML only, trust only, and EdgeTrust-VANET baseline bars with hardcoded values.

What the paper must justify:

- If these numbers appear in the paper, they must be backed by experiments or citations.
- If they are only for demo visualization, label them as illustrative and do not use them as paper results.

Recommended evidence:

- Either remove hardcoded baseline claims from paper figures or run actual baseline experiments.

## Code Consistency Issues To Fix Before Paper Results

1. `dashboard/app.py` imports `calculate_trust_score` and `classify_vehicle`, but `scripts/trust_score.py` does not define those functions.
2. The dashboard evaluator builds 7 features, the older model expects 8, and the rich VANET model expects 14.
3. `dashboard/app.py` loads older filenames such as `Random_Forest.pkl`, while the scripts save names like `Random_Forest_GridSearch.pkl` or `vanet_Random_Forest.pkl`.
4. `scripts/analyze_vanet_nodes.py` says `n_features` comment is 18, but the actual saved value is 14.
5. The feature-selection comments in `scripts/analyze_vanet_nodes.py` contradict the actual feature list about whether trust scores are excluded or included.

## Suggested Paper Experiments

For a defensible paper, run these experiments and include the results:

1. Model comparison across all nine classifiers.
2. Alpha sensitivity analysis for `alpha = 0.3, 0.5, 0.7, 0.9`.
3. Trust threshold sensitivity for blocked/suspicious/trusted boundaries.
4. Feature ablation: mobility only, network only, trust only, combinations.
5. Hybrid policy ablation: ML only, trust only, ML OR trust, ML AND trust.
6. Cross-validation with mean and standard deviation.
7. Inference latency and model size for edge deployment claims.
8. Leakage demonstration: with and without post-hoc attack counters.

## Short Paper Wording You Can Reuse

Weighted F1 was selected as the primary ranking metric because the dataset is imbalanced, with malicious vehicles forming a minority class. Accuracy alone can overstate performance by rewarding majority-class predictions, while F1 captures the tradeoff between false alarms and missed malicious vehicles.

The trust update parameter alpha controls the stability-responsiveness tradeoff. A larger alpha preserves historical behavior and avoids overreacting to isolated noisy observations, while a smaller alpha reacts faster to new malicious evidence. Therefore, alpha should be justified through sensitivity analysis rather than treated as a fixed arbitrary constant.

Post-hoc attack counters are excluded from fair model training because they represent information available only after an attack has already been identified. Including them would create data leakage and inflate evaluation performance beyond what an RSU could achieve during real-time deployment.
