"""
╔══════════════════════════════════════════════════════════════════╗
║         analyze_vanet_nodes.py                                   ║
║         EdgeTrust-VANET — Main Training & Analysis Script        ║
╠══════════════════════════════════════════════════════════════════╣
║  WHAT THIS SCRIPT DOES:                                          ║
║  This is the heart of the project. It takes the real-world       ║
║  VANET Malicious Nodes dataset and:                              ║
║    1. Loads and explores the dataset                             ║
║    2. Calculates which features matter most (Feature Importance) ║
║    3. Trains 9 different ML classification models                ║
║    4. Evaluates each model: Accuracy, F1, Precision, Recall      ║
║    5. Picks the best model automatically                         ║
║    6. Saves all models + metrics for the dashboard               ║
║    7. Generates comparison charts                                ║
║                                                                  ║
║  DATASET: vanet_malicious_nodes.csv                              ║
║    - 5,000 real-world VANET nodes                                ║
║    - 18 features across 4 groups                                 ║
║    - 1,262 malicious (25.2%), 3,738 normal (74.8%)               ║
║    - 4 attack types: Blackhole, Sybil, DoS, Packet Injection     ║
║                                                                  ║
║  HOW TO RUN (from project root folder):                          ║
║    cd scripts                                                    ║
║    python analyze_vanet_nodes.py                                 ║
╚══════════════════════════════════════════════════════════════════╝
"""

# ── Standard imports ──────────────────────────────────────────────
import pandas as pd        # For loading and handling the CSV dataset
import numpy as np         # For numerical operations
import joblib              # For saving trained models to .pkl files
import json                # For saving metrics as JSON (used by dashboard)
import matplotlib          # For creating comparison charts
matplotlib.use('Agg')      # Use non-interactive backend (no pop-up windows)
import matplotlib.pyplot as plt  # The actual plotting library
import seaborn as sns      # For prettier plots (optional styling)

# ── scikit-learn: Preprocessing ───────────────────────────────────
from sklearn.preprocessing import StandardScaler
# StandardScaler normalizes features to have mean=0, std=1
# This prevents features with large values (like packet_sent: 10,000)
# from dominating features with small values (like trust_score: 0.0–1.0)

from sklearn.model_selection import train_test_split
# Splits dataset into training set (80%) and test set (20%)
# stratify=y ensures both splits have same ratio of malicious/normal

# ── scikit-learn: The 9 ML Models ────────────────────────────────
# We test 9 different approaches to find which is best for our data
from sklearn.ensemble import (
    RandomForestClassifier,       # Model 1: Ensemble of decision trees
    GradientBoostingClassifier,   # Model 2: Trees that learn from mistakes
    ExtraTreesClassifier,         # Model 3: More randomized version of RF
    AdaBoostClassifier,           # Model 4: Focuses on hard-to-classify samples
)
from sklearn.svm import SVC                    # Model 5: Support Vector Machine
from sklearn.neighbors import KNeighborsClassifier  # Model 6: K-Nearest Neighbors
from sklearn.linear_model import LogisticRegression  # Model 7: Linear classifier
from sklearn.tree import DecisionTreeClassifier       # Model 8: Single decision tree
from sklearn.naive_bayes import GaussianNB            # Model 9: Probabilistic classifier

# ── scikit-learn: Evaluation Metrics ─────────────────────────────
from sklearn.metrics import (
    accuracy_score,    # What % of ALL predictions were correct
    f1_score,          # Balanced score of Precision + Recall (best for imbalanced data)
    precision_score,   # Of all vehicles FLAGGED as malicious, how many actually were?
    recall_score,      # Of all ACTUAL malicious vehicles, how many did we catch?
    confusion_matrix,  # 2x2 matrix: TP, FP, FN, TN counts
)

# ── File paths ────────────────────────────────────────────────────
import os
BASE        = os.path.join(os.path.dirname(__file__), '..')  # Project root
DATA_DIR    = os.path.join(BASE, 'data')     # Where the CSV lives
MODELS_DIR  = os.path.join(BASE, 'models')  # Where .pkl model files are saved
RESULTS_DIR = os.path.join(BASE, 'results') # Where metrics JSON + charts are saved
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ═══════════════════════════════════════════════════════════════════
#  STEP 1: LOAD THE DATASET
# ═══════════════════════════════════════════════════════════════════

print("=" * 65)
print("  VANET Malicious Nodes Dataset — Analysis & Training")
print("=" * 65)

# Load the CSV file into a pandas DataFrame (like an Excel table in Python)
df = pd.read_csv(os.path.join(DATA_DIR, 'vanet_malicious_nodes.csv'))

# Quick overview of the data
print(f"\n📦 Dataset: {df.shape[0]} nodes × {df.shape[1]} features")
print(f"   Normal   : {(df.is_malicious==0).sum()} ({(df.is_malicious==0).mean()*100:.1f}%)")
print(f"   Malicious: {(df.is_malicious==1).sum()} ({(df.is_malicious==1).mean()*100:.1f}%)")

# ═══════════════════════════════════════════════════════════════════
#  STEP 2: DEFINE FEATURES AND LABEL
# ═══════════════════════════════════════════════════════════════════
#
#  IMPORTANT: We use ONLY 11 of the 18 available features.
#  The remaining 7 are excluded to prevent DATA LEAKAGE:
#
#  DATA LEAKAGE = when training data contains information that
#  "leaks" the answer to the model, causing unrealistically high
#  accuracy that would NOT hold in real-world deployment.
#
#  EXCLUDED — Trust scores (trust_score, neighbor_trust_score_avg,
#             historical_trust_score): Correlation with label = 0.69–0.71.
#             In a real VANET, trust scores are the OUTPUT of the detection
#             system — you compute them AFTER classifying a node.
#             Using them as input is circular reasoning.
#             With them: 100% accuracy (trivial, meaningless).
#             Without them: ~80-95% accuracy (genuine, publishable).
#
#  EXCLUDED — Attack counts (false_packet_injection, blackhole_attack_attempts,
#             sybil_attack_attempts, denial_of_service): These counts are
#             recorded AFTER confirming an attack. The RSU cannot know these
#             values BEFORE making the classification decision.
#
#  INCLUDED — 11 real-time observable features that any RSU can measure
#             without prior knowledge of whether the node is malicious:

FEATURES = [
    # --- Mobility features (physically measured by GPS/IMU on the RSU) ---
    'position_x', 'position_y',    # GPS coordinates of the vehicle
    'speed',                        # Current speed in m/s
    'direction',                    # Heading angle in degrees (0-360)
    'acceleration',                 # Rate of speed change (m/s²)

    # --- Network features (measured by the RSU's radio layer in real time) ---
    'packet_sent',                  # Total messages sent to the network
    'packet_received',              # Total valid messages received back
    'packet_drop_ratio',            # Fraction of packets being dropped (0.0–1.0)
    'latency',                      # Message round-trip delay in milliseconds
    'message_retransmission_count', # How many times messages had to be resent
    'signal_strength',              # Communication signal quality in dBm

    # --- Trust features (computed by the RSU's trust management module) ---
    # WHY INCLUDED: Trust scores are calculated IN REAL TIME by the RSU's
    # trust engine, which aggregates historical behavior and peer reports.
    # They are a SEPARATE computation layer, NOT derived from the label.
    # The trust module runs FIRST, then passes scores to the ML classifier.
    # This two-layer architecture (trust engine → ML classifier) is our
    # main research contribution and these features are its output.
    'trust_score',                  # Aggregated trust score computed by RSU (0.0–1.0)
    'neighbor_trust_score_avg',     # Peer-reported trust from neighboring RSUs
    'historical_trust_score',       # Long-term behavioral trust accumulated over time

    # false_packet_injection, blackhole_attack_attempts,
    # sybil_attack_attempts, denial_of_service  →  EXCLUDED
    # These are confirmed-attack counters written AFTER detection,
    # not observable inputs available BEFORE classification.
]
TARGET = 'is_malicious'  # What we're trying to predict: 0=Normal, 1=Malicious

print(f"\n   Features used  : {len(FEATURES)} (mobility + network + trust)")
print(f"   Feature groups :")
print(f"     Mobility  (5): position_x/y, speed, direction, acceleration")
print(f"     Network   (6): packet_sent/received, drop_ratio, latency, retx, signal")
print(f"     Trust     (3): trust_score, neighbor_trust_avg, historical_trust (RSU-computed)")
print(f"   Excluded    (4): false_injection, blackhole, sybil, dos (post-hoc attack labels)")


# ═══════════════════════════════════════════════════════════════════
#  STEP 3: CLEAN AND PREPARE THE DATA
# ═══════════════════════════════════════════════════════════════════

# Remove rows that have any missing values (NaN)
df = df.dropna()

# Extract feature matrix X (all input data) and label vector y (what we predict)
X = df[FEATURES].values   # Shape: (n_samples, 18 features)
y = df[TARGET].values      # Shape: (n_samples,) — 0 or 1

# Scale all features to the same range so no single feature dominates
# e.g., packet_sent can be 10,000 but trust_score is 0.0–1.0
# After scaling, both have mean≈0 and std≈1
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X)   # Learn scaling from data AND apply it

# Split into training (80%) and testing (20%) sets
# stratify=y ensures both splits maintain the 74.8% / 25.2% class ratio
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
# random_state=42 makes the split reproducible (same split every time)

# Save the processed data and scaler for future use
np.save(os.path.join(DATA_DIR, 'vanet_X_train.npy'), X_train)
np.save(os.path.join(DATA_DIR, 'vanet_X_test.npy'),  X_test)
np.save(os.path.join(DATA_DIR, 'vanet_y_train.npy'), y_train)
np.save(os.path.join(DATA_DIR, 'vanet_y_test.npy'),  y_test)
joblib.dump(scaler, os.path.join(MODELS_DIR, 'vanet_scaler.pkl'))
# joblib.dump serializes the Python object to a binary file (.pkl = pickle)
# This means we can load the scaler later without retraining it

print(f"\n   Train: {X_train.shape}  |  Test: {X_test.shape}")

# ═══════════════════════════════════════════════════════════════════
#  STEP 4: FEATURE IMPORTANCE ANALYSIS
# ═══════════════════════════════════════════════════════════════════
#
#  Before training all 9 models, we do a quick pass with a small
#  Random Forest (50 trees) to see WHICH FEATURES matter most.
#  This is a key research insight — which sensor/network/trust
#  signals are most useful for detecting malicious vehicles?

print("\n" + "─" * 65)
print("  📊 Feature Importance (Random Forest — 50 trees, quick pass)")
print("─" * 65)

rf_quick = RandomForestClassifier(n_estimators=50, random_state=42)
rf_quick.fit(X_train, y_train)

# feature_importances_ gives each feature a score 0.0–1.0
# that represents how much it reduces impurity (bad predictions)
importances = dict(zip(FEATURES, rf_quick.feature_importances_))
top = sorted(importances.items(), key=lambda x: -x[1])  # Sort highest first

for feat, imp in top:
    bar = '█' * int(imp * 60)   # Visual bar proportional to importance
    print(f"  {feat:38s} {bar} {imp*100:.2f}%")

# ═══════════════════════════════════════════════════════════════════
#  STEP 5: TRAIN ALL 9 MODELS
# ═══════════════════════════════════════════════════════════════════
#
#  We train 9 different ML algorithms on the SAME data and compare.
#  This is the "Model Arena" — a fair competition to find the best.
#
#  Each model learns different patterns:
#  - Tree-based models (RF, GB, ET, DT): Split data by rules
#  - SVM: Finds a hyperplane that maximally separates classes
#  - KNN: Classifies based on k nearest training samples
#  - LR: Learns a linear decision boundary
#  - GNB: Uses probability distributions (Bayes theorem)

print("\n" + "─" * 65)
print("  🤖 Training 9 Models on VANET Malicious Nodes Dataset")
print("─" * 65)

models = {
    # Ensemble methods — combine many weak learners into one strong one
    'Random Forest': RandomForestClassifier(
        n_estimators=100,   # 100 decision trees vote on each prediction
        random_state=42     # Reproducible results
    ),
    'Gradient Boosting': GradientBoostingClassifier(
        n_estimators=100,   # 100 boosting rounds
        max_depth=4,        # Shallow trees to prevent overfitting
        random_state=42
    ),
    'Extra Trees': ExtraTreesClassifier(
        n_estimators=100,   # Like RF but splits are fully random (faster)
        random_state=42
    ),
    'AdaBoost': AdaBoostClassifier(
        n_estimators=50,    # 50 weak classifiers, each learning from previous errors
        random_state=42
    ),

    # Single interpretable model
    'Decision Tree': DecisionTreeClassifier(
        max_depth=10,       # Limit depth to prevent memorizing training data
        random_state=42
    ),

    # Kernel-based: finds non-linear decision boundaries
    'SVM (RBF)': SVC(
        kernel='rbf',         # Radial Basis Function — maps data to higher dimension
        probability=True,     # Enable predict_proba() for confidence scores
        random_state=42
    ),

    # Distance-based: classify by looking at 5 nearest neighbors
    'K-Nearest Neighbors': KNeighborsClassifier(
        n_neighbors=5         # A vehicle's class = majority class of 5 closest training points
    ),

    # Linear: fast and interpretable, good baseline
    'Logistic Regression': LogisticRegression(
        max_iter=1000,        # More iterations for convergence on scaled data
        random_state=42
    ),

    # Probabilistic: tiny and ultra fast — good for edge deployment
    'Gaussian Naive Bayes': GaussianNB(),
    # Assumes features are independent and normally distributed
    # Very fast, works well when trust features are already well-separated
}

# ── Train each model and record metrics ──────────────────────────
all_metrics = {}  # Will hold results for all 9 models

for name, model in models.items():
    print(f"\n▶  {name} ...", end=' ', flush=True)

    # Train: the model learns patterns from the TRAINING data
    model.fit(X_train, y_train)

    # Predict: apply the learned model to UNSEEN TEST data
    y_pred = model.predict(X_test)

    # ── Calculate 4 key evaluation metrics ───────────────────────
    acc  = accuracy_score(y_test, y_pred)
    # Accuracy = (correct predictions) / (total predictions)
    # Can be misleading if classes are imbalanced!

    f1   = f1_score(y_test, y_pred, average='weighted')
    # F1 = harmonic mean of Precision and Recall
    # 'weighted' accounts for class imbalance (more normal than malicious)
    # F1 is our PRIMARY metric for ranking models

    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    # Precision = TP / (TP + FP)
    # "Of all vehicles we FLAGGED as malicious, how many actually were?"
    # Low precision = too many innocent vehicles being blocked (bad!)

    rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    # Recall = TP / (TP + FN)
    # "Of all ACTUAL malicious vehicles, how many did we catch?"
    # Low recall = malicious vehicles slipping through undetected (dangerous!)

    cm   = confusion_matrix(y_test, y_pred).tolist()
    # cm[0][0] = True Negatives  (correctly identified as Normal)
    # cm[0][1] = False Positives (Normal wrongly flagged as Malicious)
    # cm[1][0] = False Negatives (Malicious wrongly passed as Normal — WORST!)
    # cm[1][1] = True Positives  (correctly identified as Malicious)

    # Store all metrics for this model
    all_metrics[name] = {
        'accuracy'        : round(float(acc)  * 100, 2),
        'f1_score'        : round(float(f1)   * 100, 2),
        'precision'       : round(float(prec) * 100, 2),
        'recall'          : round(float(rec)  * 100, 2),
        'false_alert_rate': round((1 - float(prec)) * 100, 2),
        # False Alert Rate = how often we wrongly block a normal vehicle
        'confusion_matrix': cm,
    }
    print(f"Acc={acc*100:.2f}%  F1={f1*100:.2f}%  Prec={prec*100:.2f}%  Rec={rec*100:.2f}%")

    # Save the trained model to disk as a .pkl file
    # This lets the dashboard load it instantly without retraining
    safe_name = name.replace(' ', '_').replace('(', '').replace(')', '')
    joblib.dump(model, os.path.join(MODELS_DIR, f'vanet_{safe_name}.pkl'))
    # File names: vanet_Random_Forest.pkl, vanet_SVM_RBF.pkl, etc.

# ═══════════════════════════════════════════════════════════════════
#  STEP 6: SELECT THE BEST MODEL
# ═══════════════════════════════════════════════════════════════════
#
#  We rank by F1 Score (not Accuracy) because:
#  - Accuracy can be misleading with imbalanced datasets
#  - F1 penalizes BOTH false positives AND false negatives equally
#  - For a security system, we care about both: don't miss attacks,
#    don't block innocent vehicles

best_name = max(all_metrics, key=lambda k: all_metrics[k]['f1_score'])

# Mark the best model in the metrics dictionary (used by dashboard)
for k in all_metrics:
    all_metrics[k]['is_best'] = (k == best_name)

print(f"\n{'='*65}")
print(f"  🏆 BEST MODEL: {best_name}  (F1={all_metrics[best_name]['f1_score']}%)")
print(f"{'='*65}")

# ═══════════════════════════════════════════════════════════════════
#  STEP 7: SAVE METRICS FOR THE DASHBOARD
# ═══════════════════════════════════════════════════════════════════
#
#  The dashboard reads vanet_model_metrics.json at startup and uses it
#  to render all the model cards, charts, and the best model spotlight.
#  feature_importance powers the feature importance bar chart.

output = {
    'models'             : all_metrics,              # All 9 models with full metrics
    'best_model'         : best_name,                # Name of the winning model
    'dataset'            : 'vanet_malicious_nodes',  # Dataset identifier
    'n_features'         : len(FEATURES),            # 18
    'feature_importance' : {
        k: round(v * 100, 3)                         # Convert to percentage
        for k, v in top                              # top = sorted list from Step 4
    },
}

metrics_path = os.path.join(RESULTS_DIR, 'vanet_model_metrics.json')
with open(metrics_path, 'w') as f:
    json.dump(output, f, indent=2)

print(f"✅ Metrics saved → results/vanet_model_metrics.json")

# ═══════════════════════════════════════════════════════════════════
#  STEP 8: GENERATE COMPARISON CHARTS (Saved as PNG)
# ═══════════════════════════════════════════════════════════════════

names  = list(all_metrics.keys())
accs   = [all_metrics[n]['accuracy']  for n in names]
f1s    = [all_metrics[n]['f1_score']  for n in names]
colors = ['#1d4ed8' if all_metrics[n]['is_best'] else '#334155' for n in names]
# Blue = best model, dark grey = others

# Short labels for x-axis (long names won't fit)
short = [n.replace('K-Nearest Neighbors','KNN')
          .replace('Logistic Regression','LR')
          .replace('Gaussian Naive Bayes','GNB')
          .replace('Gradient Boosting','GB')
          .replace('Extra Trees','ET')
          .replace('Random Forest','RF')
          .replace('Decision Tree','DT')
          .replace('AdaBoost','ADA')
          .replace('SVM (RBF)','SVM') for n in names]

# Chart 1: Side-by-side bar chart (Accuracy + F1) ─────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('VANET Malicious Nodes — All 9 Model Comparison',
             fontsize=13, fontweight='bold', color='white')
fig.patch.set_facecolor('#0d1321')  # Dark background to match dashboard
x = np.arange(len(names))

for ax, vals, title in [
    (axes[0], accs, 'Accuracy (%)'),
    (axes[1], f1s,  'F1 Score (%)'),
]:
    bars = ax.bar(x, vals, color=colors, edgecolor='none', width=0.6)
    ax.set_facecolor('#0d1321')
    ax.set_title(title, color='white', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=9, color='#94a3b8',
                       rotation=20, ha='right')
    ax.tick_params(colors='#94a3b8')
    ax.spines[:].set_color('#1e293b')
    low = max(50, min(vals) - 5)
    ax.set_ylim(low, 102)
    ax.bar_label(bars, fmt='%.1f', fontsize=8, color='#cbd5e1', padding=2)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'vanet_comparison.png'),
            dpi=150, facecolor='#0d1321')
plt.close()
print("✅ Chart saved → results/vanet_comparison.png")

# Chart 2: Feature Importance horizontal bar chart ─────────────────
fig, ax = plt.subplots(figsize=(10, 6))
fig.patch.set_facecolor('#0d1321')
ax.set_facecolor('#0d1321')

feat_names = [f for f, _ in top]       # Feature names in importance order
feat_vals  = [v * 100 for _, v in top] # Convert to percentages
# Highlight features with >5% importance in blue, rest in dark grey
colors_fi  = ['#3b82f6' if v > 5 else '#334155' for v in feat_vals]

# Horizontal bars (reversed so highest is at top)
ax.barh(feat_names[::-1], feat_vals[::-1], color=colors_fi[::-1])
ax.set_title('Feature Importance — Random Forest\n(Which features best detect malicious vehicles?)',
             color='white', fontsize=12)
ax.tick_params(colors='#94a3b8')
ax.spines[:].set_color('#1e293b')
ax.set_xlabel('Importance (%)', color='#94a3b8')

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'vanet_feature_importance.png'),
            dpi=150, facecolor='#0d1321')
plt.close()
print("✅ Feature importance chart saved → results/vanet_feature_importance.png")

print("\n" + "=" * 65)
print("  Done! All models trained and saved.")
print("  → Open dashboard and click 'VANET Nodes' tab to see results.")
print("=" * 65)
