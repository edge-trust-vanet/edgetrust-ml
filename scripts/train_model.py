"""
train_model.py
─────────────────────────────────────────────────────────────
Trains 9 ML classifiers on the VeReMi-style dataset and saves
all models + a metrics JSON for the dashboard.

Prerequisite:
  Run preprocess.py first to generate X_train.npy etc.

Usage (from project root):
  python scripts/train_model.py
─────────────────────────────────────────────────────────────
"""
import numpy as np, json, os, joblib
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                              ExtraTreesClassifier, AdaBoostClassifier)
from sklearn.svm      import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree     import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics  import (accuracy_score, f1_score, precision_score,
                               recall_score, confusion_matrix)

BASE        = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR    = os.path.join(BASE, 'data')
MODELS_DIR  = os.path.join(BASE, 'models')
RESULTS_DIR = os.path.join(BASE, 'results')
os.makedirs(MODELS_DIR,  exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

print("=" * 65)
print("  VeReMi — Training 9 Models")
print("=" * 65)

# ── Load preprocessed data ────────────────────────────────
for f in ['X_train.npy', 'X_test.npy', 'y_train.npy', 'y_test.npy']:
    if not os.path.exists(os.path.join(DATA_DIR, f)):
        print(f"\n❌ Missing {f}. Run: python scripts/preprocess.py first")
        exit(1)

X_train = np.load(os.path.join(DATA_DIR, 'X_train.npy'))
X_test  = np.load(os.path.join(DATA_DIR, 'X_test.npy'))
y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
y_test  = np.load(os.path.join(DATA_DIR, 'y_test.npy'))
print(f"\n   Train: {X_train.shape}  |  Test: {X_test.shape}")

# ── Model suite ───────────────────────────────────────────
models = {
    'Random Forest'        : RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting'    : GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42),
    'Extra Trees'          : ExtraTreesClassifier(n_estimators=100, random_state=42),
    'AdaBoost'             : AdaBoostClassifier(n_estimators=50, random_state=42),
    'Decision Tree'        : DecisionTreeClassifier(max_depth=10, random_state=42),
    'SVM (RBF)'            : SVC(kernel='rbf', probability=True, random_state=42),
    'K-Nearest Neighbors'  : KNeighborsClassifier(n_neighbors=5),
    'Logistic Regression'  : LogisticRegression(max_iter=1000, random_state=42),
    'Gaussian Naive Bayes' : GaussianNB(),
}

print("\n" + "─" * 65)
all_metrics = {}

for name, model in models.items():
    print(f"▶  {name} ...", end=' ', flush=True)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc  = accuracy_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred, average='weighted')
    prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
    rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
    cm   = confusion_matrix(y_test, y_pred).tolist()

    all_metrics[name] = {
        'accuracy'        : round(float(acc)  * 100, 2),
        'f1_score'        : round(float(f1)   * 100, 2),
        'precision'       : round(float(prec) * 100, 2),
        'recall'          : round(float(rec)  * 100, 2),
        'false_alert_rate': round((1 - float(prec)) * 100, 2),
        'confusion_matrix': cm,
    }
    print(f"Acc={acc*100:.2f}%  F1={f1*100:.2f}%  Prec={prec*100:.2f}%  Rec={rec*100:.2f}%")

    safe = name.replace(' ', '_').replace('(', '').replace(')', '')
    joblib.dump(model, os.path.join(MODELS_DIR, f'{safe}.pkl'))

# ── Best model ────────────────────────────────────────────
best_name = max(all_metrics, key=lambda k: all_metrics[k]['f1_score'])
for k in all_metrics:
    all_metrics[k]['is_best'] = (k == best_name)

print(f"\n{'='*65}")
print(f"  🏆 BEST MODEL: {best_name}  (F1={all_metrics[best_name]['f1_score']}%)")
print(f"{'='*65}")

# ── Save metrics ──────────────────────────────────────────
out = {'models': all_metrics, 'best_model': best_name, 'dataset': 'veremi_style'}
with open(os.path.join(RESULTS_DIR, 'model_metrics.json'), 'w') as f:
    json.dump(out, f, indent=2)

print("✅ Metrics saved → results/model_metrics.json")
print("✅ All models saved → models/*.pkl")
print("\nRefresh the dashboard VeReMi Arena tab to see updated results.")
