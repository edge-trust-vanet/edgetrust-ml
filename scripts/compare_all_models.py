"""
╔══════════════════════════════════════════════════════════════════╗
║         compare_all_models.py                                    ║
║         EdgeTrust-VANET — Full Benchmark & Comparison Script     ║
╠══════════════════════════════════════════════════════════════════╣
║  Compares all 9 existing traditional baseline models with 8 new  ║
║  and unique ML architectures (XGBoost, LightGBM, CatBoost,       ║
║  HistGradientBoosting, MLP Neural Network, LDA, Stacking         ║
║  Ensemble, Voting Ensemble) across multiple dimensions:          ║
║    1. Classification Accuracy, F1 (Weighted & Macro), Prec, Rec  ║
║    2. Security Critical Errors (False Positives, False Negatives)║
║    3. ROC-AUC & Discriminative Power                             ║
║    4. Edge Inference Latency (µs/sample) for RSU units           ║
║    5. Training Time & Memory Footprint                           ║
║    6. Monte-Carlo Stability (20 Stratified Splits)               ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import time
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, roc_auc_score
)

# Existing 9 Classifiers
from sklearn.ensemble import (
    RandomForestClassifier, GradientBoostingClassifier,
    ExtraTreesClassifier, AdaBoostClassifier,
    HistGradientBoostingClassifier, StackingClassifier, VotingClassifier
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB

# New & Unique Classifiers
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

BASE = Path(__file__).resolve().parents[1]
DATA_PATH = BASE / "data" / "vanet_malicious_nodes.csv"
RESULTS_DIR = BASE / "results"
RESULTS_DIR.mkdir(exist_ok=True)

FEATURES = [
    'position_x', 'position_y', 'speed', 'direction', 'acceleration',
    'packet_sent', 'packet_received', 'packet_drop_ratio', 'latency',
    'message_retransmission_count', 'signal_strength',
    'trust_score', 'neighbor_trust_score_avg', 'historical_trust_score'
]
TARGET = 'is_malicious'


def get_model_suite():
    return {
        # ── Existing Traditional ML Models ──
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42),
        'Extra Trees': ExtraTreesClassifier(n_estimators=100, random_state=42),
        'AdaBoost': AdaBoostClassifier(n_estimators=50, random_state=42),
        'Decision Tree': DecisionTreeClassifier(max_depth=10, random_state=42),
        'SVM (RBF)': SVC(kernel='rbf', probability=True, random_state=42),
        'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Gaussian Naive Bayes': GaussianNB(),

        # ── New & Unique ML Models ──
        'XGBoost': xgb.XGBClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.08, subsample=0.85,
            colsample_bytree=0.85, random_state=42, eval_metric='logloss'
        ),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=150, max_depth=5, learning_rate=0.08, num_leaves=31,
            subsample=0.85, random_state=42, verbose=-1
        ),
        'CatBoost': cb.CatBoostClassifier(
            iterations=200, depth=5, learning_rate=0.08, l2_leaf_reg=3,
            random_seed=42, verbose=0
        ),
        'Hist Gradient Boosting': HistGradientBoostingClassifier(
            max_iter=150, max_depth=5, learning_rate=0.08, random_state=42
        ),
        'MLP Neural Network': MLPClassifier(
            hidden_layer_sizes=(128, 64), max_iter=600, activation='relu',
            solver='adam', alpha=1e-4, random_state=42
        ),
        'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
        'Stacking Ensemble': StackingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=80, random_state=42)),
                ('xgb', xgb.XGBClassifier(n_estimators=80, max_depth=4, learning_rate=0.08, random_state=42, eval_metric='logloss')),
                ('cb', cb.CatBoostClassifier(iterations=80, depth=4, learning_rate=0.08, random_seed=42, verbose=0)),
                ('ada', AdaBoostClassifier(n_estimators=50, random_state=42)),
            ],
            final_estimator=LogisticRegression(random_state=42),
            cv=5
        ),
        'Voting Ensemble': VotingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=100, random_state=42)),
                ('ada', AdaBoostClassifier(n_estimators=50, random_state=42)),
                ('xgb', xgb.XGBClassifier(n_estimators=100, max_depth=4, learning_rate=0.08, random_state=42, eval_metric='logloss')),
                ('cb', cb.CatBoostClassifier(iterations=100, depth=4, learning_rate=0.08, random_seed=42, verbose=0)),
            ],
            voting='soft'
        )
    }


def evaluate_single_split(models, x_train, x_test, y_train, y_test):
    results = {}
    for name, model in models.items():
        t0 = time.perf_counter()
        model.fit(x_train, y_train)
        fit_time = time.perf_counter() - t0

        t0 = time.perf_counter()
        y_pred = model.predict(x_test)
        pred_time = time.perf_counter() - t0

        try:
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(x_test)[:, 1]
                auc = roc_auc_score(y_test, y_prob)
            elif hasattr(model, "decision_function"):
                y_prob = model.decision_function(x_test)
                auc = roc_auc_score(y_test, y_prob)
            else:
                auc = 0.0
        except Exception:
            auc = 0.0

        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()

        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred) * 100,
            'weighted_f1': f1_score(y_test, y_pred, average='weighted') * 100,
            'macro_f1': f1_score(y_test, y_pred, average='macro') * 100,
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0) * 100,
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0) * 100,
            'roc_auc': auc * 100 if auc > 0 else 0.0,
            'fp': int(fp),
            'fn': int(fn),
            'tp': int(tp),
            'tn': int(tn),
            'fit_seconds': fit_time,
            'predict_micros_per_sample': (pred_time / len(y_test)) * 1_000_000,
            'model_size_kb': len(pickle.dumps(model)) / 1024,
            'confusion_matrix': cm.tolist()
        }
    return results


def run_benchmark():
    print("=" * 75)
    print("  EdgeTrust-VANET: Comprehensive ML Algorithm Benchmark")
    print("  (9 Existing Models + 8 New & Unique Models)")
    print("=" * 75)

    df = pd.read_csv(DATA_PATH).dropna(subset=FEATURES + [TARGET])
    X = df[FEATURES].values
    y = df[TARGET].values

    # 1. Exact Seed-42 Holdout Evaluation
    X_train_42, X_test_42, y_train_42, y_test_42 = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler_42 = StandardScaler()
    X_train_42 = scaler_42.fit_transform(X_train_42)
    X_test_42 = scaler_42.transform(X_test_42)

    print("\n▶ Running Exact Seed-42 Holdout...")
    models_42 = get_model_suite()
    holdout_results = evaluate_single_split(models_42, X_train_42, X_test_42, y_train_42, y_test_42)

    # 2. 20-Split Monte Carlo Cross-Validation
    print("\n▶ Running 20-Fold Repeated Stratified Evaluation...")
    all_runs = []
    for seed in range(20):
        print(f"  Split {seed+1:2d}/20 ...", end="\r", flush=True)
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=0.2, random_state=seed, stratify=y
        )
        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr)
        X_te = sc.transform(X_te)
        split_models = get_model_suite()
        split_res = evaluate_single_split(split_models, X_tr, X_te, y_tr, y_te)
        for name, m in split_res.items():
            row = {'seed': seed, 'model': name, **m}
            all_runs.append(row)

    print("\n✔ Completed 20 splits for all 17 models.")
    runs_df = pd.DataFrame(all_runs)

    # Aggregate 20-split metrics
    summary = {}
    for name, group in runs_df.groupby('model'):
        summary[name] = {
            'accuracy_mean': round(float(group['accuracy'].mean()), 2),
            'accuracy_std': round(float(group['accuracy'].std()), 2),
            'weighted_f1_mean': round(float(group['weighted_f1'].mean()), 2),
            'weighted_f1_std': round(float(group['weighted_f1'].std()), 2),
            'macro_f1_mean': round(float(group['macro_f1'].mean()), 2),
            'precision_mean': round(float(group['precision'].mean()), 2),
            'recall_mean': round(float(group['recall'].mean()), 2),
            'roc_auc_mean': round(float(group['roc_auc'].mean()), 2),
            'fp_mean': round(float(group['fp'].mean()), 1),
            'fn_mean': round(float(group['fn'].mean()), 1),
            'fit_ms_mean': round(float(group['fit_seconds'].mean() * 1000), 1),
            'predict_micros_mean': round(float(group['predict_micros_per_sample'].mean()), 2),
            'model_size_kb_mean': round(float(group['model_size_kb'].mean()), 1),
        }

    # Save comparison data
    output_data = {
        'seed_42_holdout': holdout_results,
        'repeated_20_splits_summary': summary,
        'model_categories': {
            'Traditional Tree & Ensemble': ['Random Forest', 'Gradient Boosting', 'Extra Trees', 'AdaBoost', 'Decision Tree'],
            'Traditional Non-Tree': ['SVM (RBF)', 'K-Nearest Neighbors', 'Logistic Regression', 'Gaussian Naive Bayes'],
            'Modern Regularized GBDT': ['XGBoost', 'LightGBM', 'CatBoost', 'Hist Gradient Boosting'],
            'Deep Learning & Generative': ['MLP Neural Network', 'Linear Discriminant Analysis'],
            'Advanced Heterogeneous Meta-Ensembles': ['Stacking Ensemble', 'Voting Ensemble']
        }
    }

    report_path = RESULTS_DIR / "model_comparison_report.json"
    with open(report_path, "w") as f:
        json.dump(output_data, f, indent=2)

    print(f"\n✅ Benchmark report saved to: {report_path}")

    # Print Formatted Comparison Table
    print("\n" + "=" * 115)
    print(f"{'Model Name':30s} | {'Category':24s} | {'Holdout F1':11s} | {'20-Split F1 (±std)':18s} | {'Latency':12s} | {'Model Size':10s}")
    print("-" * 115)

    # Sort models by 20-split F1 mean descending
    sorted_models = sorted(summary.keys(), key=lambda k: summary[k]['weighted_f1_mean'], reverse=True)

    def get_category(mname):
        for cat, mlist in output_data['model_categories'].items():
            if mname in mlist:
                return cat
        return 'Other'

    for name in sorted_models:
        s = summary[name]
        h = holdout_results[name]
        cat = get_category(name)
        print(f"{name:30s} | {cat:24s} | {h['weighted_f1']:6.2f}%    | {s['weighted_f1_mean']:5.2f}% (±{s['weighted_f1_std']:4.2f})  | {s['predict_micros_mean']:6.1f} µs/samp | {s['model_size_kb_mean']:7.1f} KB")

    print("=" * 115)


if __name__ == "__main__":
    run_benchmark()
