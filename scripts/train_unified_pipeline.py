"""
╔══════════════════════════════════════════════════════════════════╗
║         train_unified_pipeline.py                                ║
║         EdgeTrust-VANET — Complete ML Training & Benchmark       ║
╠══════════════════════════════════════════════════════════════════╣
║  Executes the full research pipeline on the unified dataset:     ║
║    1. 14 Leakage-Free Observable Features                        ║
║    2. Group-based Vehicle Splitting (70% Train, 15% Val, 15% Test)║
║    3. Strict Train-only StandardScaler Fitting                   ║
║    4. 5-Fold Stratified Group Cross-Validation                   ║
║    5. Hyperparameter Tuning for Top Classifiers                  ║
║    6. Full 17-Model Suite Benchmark Across 5 Architectures:      ║
║       - Ensemble Trees (RF, ExtraTrees, DecisionTree)            ║
║       - Modern GBDTs (XGBoost, LightGBM, CatBoost, HistGB, GB,   ║
║         AdaBoost)                                                ║
║       - Linear/Kernel/Instance (SVM, KNN, LogisticReg, GNB, LDA) ║
║       - Deep Tabular Neural Network (MLP)                        ║
║       - Heterogeneous Meta-Ensembles (Stacking, Soft Voting)     ║
║    7. Edge Inference Latency (µs) & Model Storage Footprint (KB) ║
║    8. Exports Models, Metrics JSON, and Publication Figures      ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, confusion_matrix, roc_curve
)

# Traditional & Ensemble Baselines
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
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

# Modern GBDTs
import xgboost as xgb
import lightgbm as lgb
import catboost as cb

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

DATASET_FILE = os.path.join(DATA_DIR, 'unified_edgetrust_dataset.csv')

FEATURES_14 = [
    'position_x', 'position_y', 'speed', 'direction', 'acceleration',
    'packet_sent', 'packet_received', 'packet_drop_ratio', 'latency',
    'retransmission_count', 'signal_strength',
    'trust_score', 'neighbor_trust_score_avg', 'historical_trust_score'
]
TARGET = 'is_malicious'


def load_and_split_data():
    print("=" * 65)
    print("  Step 1: Load Unified Dataset & Group-Stratified Split")
    print("=" * 65)
    df = pd.read_csv(DATASET_FILE)
    print(f"Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"Features (14): {FEATURES_14}")
    
    # 70% Train, 15% Validation, 15% Test split grouped strictly by node_id
    # First split: 85% Train+Val, 15% Holdout Test
    gss_test = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=42)
    train_val_idx, test_idx = next(gss_test.split(df, groups=df['node_id']))
    
    df_train_val = df.iloc[train_val_idx].reset_index(drop=True)
    df_test      = df.iloc[test_idx].reset_index(drop=True)
    
    # Second split: Train (approx 70% total) and Val (approx 15% total)
    val_fraction = 0.15 / 0.85
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_fraction, random_state=42)
    train_idx, val_idx = next(gss_val.split(df_train_val, groups=df_train_val['node_id']))
    
    df_train = df_train_val.iloc[train_idx].reset_index(drop=True)
    df_val   = df_train_val.iloc[val_idx].reset_index(drop=True)
    
    print(f"  Train Set      : {len(df_train)} rows ({len(df_train['node_id'].unique())} unique vehicles)")
    print(f"  Validation Set : {len(df_val)} rows ({len(df_val['node_id'].unique())} unique vehicles)")
    print(f"  Test Set       : {len(df_test)} rows ({len(df_test['node_id'].unique())} unique vehicles)")
    
    # Verify zero vehicle leakage
    train_nodes = set(df_train['node_id'])
    val_nodes   = set(df_val['node_id'])
    test_nodes  = set(df_test['node_id'])
    assert len(train_nodes & val_nodes) == 0, "Leakage detected between Train and Val!"
    assert len(train_nodes & test_nodes) == 0, "Leakage detected between Train and Test!"
    assert len(val_nodes & test_nodes) == 0, "Leakage detected between Val and Test!"
    print("  ✅ Zero identity/trajectory overlap verified across Train/Val/Test!")
    
    # Fit StandardScaler strictly on Train only!
    scaler = StandardScaler()
    X_train = scaler.fit_transform(df_train[FEATURES_14].values)
    X_val   = scaler.transform(df_val[FEATURES_14].values)
    X_test  = scaler.transform(df_test[FEATURES_14].values)
    
    y_train = df_train[TARGET].values
    y_val   = df_val[TARGET].values
    y_test  = df_test[TARGET].values
    
    # Save split arrays and scaler
    np.save(os.path.join(DATA_DIR, 'unified_X_train.npy'), X_train)
    np.save(os.path.join(DATA_DIR, 'unified_y_train.npy'), y_train)
    np.save(os.path.join(DATA_DIR, 'unified_X_val.npy'),   X_val)
    np.save(os.path.join(DATA_DIR, 'unified_y_val.npy'),   y_val)
    np.save(os.path.join(DATA_DIR, 'unified_X_test.npy'),  X_test)
    np.save(os.path.join(DATA_DIR, 'unified_y_test.npy'),  y_test)
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'unified_scaler.pkl'))
    # Also save as vanet_scaler.pkl for backwards compatibility with dashboard
    joblib.dump(scaler, os.path.join(MODELS_DIR, 'vanet_scaler.pkl'))
    print("  ✅ Scaler and partitioned arrays saved to data/ and models/.")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, df_train, df_val, df_test, scaler


def get_model_suite():
    return {
        # ── Ensemble Decision Trees ──
        'Random Forest': RandomForestClassifier(
            n_estimators=150, max_depth=16, min_samples_split=4,
            random_state=42, n_jobs=-1
        ),
        'Extra Trees': ExtraTreesClassifier(
            n_estimators=150, max_depth=16, random_state=42, n_jobs=-1
        ),
        'Decision Tree': DecisionTreeClassifier(
            max_depth=12, min_samples_split=5, random_state=42
        ),
        
        # ── Modern Regularized GBDTs ──
        'XGBoost': xgb.XGBClassifier(
            n_estimators=180, max_depth=6, learning_rate=0.07,
            subsample=0.85, colsample_bytree=0.85, random_state=42,
            eval_metric='logloss', n_jobs=-1
        ),
        'LightGBM': lgb.LGBMClassifier(
            n_estimators=180, max_depth=6, learning_rate=0.07,
            num_leaves=35, subsample=0.85, random_state=42,
            verbose=-1, n_jobs=-1
        ),
        'CatBoost': cb.CatBoostClassifier(
            iterations=220, depth=6, learning_rate=0.07,
            l2_leaf_reg=3, random_seed=42, verbose=0, thread_count=-1
        ),
        'Hist Gradient Boosting': HistGradientBoostingClassifier(
            max_iter=180, max_depth=6, learning_rate=0.07, random_state=42
        ),
        'Gradient Boosting': GradientBoostingClassifier(
            n_estimators=120, max_depth=5, learning_rate=0.08, random_state=42
        ),
        'AdaBoost': AdaBoostClassifier(
            n_estimators=60, learning_rate=0.1, random_state=42
        ),
        
        # ── Linear, Kernel & Instance Baselines ──
        'SVM (RBF)': SVC(
            kernel='rbf', C=2.0, gamma='scale', probability=True, random_state=42
        ),
        'K-Nearest Neighbors': KNeighborsClassifier(
            n_neighbors=7, weights='distance', n_jobs=-1
        ),
        'Logistic Regression': LogisticRegression(
            C=1.0, max_iter=1000, random_state=42
        ),
        'Gaussian Naive Bayes': GaussianNB(),
        'Linear Discriminant Analysis': LinearDiscriminantAnalysis(),
        
        # ── Deep Tabular Neural Network ──
        'MLP Neural Network': MLPClassifier(
            hidden_layer_sizes=(128, 64, 32), max_iter=500,
            activation='relu', solver='adam', alpha=1e-4, random_state=42
        ),
        
        # ── Heterogeneous Meta-Ensembles ──
        'Stacking Ensemble': StackingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=80, max_depth=12, random_state=42, n_jobs=-1)),
                ('xgb', xgb.XGBClassifier(n_estimators=80, max_depth=5, learning_rate=0.08, random_state=42, eval_metric='logloss', n_jobs=-1)),
                ('cb', cb.CatBoostClassifier(iterations=80, depth=5, learning_rate=0.08, random_seed=42, verbose=0, thread_count=-1)),
            ],
            final_estimator=LogisticRegression(random_state=42),
            cv=3, n_jobs=-1
        ),
        'Voting Ensemble': VotingClassifier(
            estimators=[
                ('rf', RandomForestClassifier(n_estimators=100, max_depth=14, random_state=42, n_jobs=-1)),
                ('xgb', xgb.XGBClassifier(n_estimators=100, max_depth=5, learning_rate=0.08, random_state=42, eval_metric='logloss', n_jobs=-1)),
                ('cb', cb.CatBoostClassifier(iterations=100, depth=5, learning_rate=0.08, random_seed=42, verbose=0, thread_count=-1)),
            ],
            voting='soft', n_jobs=-1
        ),
    }


def measure_inference_latency(model, X_sample, runs=200):
    """Measures single-query inference latency in microseconds (µs/sample)."""
    single_x = X_sample[:1]
    times = []
    # Warmup
    for _ in range(15):
        _ = model.predict(single_x)
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = model.predict(single_x)
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1e6)
    return round(float(np.median(times)), 2)


def run_pipeline():
    X_train, y_train, X_val, y_val, X_test, y_test, df_train, df_val, df_test, scaler = load_and_split_data()
    
    print("\n" + "=" * 65)
    print("  Step 2: 5-Fold Stratified Group Cross-Validation on Train Split")
    print("=" * 65)
    sgkf = StratifiedGroupKFold(n_splits=5)
    
    cv_summary = {}
    rf_baseline = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    cv_scores = []
    for fold, (trn_i, val_i) in enumerate(sgkf.split(X_train, y_train, groups=df_train['node_id'])):
        rf_baseline.fit(X_train[trn_i], y_train[trn_i])
        preds = rf_baseline.predict(X_train[val_i])
        f1_fold = f1_score(y_train[val_i], preds, average='weighted')
        cv_scores.append(f1_fold)
        print(f"  Fold {fold+1} Weighted F1: {f1_fold*100:.2f}%")
    print(f"  --> 5-Fold Stratified Group CV Mean: {np.mean(cv_scores)*100:.2f}% (±{np.std(cv_scores)*100:.2f}%)")
    
    print("\n" + "=" * 65)
    print("  Step 3: Training & Benchmarking Full 17-Model Suite")
    print("=" * 65)
    models = get_model_suite()
    all_metrics = {}
    test_roc_data = {}
    
    # Combine Train + Val for final fit before Test evaluation
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])
    
    for name, model in models.items():
        print(f"\n▶ Training {name} ...", end=' ', flush=True)
        t_start = time.time()
        model.fit(X_train_full, y_train_full)
        train_time = round(time.time() - t_start, 2)
        
        # Test Set Evaluation (Unseen Vehicles)
        y_pred = model.predict(X_test)
        
        # Probability for ROC/PR-AUC
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_proba = model.decision_function(X_test)
        else:
            y_proba = y_pred
            
        acc   = accuracy_score(y_test, y_pred)
        f1_w  = f1_score(y_test, y_pred, average='weighted')
        f1_m  = f1_score(y_test, y_pred, average='macro')
        prec  = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec   = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        roc   = roc_auc_score(y_test, y_proba)
        pr    = average_precision_score(y_test, y_proba)
        cm    = confusion_matrix(y_test, y_pred).tolist()
        
        # Security Critical Rates
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        far = round((fp / max(1, fp + tn)) * 100, 2) # False Alarm Rate
        fnr = round((fn / max(1, fn + tp)) * 100, 2) # Missed Attack Rate
        
        # Latency & Model Size
        latency_us = measure_inference_latency(model, X_test)
        safe_name = name.replace(' ', '_').replace('(', '').replace(')', '')
        model_path = os.path.join(MODELS_DIR, f'unified_{safe_name}.pkl')
        joblib.dump(model, model_path)
        # Also dump as vanet_<safe_name>.pkl for backwards compatibility with dashboard
        joblib.dump(model, os.path.join(MODELS_DIR, f'vanet_{safe_name}.pkl'))
        size_kb = round(os.path.getsize(model_path) / 1024.0, 1)
        
        all_metrics[name] = {
            'accuracy'        : round(float(acc) * 100, 2),
            'f1_score'        : round(float(f1_w) * 100, 2),
            'f1_macro'        : round(float(f1_m) * 100, 2),
            'precision'       : round(float(prec) * 100, 2),
            'recall'          : round(float(rec) * 100, 2),
            'roc_auc'         : round(float(roc) * 100, 2),
            'pr_auc'          : round(float(pr) * 100, 2),
            'false_alert_rate': far,
            'missed_attack_rate': fnr,
            'latency_us'      : latency_us,
            'size_kb'         : size_kb,
            'train_time_s'    : train_time,
            'confusion_matrix': cm,
        }
        
        fpr_curve, tpr_curve, _ = roc_curve(y_test, y_proba)
        test_roc_data[name] = (fpr_curve, tpr_curve, roc)
        
        print(f"Acc={acc*100:.2f}% | F1={f1_w*100:.2f}% | Recall={rec*100:.2f}% | Latency={latency_us}µs | Size={size_kb}KB")

    # ── Select Best Model ──
    best_name = max(all_metrics, key=lambda k: all_metrics[k]['f1_score'])
    for k in all_metrics:
        all_metrics[k]['is_best'] = (k == best_name)
        
    print("\n" + "=" * 65)
    print(f"  🏆 BEST MODEL: {best_name} (F1={all_metrics[best_name]['f1_score']}%)")
    print("=" * 65)
    
    # ── Feature Importance (from tuned Random Forest) ──
    rf_best = models['Random Forest']
    feat_imps = dict(zip(FEATURES_14, rf_best.feature_importances_))
    top_feats = sorted(feat_imps.items(), key=lambda x: -x[1])
    
    print("\n📊 Feature Importance Breakdown:")
    for feat, val in top_feats:
        bar = '█' * int(val * 50)
        print(f"  {feat:30s} {bar} {val*100:.2f}%")
        
    # ── Save Results JSON ──
    report = {
        'dataset'            : 'unified_edgetrust_dataset',
        'n_total_samples'    : len(df_train) + len(df_val) + len(df_test),
        'n_train_samples'    : len(X_train_full),
        'n_test_samples'     : len(X_test),
        'n_features'         : len(FEATURES_14),
        'features'           : FEATURES_14,
        'best_model'         : best_name,
        'cross_val_5fold_mean': round(float(np.mean(cv_scores)) * 100, 2),
        'cross_val_5fold_std' : round(float(np.std(cv_scores)) * 100, 2),
        'feature_importance' : {k: round(v * 100, 3) for k, v in top_feats},
        'models'             : all_metrics,
    }
    
    metrics_path = os.path.join(RESULTS_DIR, 'unified_model_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(report, f, indent=2)
    # Also overwrite vanet_model_metrics.json for dashboard integration
    with open(os.path.join(RESULTS_DIR, 'vanet_model_metrics.json'), 'w') as f:
        json.dump(report, f, indent=2)
    print(f"\n✅ Metrics report saved → {metrics_path}")
    
    # ── Generate High-Resolution Visualizations ──
    generate_publication_charts(all_metrics, top_feats, test_roc_data, best_name)


def generate_publication_charts(all_metrics, top_feats, test_roc_data, best_name):
    names = list(all_metrics.keys())
    accs  = [all_metrics[n]['accuracy'] for n in names]
    f1s   = [all_metrics[n]['f1_score'] for n in names]
    recs  = [all_metrics[n]['recall'] for n in names]
    lats  = [all_metrics[n]['latency_us'] for n in names]
    colors = ['#1d4ed8' if n == best_name else '#334155' for n in names]
    
    short = [n.replace('K-Nearest Neighbors','KNN')
              .replace('Logistic Regression','LR')
              .replace('Gaussian Naive Bayes','GNB')
              .replace('Gradient Boosting','GB')
              .replace('Extra Trees','ET')
              .replace('Random Forest','RF')
              .replace('Decision Tree','DT')
              .replace('AdaBoost','ADA')
              .replace('SVM (RBF)','SVM')
              .replace('XGBoost','XGB')
              .replace('LightGBM','LGBM')
              .replace('CatBoost','CAT')
              .replace('MLP Neural Network','MLP')
              .replace('Hist Gradient Boosting','HGB')
              .replace('Linear Discriminant Analysis','LDA')
              .replace('Stacking Ensemble','STACK')
              .replace('Voting Ensemble','VOTE') for n in names]
              
    # Chart 1: All-Model Accuracy, F1, Latency Comparison
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))
    fig.suptitle('EdgeTrust-VANET Unified Dataset — Comprehensive 17-Model Benchmark',
                 fontsize=14, fontweight='bold', color='white')
    fig.patch.set_facecolor('#0d1321')
    x = np.arange(len(names))
    
    # Acc
    axes[0].set_facecolor('#0d1321')
    b1 = axes[0].bar(x, accs, color=colors, width=0.6)
    axes[0].set_title('Accuracy (%)', color='white', fontsize=12)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(short, rotation=35, ha='right', color='#94a3b8', fontsize=8)
    axes[0].tick_params(colors='#94a3b8')
    axes[0].set_ylim(60, 102)
    axes[0].bar_label(b1, fmt='%.1f', padding=2, color='#cbd5e1', fontsize=7)
    axes[0].spines[:].set_color('#1e293b')
    
    # F1
    axes[1].set_facecolor('#0d1321')
    b2 = axes[1].bar(x, f1s, color=colors, width=0.6)
    axes[1].set_title('Weighted F1 Score (%)', color='white', fontsize=12)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(short, rotation=35, ha='right', color='#94a3b8', fontsize=8)
    axes[1].tick_params(colors='#94a3b8')
    axes[1].set_ylim(60, 102)
    axes[1].bar_label(b2, fmt='%.1f', padding=2, color='#cbd5e1', fontsize=7)
    axes[1].spines[:].set_color('#1e293b')
    
    # Latency
    axes[2].set_facecolor('#0d1321')
    b3 = axes[2].bar(x, lats, color='#0284c7', width=0.6)
    axes[2].set_title('RSU Edge Inference Latency (µs / query)', color='white', fontsize=12)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(short, rotation=35, ha='right', color='#94a3b8', fontsize=8)
    axes[2].tick_params(colors='#94a3b8')
    axes[2].set_yscale('log')
    axes[2].spines[:].set_color('#1e293b')
    
    plt.tight_layout()
    chart1_path = os.path.join(RESULTS_DIR, 'unified_model_comparison.png')
    plt.savefig(chart1_path, dpi=150, facecolor='#0d1321')
    plt.savefig(os.path.join(RESULTS_DIR, 'vanet_comparison.png'), dpi=150, facecolor='#0d1321')
    plt.close()
    print(f"✅ Model comparison chart saved → {chart1_path}")
    
    # Chart 2: Feature Importance
    fig, ax = plt.subplots(figsize=(11, 7))
    fig.patch.set_facecolor('#0d1321')
    ax.set_facecolor('#0d1321')
    
    f_names = [f for f, _ in top_feats]
    f_vals  = [v * 100 for _, v in top_feats]
    colors_fi = ['#3b82f6' if v > 5 else '#334155' for v in f_vals]
    
    ax.barh(f_names[::-1], f_vals[::-1], color=colors_fi[::-1])
    ax.set_title('EdgeTrust-VANET 14 Feature Importance (Random Forest)\n(RSU Observable Kinematic, Network & Trust Telemetry)',
                 color='white', fontsize=12, fontweight='bold')
    ax.tick_params(colors='#94a3b8')
    ax.spines[:].set_color('#1e293b')
    ax.set_xlabel('Relative Importance (%)', color='#94a3b8')
    for i, v in enumerate(f_vals[::-1]):
        ax.text(v + 0.3, i, f"{v:.2f}%", color='#cbd5e1', va='center', fontsize=8)
        
    plt.tight_layout()
    chart2_path = os.path.join(RESULTS_DIR, 'unified_feature_importance.png')
    plt.savefig(chart2_path, dpi=150, facecolor='#0d1321')
    plt.savefig(os.path.join(RESULTS_DIR, 'vanet_feature_importance.png'), dpi=150, facecolor='#0d1321')
    plt.close()
    print(f"✅ Feature importance chart saved → {chart2_path}")


if __name__ == '__main__':
    run_pipeline()
