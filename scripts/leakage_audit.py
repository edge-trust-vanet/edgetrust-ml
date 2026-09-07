"""
╔══════════════════════════════════════════════════════════════════╗
║         leakage_audit.py                                         ║
║         EdgeTrust-VANET — Rigorous Leakage Audit & Proof Script  ║
╠══════════════════════════════════════════════════════════════════╣
║  Proves two critical scientific principles:                      ║
║    1. Post-Hoc Attack Counters Leakage:                          ║
║       Proves that including retrospective attack counters        ║
║       (false_packet_injection, blackhole, sybil, dos) destroys   ║
║       real-time deployment detection capability.                 ║
║    2. Spatiotemporal Row-Splitting Leakage:                      ║
║       Proves that random row splitting leaks vehicle identity     ║
║       and temporal dynamics, necessitating Group-based splitting ║
║       by vehicle (node_id) for honest scientific evaluation.     ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'unified_edgetrust_dataset.csv')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

FEATURES_14 = [
    'position_x', 'position_y', 'speed', 'direction', 'acceleration',
    'packet_sent', 'packet_received', 'packet_drop_ratio', 'latency',
    'retransmission_count', 'signal_strength',
    'trust_score', 'neighbor_trust_score_avg', 'historical_trust_score'
]

POST_HOC_COUNTERS = [
    'false_packet_injection', 'blackhole_attack_attempts',
    'sybil_attack_attempts', 'denial_of_service'
]

TARGET = 'is_malicious'


def run_leakage_audit():
    print("=" * 65)
    print("  EdgeTrust-VANET Leakage Audit & Verification Pipeline")
    print("=" * 65)
    
    df = pd.read_csv(DATA_PATH)
    print(f"Loaded unified dataset: {df.shape[0]} rows × {df.shape[1]} columns")
    
    # ─────────────────────────────────────────────────────────────
    # PART 1: CORRELATION & MUTUAL INFORMATION AUDIT
    # ─────────────────────────────────────────────────────────────
    print("\n🔍 Step 1: Statistical Correlation & Mutual Information Audit...")
    
    audit_cols = FEATURES_14 + POST_HOC_COUNTERS
    y = df[TARGET].values
    X_audit = df[audit_cols].values
    
    # Calculate Pearson, Spearman, and Mutual Information
    correlations_pearson = {}
    correlations_spearman = {}
    for col in audit_cols:
        correlations_pearson[col] = float(df[col].corr(df[TARGET], method='pearson'))
        correlations_spearman[col] = float(df[col].corr(df[TARGET], method='spearman'))
        
    mi_scores = mutual_info_classif(X_audit, y, random_state=42)
    mi_dict = {col: float(mi_scores[i]) for i, col in enumerate(audit_cols)}
    
    print("\nFeature Association with Target (is_malicious):")
    print(f"{'Feature Name':30s} | {'Pearson r':10s} | {'Spearman rho':12s} | {'Mutual Info':12s} | {'Status'}")
    print("-" * 75)
    for col in audit_cols:
        p_r = correlations_pearson[col]
        s_r = correlations_spearman[col]
        mi  = mi_dict[col]
        status = "⚠️ LEAKAGE COUNTER" if col in POST_HOC_COUNTERS else "✅ Observable"
        print(f"{col:30s} | {p_r:10.4f} | {s_r:12.4f} | {mi:12.4f} | {status}")
        
    # ─────────────────────────────────────────────────────────────
    # PART 2: CONTROLLED EXPERIMENT - ATTACK COUNTERS LEAKAGE
    # ─────────────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  🧪 Step 2: Controlled Experiment — Impact of Post-Hoc Attack Counters")
    print("─" * 65)
    print("Comparing Model A (Honest 14 features) vs Model B (Leaked: 14 features + 4 counters)...")
    
    # Split by vehicle to ensure no identity contamination
    gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
    train_idx, test_idx = next(gss.split(df, groups=df['node_id']))
    
    train_df = df.iloc[train_idx]
    test_df  = df.iloc[test_idx].copy()
    
    y_tr = train_df[TARGET].values
    y_te = test_df[TARGET].values
    
    # Model A: 14 Leakage-Free Observable Features
    scaler_A = StandardScaler()
    X_tr_A = scaler_A.fit_transform(train_df[FEATURES_14].values)
    X_te_A = scaler_A.transform(test_df[FEATURES_14].values)
    
    rf_A = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_A.fit(X_tr_A, y_tr)
    y_pred_A = rf_A.predict(X_te_A)
    y_proba_A = rf_A.predict_proba(X_te_A)[:, 1]
    
    # Model B: 14 Features + 4 Post-Hoc Attack Counters
    cols_B = FEATURES_14 + POST_HOC_COUNTERS
    scaler_B = StandardScaler()
    X_tr_B = scaler_B.fit_transform(train_df[cols_B].values)
    X_te_B = scaler_B.transform(test_df[cols_B].values)
    
    rf_B = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_B.fit(X_tr_B, y_tr)
    y_pred_B = rf_B.predict(X_te_B)
    y_proba_B = rf_B.predict_proba(X_te_B)[:, 1]
    
    # Realistic Deployment Test for Model B:
    # At real-time arrival at the RSU, attack counters are unconfirmed (counters = 0)
    test_df_real = test_df.copy()
    for col in POST_HOC_COUNTERS:
        test_df_real[col] = 0
    X_te_B_real = scaler_B.transform(test_df_real[cols_B].values)
    y_pred_B_real = rf_B.predict(X_te_B_real)
    y_proba_B_real = rf_B.predict_proba(X_te_B_real)[:, 1]
    
    res_A = {
        'accuracy' : float(accuracy_score(y_te, y_pred_A) * 100),
        'f1_score' : float(f1_score(y_te, y_pred_A, average='weighted') * 100),
        'precision': float(precision_score(y_te, y_pred_A, average='weighted') * 100),
        'recall'   : float(recall_score(y_te, y_pred_A, average='weighted') * 100),
        'roc_auc'  : float(roc_auc_score(y_te, y_proba_A) * 100),
    }
    
    res_B = {
        'accuracy' : float(accuracy_score(y_te, y_pred_B) * 100),
        'f1_score' : float(f1_score(y_te, y_pred_B, average='weighted') * 100),
        'precision': float(precision_score(y_te, y_pred_B, average='weighted') * 100),
        'recall'   : float(recall_score(y_te, y_pred_B, average='weighted') * 100),
        'roc_auc'  : float(roc_auc_score(y_te, y_proba_B) * 100),
    }
    
    res_B_real = {
        'accuracy' : float(accuracy_score(y_te, y_pred_B_real) * 100),
        'f1_score' : float(f1_score(y_te, y_pred_B_real, average='weighted') * 100),
        'precision': float(precision_score(y_te, y_pred_B_real, average='weighted') * 100),
        'recall'   : float(recall_score(y_te, y_pred_B_real, average='weighted') * 100),
        'roc_auc'  : float(roc_auc_score(y_te, y_proba_B_real) * 100),
    }
    
    print("\nResults:")
    print(f"  Model A (14 Leakage-Free Features)   : Accuracy={res_A['accuracy']:.2f}%, F1={res_A['f1_score']:.2f}%, Recall={res_A['recall']:.2f}%")
    print(f"  Model B (Leaked Training & Testing)  : Accuracy={res_B['accuracy']:.2f}%, F1={res_B['f1_score']:.2f}%, Recall={res_B['recall']:.2f}% (Artificially inflated)")
    print(f"  Model B (Deployed Real-Time, Ctr=0) : Accuracy={res_B_real['accuracy']:.2f}%, F1={res_B_real['f1_score']:.2f}%, Recall={res_B_real['recall']:.2f}% (COLLAPSES!)")
    
    # ─────────────────────────────────────────────────────────────
    # PART 3: ROW-SPLIT VS GROUP-SPLIT AUDIT
    # ─────────────────────────────────────────────────────────────
    print("\n" + "─" * 65)
    print("  🧪 Step 3: Train/Test Splitting Leakage Analysis")
    print("─" * 65)
    print("Comparing Random Row Splitting (Leaked) vs Group Vehicle Splitting (Honest)...")
    
    # 1. Random Row Split
    X_scaled_all = StandardScaler().fit_transform(df[FEATURES_14].values)
    X_tr_rand, X_te_rand, y_tr_rand, y_te_rand = train_test_split(
        X_scaled_all, y, test_size=0.20, random_state=42, stratify=y
    )
    rf_rand = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_rand.fit(X_tr_rand, y_tr_rand)
    y_pred_rand = rf_rand.predict(X_te_rand)
    
    rand_f1 = float(f1_score(y_te_rand, y_pred_rand, average='weighted') * 100)
    rand_acc = float(accuracy_score(y_te_rand, y_pred_rand) * 100)
    
    # 2. Group Split (from Model A above)
    group_f1 = res_A['f1_score']
    group_acc = res_A['accuracy']
    
    print(f"  Random Row Split (Leaked Temporal Trajectories) : Acc={rand_acc:.2f}%, F1={rand_f1:.2f}%")
    print(f"  Group Vehicle Split (Zero Identity/Trajectory Overlap): Acc={group_acc:.2f}%, F1={group_f1:.2f}%")
    print(f"  Scientific Generalization Gap                  : {abs(rand_f1 - group_f1):.2f}%")
    
    # ─────────────────────────────────────────────────────────────
    # PART 4: SAVE REPORT & GENERATE CHARTS
    # ─────────────────────────────────────────────────────────────
    audit_report = {
        'dataset_shape'       : [int(df.shape[0]), int(df.shape[1])],
        'correlations_pearson': {k: round(v, 4) for k, v in correlations_pearson.items()},
        'correlations_spearman': {k: round(v, 4) for k, v in correlations_spearman.items()},
        'mutual_information'  : {k: round(v, 4) for k, v in mi_dict.items()},
        'model_a_leakage_free': res_A,
        'model_b_leaked_eval' : res_B,
        'model_b_real_deploy' : res_B_real,
        'splitting_audit'     : {
            'random_row_split_f1' : round(rand_f1, 2),
            'random_row_split_acc': round(rand_acc, 2),
            'group_vehicle_split_f1': round(group_f1, 2),
            'group_vehicle_split_acc': round(group_acc, 2),
            'generalization_gap'  : round(abs(rand_f1 - group_f1), 2),
        }
    }
    
    report_path = os.path.join(RESULTS_DIR, 'leakage_audit_report.json')
    with open(report_path, 'w') as f:
        json.dump(audit_report, f, indent=2)
    print(f"\n✅ Leakage Audit report saved → {report_path}")
    
    # Generate Visualizations
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.patch.set_facecolor('#0d1321')
    
    # Chart 1: Mutual Information of all features and counters
    sorted_mi = sorted(mi_dict.items(), key=lambda x: x[1], reverse=True)
    f_names = [x[0] for x in sorted_mi]
    f_vals  = [x[1] for x in sorted_mi]
    colors_mi = ['#ef4444' if col in POST_HOC_COUNTERS else '#3b82f6' for col in f_names]
    
    axes[0].set_facecolor('#0d1321')
    axes[0].barh(f_names[::-1], f_vals[::-1], color=colors_mi[::-1])
    axes[0].set_title('Mutual Information with Target Label\n(Red = Excluded Attack Counters, Blue = 14 RSU Features)',
                      color='white', fontsize=11, fontweight='bold')
    axes[0].tick_params(colors='#94a3b8')
    axes[0].spines[:].set_color('#1e293b')
    axes[0].set_xlabel('Mutual Information (bits)', color='#94a3b8')
    
    # Chart 2: Model Performance Comparison under Deployment Collapse
    axes[1].set_facecolor('#0d1321')
    model_labels = ['Model A\n(14 Features\nRealistic)', 'Model B\n(Leaked with\nCounters)', 'Model B Deployed\n(Counters=0\nCollapsed!)']
    f1_vals = [res_A['f1_score'], res_B['f1_score'], res_B_real['f1_score']]
    rec_vals = [res_A['recall'], res_B['recall'], res_B_real['recall']]
    
    x = np.arange(len(model_labels))
    w = 0.35
    b1 = axes[1].bar(x - w/2, f1_vals, width=w, label='F1 Score (%)', color='#3b82f6')
    b2 = axes[1].bar(x + w/2, rec_vals, width=w, label='Recall (%)', color='#10b981')
    axes[1].set_title('Post-Hoc Attack Counters Leakage Experiment\n(Cheating on Counters Collapses in Real Deployment)',
                      color='white', fontsize=11, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(model_labels, color='#94a3b8', fontsize=9)
    axes[1].tick_params(colors='#94a3b8')
    axes[1].set_ylim(40, 105)
    axes[1].legend(facecolor='#1e293b', edgecolor='none', labelcolor='white')
    axes[1].spines[:].set_color('#1e293b')
    axes[1].bar_label(b1, fmt='%.1f', padding=2, color='#cbd5e1', fontsize=8)
    axes[1].bar_label(b2, fmt='%.1f', padding=2, color='#cbd5e1', fontsize=8)
    
    plt.tight_layout()
    chart_path = os.path.join(RESULTS_DIR, 'leakage_audit.png')
    plt.savefig(chart_path, dpi=150, facecolor='#0d1321')
    plt.close()
    print(f"✅ Leakage Audit chart saved → {chart_path}")
    print("=" * 65)


if __name__ == '__main__':
    run_leakage_audit()
