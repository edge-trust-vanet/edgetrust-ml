"""
╔══════════════════════════════════════════════════════════════════╗
║         generate_slide_charts.py                                 ║
║         EdgeTrust-VANET — Slide & Panel Review Figure Generator  ║
╠══════════════════════════════════════════════════════════════════╣
║  Generates presentation-ready, high-resolution figures based on: ║
║    1. Unified Dataset (23,970 samples) offline training/metrics  ║
║    2. 2,331 Live OMNeT++ / Veins RSU Simulation Telemetry Tests  ║
║    3. Pareto Trade-off: Live F1 vs Edge Latency vs Model Size    ║
║    4. False Alarm Rate vs Missed Attack Rate Error Analysis      ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Live OMNeT++ / Veins RSU Simulation Data ──────────────────────
# ── Live OMNeT++ / Veins RSU Simulation Test Data ───────────────────
# Validation Acc is from the Split Dataset (results/unified_model_metrics.json, 3,539 samples)
# Live Acc/F1/Latency is from Live OMNeT++/Veins RSU Simulation (2,331 packets)
LIVE_DATA = [
    {"rank": 1,  "name": "Random Forest",          "live_acc": 90.99, "live_f1": 90.97, "live_auc": 90.59, "far": 11.91, "mar": 6.89,  "latency": 23499.6, "throughput": 42,    "size_kb": 6441.0, "val_acc": 97.77},
    {"rank": 2,  "name": "Extra Trees",            "live_acc": 90.95, "live_f1": 90.94, "live_auc": 92.17, "far": 11.71, "mar": 7.12,  "latency": 24162.8, "throughput": 41,    "size_kb": 11314.0,"val_acc": 92.43},
    {"rank": 3,  "name": "K-Nearest Neighbors",    "live_acc": 90.48, "live_f1": 90.48, "live_auc": 90.88, "far": 11.30, "mar": 8.23,  "latency": 13903.8, "throughput": 71,    "size_kb": 5045.0, "val_acc": 90.96},
    {"rank": 4,  "name": "Logistic Regression",    "live_acc": 89.27, "live_f1": 89.28, "live_auc": 88.79, "far": 12.42, "mar": 9.49,  "latency": 82.8,    "throughput": 12084, "size_kb": 1.0,    "val_acc": 88.08},
    {"rank": 5,  "name": "Linear Discriminant",     "live_acc": 89.27, "live_f1": 89.28, "live_auc": 88.87, "far": 12.42, "mar": 9.49,  "latency": 94.9,    "throughput": 10536, "size_kb": 1.6,    "val_acc": 87.93},
    {"rank": 6,  "name": "AdaBoost",               "live_acc": 86.06, "live_f1": 86.14, "live_auc": 91.25, "far": 9.27,  "mar": 17.35, "latency": 5040.1,  "throughput": 198,   "size_kb": 38.9,   "val_acc": 88.25},
    {"rank": 7,  "name": "Voting Ensemble",        "live_acc": 83.27, "live_f1": 82.72, "live_auc": 90.36, "far": 33.20, "mar": 4.74,  "latency": 39873.5, "throughput": 25,    "size_kb": 3905.0, "val_acc": 97.20},
    {"rank": 8,  "name": "CatBoost",               "live_acc": 80.69, "live_f1": 79.79, "live_auc": 90.77, "far": 39.82, "mar": 4.37,  "latency": 118.3,   "throughput": 8452,  "size_kb": 248.0,  "val_acc": 97.29},
    {"rank": 9,  "name": "LightGBM",               "live_acc": 72.42, "live_f1": 69.32, "live_auc": 89.77, "far": 61.10, "mar": 3.19,  "latency": 536.9,   "throughput": 1862,  "size_kb": 669.0,  "val_acc": 97.74},
    {"rank": 10, "name": "Hist Gradient Boosting", "live_acc": 71.69, "live_f1": 68.28, "live_auc": 89.50, "far": 63.03, "mar": 3.04,  "latency": 3880.7,  "throughput": 257,   "size_kb": 653.0,  "val_acc": 97.40},
    {"rank": 11, "name": "XGBoost",                "live_acc": 71.56, "live_f1": 68.23, "live_auc": 89.32, "far": 62.73, "mar": 3.48,  "latency": 280.2,   "throughput": 3568,  "size_kb": 464.0,  "val_acc": 97.29},
    {"rank": 12, "name": "Stacking Ensemble",      "live_acc": 69.24, "live_f1": 64.47, "live_auc": 91.76, "far": 70.16, "mar": 2.08,  "latency": 38560.3, "throughput": 25,    "size_kb": 2256.0, "val_acc": 97.51},
    {"rank": 13, "name": "Gradient Boosting",      "live_acc": 67.14, "live_f1": 60.67, "live_auc": 89.90, "far": 76.88, "mar": 0.82,  "latency": 237.8,   "throughput": 4205,  "size_kb": 535.0,  "val_acc": 97.17},
    {"rank": 14, "name": "MLP Neural Network",     "live_acc": 58.99, "live_f1": 44.90, "live_auc": 76.14, "far": 97.35, "mar": 0.00,  "latency": 119.1,   "throughput": 8393,  "size_kb": 392.0,  "val_acc": 96.13},
    {"rank": 15, "name": "Gaussian Naive Bayes",   "live_acc": 58.13, "live_f1": 43.24, "live_auc": 90.17, "far": 99.08, "mar": 0.22,  "latency": 83.6,    "throughput": 11963, "size_kb": 1.2,    "val_acc": 82.17},
    {"rank": 16, "name": "SVM (RBF)",              "live_acc": 58.04, "live_f1": 42.82, "live_auc": 87.82, "far": 99.59, "mar": 0.00,  "latency": 255.4,   "throughput": 3915,  "size_kb": 742.0,  "val_acc": 90.03},
    {"rank": 17, "name": "Decision Tree",          "live_acc": 57.87, "live_f1": 42.43, "live_auc": 50.00, "far": 100.0,  "mar": 0.00,  "latency": 70.9,    "throughput": 14110, "size_kb": 18.1,   "val_acc": 96.84},
]

# Save JSON file
json_path = os.path.join(RESULTS_DIR, 'live_rsu_simulation_benchmark.json')
with open(json_path, 'w') as f:
    json.dump({
        "environment": "OMNeT++ / Veins Live RSU Simulation",
        "description": "Validation performed on Split Dataset; Testing performed on Live RSU Environment",
        "live_test_packets": 2331,
        "attack_packets": 1349,
        "legitimate_packets": 982,
        "models": LIVE_DATA
    }, f, indent=2)
print(f"Saved: {json_path}")

# Short abbreviations
short_names = [
    m['name'].replace('Random Forest', 'RF (Live Demo)')
            .replace('Extra Trees', 'Extra Trees')
            .replace('K-Nearest Neighbors', 'KNN')
            .replace('Logistic Regression', 'LogReg')
            .replace('Linear Discriminant', 'LDA')
            .replace('AdaBoost', 'AdaBoost')
            .replace('Voting Ensemble', 'Voting')
            .replace('CatBoost', 'CatBoost')
            .replace('LightGBM', 'LightGBM')
            .replace('Hist Gradient Boosting', 'HistGB')
            .replace('XGBoost', 'XGBoost')
            .replace('Stacking Ensemble', 'Stacking')
            .replace('Gradient Boosting', 'GradBoost')
            .replace('MLP Neural Network', 'MLP Net')
            .replace('Gaussian Naive Bayes', 'GNB')
            .replace('SVM (RBF)', 'SVM (RBF)')
            .replace('Decision Tree', 'Decision Tree')
    for m in LIVE_DATA
]


# ═══════════════════════════════════════════════════════════════════
# CHART 1: Dataset Validation vs Live RSU Simulation Test Accuracy & F1
# ═══════════════════════════════════════════════════════════════════
def plot_live_vs_training():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6.5))
    fig.patch.set_facecolor('#0d1321')
    
    x = np.arange(len(LIVE_DATA))
    width = 0.38
    
    val_accs  = [m['val_acc']  for m in LIVE_DATA]
    live_accs = [m['live_acc'] for m in LIVE_DATA]
    live_f1s  = [m['live_f1']  for m in LIVE_DATA]
    
    # Left: Validation vs Live Test Accuracy
    ax1.set_facecolor('#0d1321')
    b1 = ax1.bar(x - width/2, val_accs,  width, label='Validation Acc (Split Dataset, %)', color='#3b82f6')
    b2 = ax1.bar(x + width/2, live_accs, width, label='Live Test Acc (OMNeT++/Veins RSU, %)', color='#10b981')
    ax1.set_title('Dataset Validation vs Live RSU Simulation Test Accuracy\n(Split Dataset Validation vs 2,331 Live OMNeT++/Veins Packets)',
                  color='white', fontsize=12, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(short_names, rotation=45, ha='right', color='#94a3b8', fontsize=8)
    ax1.tick_params(colors='#94a3b8')
    ax1.set_ylim(45, 105)
    ax1.spines[:].set_color('#1e293b')
    ax1.legend(facecolor='#1e293b', edgecolor='none', labelcolor='white', loc='upper right')
    
    # Highlight top model bar
    b2[0].set_color('#f59e0b')
    
    # Right: Live Test F1 Score
    ax2.set_facecolor('#0d1321')
    colors_f1 = ['#f59e0b' if i == 0 else ('#0284c7' if m['live_f1'] >= 80 else '#475569') for i, m in enumerate(LIVE_DATA)]
    b3 = ax2.bar(x, live_f1s, color=colors_f1, width=0.65)
    ax2.set_title('Live RSU Simulation Test F1-Score (%)\n(Ranked by Live Detection Performance on 2,331 Packets)',
                  color='white', fontsize=12, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(short_names, rotation=45, ha='right', color='#94a3b8', fontsize=8)
    ax2.tick_params(colors='#94a3b8')
    ax2.set_ylim(35, 100)
    ax2.spines[:].set_color('#1e293b')
    ax2.bar_label(b3, fmt='%.1f', padding=2, color='#cbd5e1', fontsize=7.5)
    
    plt.tight_layout()
    p1 = os.path.join(RESULTS_DIR, 'live_rsu_benchmark.png')
    plt.savefig(p1, dpi=180, facecolor='#0d1321')
    plt.savefig(os.path.join(RESULTS_DIR, 'vanet_comparison.png'), dpi=180, facecolor='#0d1321')
    plt.close()
    print(f"Saved: {p1}")


# ═══════════════════════════════════════════════════════════════════
# CHART 2: Pareto Frontier: Live F1 vs Edge Latency & Size
# ═══════════════════════════════════════════════════════════════════
def plot_edge_tradeoff():
    fig, ax = plt.subplots(figsize=(12, 7))
    fig.patch.set_facecolor('#0d1321')
    ax.set_facecolor('#0d1321')
    
    latencies = [m['latency'] for m in LIVE_DATA]
    f1_scores = [m['live_f1'] for m in LIVE_DATA]
    sizes     = [max(40, np.sqrt(m['size_kb']) * 14) for m in LIVE_DATA] # Scaled for bubble size
    
    # Scatter plot
    scatter = ax.scatter(latencies, f1_scores, s=sizes, c=f1_scores, cmap='viridis',
                         alpha=0.85, edgecolors='white', linewidth=1.2)
    
    ax.set_xscale('log')
    ax.set_title('Edge RSU Deployment Trade-off: Live F1 vs Inference Latency\n(Bubble size proportional to Model Storage Footprint; Higher & Left is Best)',
                 color='white', fontsize=13, fontweight='bold', pad=15)
    ax.set_xlabel('Edge Inference Latency (µs, log scale)', color='#94a3b8', fontsize=11)
    ax.set_ylabel('Live RSU Simulation F1-Score (%)', color='#94a3b8', fontsize=11)
    ax.tick_params(colors='#94a3b8')
    ax.spines[:].set_color('#1e293b')
    ax.grid(True, which='both', linestyle='--', alpha=0.15, color='#94a3b8')
    
    # Annotate key models
    key_models = [
        ("Random Forest (Live Demo)", (1.1, 1.2)),
        ("Extra Trees", (1.1, -1.8)),
        ("Logistic Regression", (1.15, 0.8)),
        ("Linear Discriminant", (1.15, -1.5)),
        ("CatBoost", (1.15, 1.0)),
        ("AdaBoost", (1.15, 1.0)),
        ("LightGBM", (1.15, 1.0)),
        ("MLP Neural Network", (1.15, -1.2)),
        ("Decision Tree", (1.15, 0.5))
    ]
    for name, offset in key_models:
        for m in LIVE_DATA:
            if m['name'] in name or name in m['name']:
                ax.annotate(
                    f"{m['name']}\n(F1={m['live_f1']}%, {m['latency']:.1f}µs)",
                    xy=(m['latency'], m['live_f1']),
                    xytext=(m['latency'] * offset[0], m['live_f1'] + offset[1]),
                    color='#e2e8f0', fontsize=8.5, fontweight='semibold',
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#1e293b', edgecolor='#3b82f6', alpha=0.85),
                    arrowprops=dict(arrowstyle='->', color='#38bdf8', lw=0.9)
                )
                break
                
    # Add Pareto optimal highlight box
    ax.axvspan(50, 200, color='#10b981', alpha=0.08)
    ax.text(60, 48, '⚡ Ultra-High Throughput Zone\n(>8,000 pkt/s, <200 µs)\nLogReg, LDA, CatBoost',
            color='#34d399', fontsize=9.5, fontweight='bold', bbox=dict(boxstyle='square', facecolor='#064e3b', alpha=0.6))
            
    ax.set_ylim(38, 96)
    
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Live F1 Score (%)', color='#94a3b8')
    cbar.ax.tick_params(colors='#94a3b8')
    
    plt.tight_layout()
    p2 = os.path.join(RESULTS_DIR, 'edge_rsu_tradeoff.png')
    plt.savefig(p2, dpi=180, facecolor='#0d1321')
    plt.savefig(os.path.join(RESULTS_DIR, 'adaboost_size_speed.png'), dpi=180, facecolor='#0d1321')
    plt.savefig(os.path.join(RESULTS_DIR, 'adaboost_edge_analysis.png'), dpi=180, facecolor='#0d1321')
    plt.close()
    print(f"Saved: {p2}")


# ═══════════════════════════════════════════════════════════════════
# CHART 3: Alpha Trust Update Sensitivity Analysis
# ═══════════════════════════════════════════════════════════════════
def plot_alpha_sensitivity():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor('#0d1321')
    ax.set_facecolor('#0d1321')
    
    alphas = [0.3, 0.5, 0.7, 0.9]
    packets = np.arange(0, 15)
    
    # Attack scenario: malicious evidence = 0.15
    for a in alphas:
        trusts = [1.0]
        for p in range(1, len(packets)):
            t_new = a * trusts[-1] + (1 - a) * 0.15
            trusts.append(t_new)
        line_style = '-' if a == 0.7 else '--'
        lw = 2.8 if a == 0.7 else 1.6
        color = '#f59e0b' if a == 0.7 else ('#38bdf8' if a == 0.5 else ('#ef4444' if a == 0.3 else '#94a3b8'))
        ax.plot(packets, trusts, label=f'α = {a} ' + ('(Optimal EdgeTrust)' if a == 0.7 else ''),
                linestyle=line_style, linewidth=lw, color=color, marker='o' if a==0.7 else None)
                
    # Decision boundaries
    ax.axhline(0.70, color='#10b981', linestyle=':', label='Trusted Threshold (0.70)')
    ax.axhline(0.40, color='#ef4444', linestyle=':', label='Blocked Threshold (0.40)')
    
    ax.set_title('Why α = 0.70 in the Hybrid Trust Update Equation\n(Simulating Attack Onset: Rapid Penalty with Noise Resistance)',
                 color='white', fontsize=12, fontweight='bold')
    ax.set_xlabel('Successive Telemetry Observations (Packets)', color='#94a3b8', fontsize=10)
    ax.set_ylabel('Vehicle Trust Score T(n)', color='#94a3b8', fontsize=10)
    ax.tick_params(colors='#94a3b8')
    ax.spines[:].set_color('#1e293b')
    ax.legend(facecolor='#1e293b', edgecolor='none', labelcolor='white')
    ax.grid(True, linestyle='--', alpha=0.15, color='#94a3b8')
    
    ax.annotate('α=0.7 drops to BLOCKED at Packet 4\nBalanced against isolated sensor noise',
                xy=(4, 0.38), xytext=(5.5, 0.52),
                color='#fef08a', fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#78350f', alpha=0.8),
                arrowprops=dict(arrowstyle='->', color='#f59e0b', lw=1.5))
                
    plt.tight_layout()
    p3 = os.path.join(RESULTS_DIR, 'alpha_trust_update.png')
    plt.savefig(p3, dpi=180, facecolor='#0d1321')
    plt.savefig(os.path.join(RESULTS_DIR, 'trust_alpha_sensitivity.png'), dpi=180, facecolor='#0d1321')
    plt.close()
    print(f"Saved: {p3}")


# ═══════════════════════════════════════════════════════════════════
# CHART 4: False Alarm Rate vs Missed Attack Rate
# ═══════════════════════════════════════════════════════════════════
def plot_error_tradeoff():
    fig, ax = plt.subplots(figsize=(11, 6))
    fig.patch.set_facecolor('#0d1321')
    ax.set_facecolor('#0d1321')
    
    fars = [m['far'] for m in LIVE_DATA]
    mars = [m['mar'] for m in LIVE_DATA]
    
    ax.scatter(fars, mars, color='#38bdf8', s=120, edgecolors='white', linewidth=1)
    
    for m in LIVE_DATA:
        if m['rank'] <= 8 or m['name'] in ['Decision Tree', 'MLP Neural Network']:
            ax.annotate(f"{m['name']} (#{m['rank']})",
                        xy=(m['far'], m['mar']),
                        xytext=(m['far'] + 2, m['mar'] + 0.8),
                        color='#cbd5e1', fontsize=8.5,
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='#1e293b', alpha=0.75),
                        arrowprops=dict(arrowstyle='->', color='#38bdf8', lw=0.7))
                        
    ax.set_title('Security-Critical Error Rates on Live RSU Telemetry\n(False Alarm Rate vs Missed Attack Rate; Bottom-Left is Best)',
                 color='white', fontsize=12, fontweight='bold')
    ax.set_xlabel('False Alarm Rate (FAR %) — Benign Flagged as Malicious', color='#94a3b8', fontsize=10)
    ax.set_ylabel('Missed Attack Rate (MAR %) — Attacks Escaped Detection', color='#94a3b8', fontsize=10)
    ax.tick_params(colors='#94a3b8')
    ax.spines[:].set_color('#1e293b')
    ax.grid(True, linestyle='--', alpha=0.15, color='#94a3b8')
    
    # Highlight safe zone
    ax.axvspan(0, 15, color='#10b981', alpha=0.1)
    ax.axhspan(0, 10, color='#10b981', alpha=0.1)
    ax.text(1.5, 2.5, '[SECURE OPERATING ZONE]\n(FAR < 15%, MAR < 10%)\nRF, ExtraTrees, KNN, LogReg, LDA',
            color='#34d399', fontsize=9, fontweight='bold', bbox=dict(boxstyle='square', facecolor='#064e3b', alpha=0.7))
            
    plt.tight_layout()
    p4 = os.path.join(RESULTS_DIR, 'live_rsu_error_tradeoff.png')
    plt.savefig(p4, dpi=180, facecolor='#0d1321')
    plt.close()
    print(f"Saved: {p4}")


if __name__ == '__main__':
    plot_live_vs_training()
    plot_edge_tradeoff()
    plot_alpha_sensitivity()
    plot_error_tradeoff()
    print("All slide figures successfully generated!")
