"""
visualize_results.py
─────────────────────────────────────────────────────────────
Generates comparison charts from saved model_metrics.json and
saves PNGs to the results/ folder.

Usage (from project root):
  python scripts/visualize_results.py
─────────────────────────────────────────────────────────────
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE        = os.path.join(os.path.dirname(__file__), '..')
RESULTS_DIR = os.path.join(BASE, 'results')

# ── Load metrics ──────────────────────────────────────────
metrics_path = os.path.join(RESULTS_DIR, 'model_metrics.json')
if not os.path.exists(metrics_path):
    print("❌ model_metrics.json not found. Run train_model.py first.")
    exit(1)

with open(metrics_path) as f:
    data = json.load(f)

models   = data['models']
best     = data['best_model']
names    = list(models.keys())
short    = [n.replace('K-Nearest Neighbors','KNN').replace('Logistic Regression','LR')
              .replace('Gaussian Naive Bayes','GNB').replace('Gradient Boosting','GB')
              .replace('Extra Trees','ET').replace('Random Forest','RF')
              .replace('Decision Tree','DT').replace('AdaBoost','ADA')
              .replace('SVM (RBF)','SVM') for n in names]
colors   = ['#1d4ed8' if models[n]['is_best'] else '#334155' for n in names]

print("=" * 55)
print("  VeReMi — Model Comparison Charts")
print("=" * 55)
print(f"\n  Best model: {best}")

# ── Bar chart: Accuracy & F1 ──────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('VeReMi Dataset — Model Comparison', fontsize=13, fontweight='bold', color='white')
fig.patch.set_facecolor('#0d1321')
x = np.arange(len(names))

for ax, metric, title in [
    (axes[0], 'accuracy', 'Accuracy (%)'),
    (axes[1], 'f1_score', 'F1 Score (%)'),
]:
    vals = [models[n][metric] for n in names]
    bars = ax.bar(x, vals, color=colors, edgecolor='none', width=0.6)
    ax.set_facecolor('#0d1321')
    ax.set_title(title, color='white', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(short, fontsize=9, color='#94a3b8', rotation=20, ha='right')
    ax.tick_params(colors='#94a3b8')
    ax.spines[:].set_color('#1e293b')
    ax.set_ylim(max(50, min(vals) - 5), 102)
    ax.bar_label(bars, fmt='%.1f', fontsize=8, color='#cbd5e1', padding=2)

plt.tight_layout()
out1 = os.path.join(RESULTS_DIR, 'all_models_comparison.png')
plt.savefig(out1, dpi=150, facecolor='#0d1321')
plt.close()
print(f"✅ Saved → results/all_models_comparison.png")

# ── Radar chart: Top 5 ────────────────────────────────────
top5   = sorted(names, key=lambda n: -models[n]['f1_score'])[:5]
labels = ['Accuracy', 'F1 Score', 'Precision', 'Recall']
palette = ['#3b82f6', '#22c55e', '#f59e0b', '#ef4444', '#a855f7']
angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
fig.patch.set_facecolor('#0d1321')
ax.set_facecolor('#0d1321')
ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_thetagrids(np.degrees(angles[:-1]), labels, color='#94a3b8', fontsize=10)
ax.tick_params(colors='#94a3b8')

for i, name in enumerate(top5):
    m    = models[name]
    vals = [m['accuracy'], m['f1_score'], m['precision'], m['recall']]
    vals = [v / 100 for v in vals] + [vals[0] / 100]
    ax.plot(angles, vals, color=palette[i], linewidth=2 if models[name]['is_best'] else 1.5, label=name)
    ax.fill(angles, vals, color=palette[i], alpha=0.07)

ax.set_ylim(0.5, 1.0)
ax.set_title('Top 5 Models — Radar Comparison', color='white', pad=20, fontsize=12)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1),
          labelcolor='#94a3b8', facecolor='#0d1321', edgecolor='#1e293b', fontsize=9)

out2 = os.path.join(RESULTS_DIR, 'radar_comparison.png')
plt.tight_layout()
plt.savefig(out2, dpi=150, facecolor='#0d1321', bbox_inches='tight')
plt.close()
print(f"✅ Saved → results/radar_comparison.png")
print("\nDone! Open the results/ folder to view the charts.")
