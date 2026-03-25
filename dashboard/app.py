import sys, os, json, time, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
import joblib, numpy as np
from trust_score import calculate_trust_score, classify_vehicle

app = Flask(__name__)
CORS(app)

BASE        = os.path.join(os.path.dirname(__file__), '..')
MODELS_DIR  = os.path.join(BASE, 'models')
RESULTS_DIR = os.path.join(BASE, 'results')

# ── Load metrics JSON files ───────────────────────────────────
def load_metrics(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

metrics_data       = load_metrics('model_metrics.json')
vanet_metrics_data = load_metrics('vanet_model_metrics.json')

# ── Load VeReMi models ────────────────────────────────────────
MODEL_FILE_MAP = {
    'Random Forest'        : 'Random_Forest.pkl',
    'SVM (RBF)'            : 'SVM_RBF.pkl',
    'K-Nearest Neighbors'  : 'K-Nearest_Neighbors.pkl',
    'Logistic Regression'  : 'Logistic_Regression.pkl',
    'Decision Tree'        : 'Decision_Tree.pkl',
    'Gaussian Naive Bayes' : 'Gaussian_Naive_Bayes.pkl',
    'AdaBoost'             : 'AdaBoost.pkl',
    'Gradient Boosting'    : 'Gradient_Boosting.pkl',
    'Extra Trees'          : 'Extra_Trees.pkl',
}
loaded_models = {}
for name, fname in MODEL_FILE_MAP.items():
    p = os.path.join(MODELS_DIR, fname)
    if os.path.exists(p):
        try:
            loaded_models[name] = joblib.load(p)
            print(f"  \u2714 Loaded: {name}")
        except Exception as e:
            print(f"  \u2718 {name}: {e}")

best_name    = (metrics_data or {}).get('best_model', 'Random Forest')
active_model = loaded_models.get(best_name) or next(iter(loaded_models.values()), None)

print(f"\n  Active model : {best_name}")
print(f"  Dashboard    → http://127.0.0.1:5000\n")

# ── In-memory state ───────────────────────────────────────────
activity_log = []
eval_counter = 0

# ── Decision Logic ────────────────────────────────────────────
def make_decision(features, trust_data, model=None):
    m = model or active_model
    if m is None:
        return {'error': 'No model loaded. Run train_model.py first.'}
    pred = m.predict([features])[0]
    try:
        prob = m.predict_proba([features])[0][1]
    except Exception:
        prob = float(pred)
    trust = calculate_trust_score(trust_data)
    label = classify_vehicle(trust)
    if trust < 0.4 and pred == 1:
        decision = 'BLOCK'
    elif trust < 0.4 or pred == 1:
        decision = 'WARN'
    else:
        decision = 'ACCEPT'
    return {
        'ml_prediction'  : 'Malicious' if pred else 'Normal',
        'ml_confidence'  : round(float(prob) * 100, 1),
        'trust_score'    : trust,
        'trust_label'    : label,
        'final_decision' : decision,
    }

def build_why_best(name, bm, all_m):
    total = len(all_m)
    others = {k: v for k, v in all_m.items() if k != name}
    f1r  = sum(1 for v in others.values() if v['f1_score']  > bm['f1_score'])  + 1
    accr = sum(1 for v in others.values() if v['accuracy']  > bm['accuracy'])  + 1
    pr   = sum(1 for v in others.values() if v['precision'] > bm['precision']) + 1
    rr   = sum(1 for v in others.values() if v['recall']    > bm['recall'])    + 1
    notes = {
        'Random Forest'        : 'Ensemble of 100 decision trees — handles noisy sensor data robustly and resists overfitting.',
        'Extra Trees'          : 'Extremely randomized trees — faster training with equally robust generalization.',
        'Gradient Boosting'    : 'Iterative boosting corrects errors each round — excellent at detecting subtle attacks.',
        'AdaBoost'             : 'Weights misclassified examples more each round — strong on borderline cases.',
        'SVM (RBF)'            : 'Maximum-margin hyperplane with RBF kernel — highly effective in high-dimensional feature space.',
        'K-Nearest Neighbors'  : 'Non-parametric — classifies based on neighborhood behavior in feature space.',
        'Logistic Regression'  : 'Linear, highly interpretable — strong baseline for binary classification.',
        'Decision Tree'        : 'Fully interpretable — rules can be extracted for deployment on resource-constrained RSUs.',
        'Gaussian Naive Bayes' : 'Probabilistic — extremely lightweight for resource-constrained roadside units.',
    }
    return [
        f"Ranked #{f1r} of {total} by F1 Score ({bm['f1_score']}%) — the primary metric for imbalanced security datasets.",
        f"Achieved {bm['accuracy']}% accuracy (Rank #{accr}/{total}), correctly identifying nearly all attack patterns.",
        f"Precision of {bm['precision']}% (Rank #{pr}/{total}) means very few legitimate vehicles are falsely flagged.",
        f"Recall of {bm['recall']}% (Rank #{rr}/{total}) ensures malicious vehicles are almost never missed.",
        notes.get(name, 'Strong performer across all key metrics.'),
    ]

# ═══════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')

# ── Evaluate ──────────────────────────────────────────────────
@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    global eval_counter
    d = request.json
    features = [
        float(d['speed']), float(d['acceleration']),
        float(d['position_x']), float(d['position_y']),
        float(d['heading']), float(d['message_frequency']),
        float(d['neighbor_count']),
    ]
    trust_data = {k: float(d[k]) for k in
                  ['message_consistency', 'behavior_history', 'neighbor_validation', 'plausibility']}
    result = make_decision(features, trust_data)
    eval_counter += 1
    entry = {
        'id': eval_counter,
        'vehicle_id': d.get('vehicle_id', f'V{eval_counter:04d}'),
        'timestamp': time.strftime('%H:%M:%S'),
        **result,
    }
    activity_log.insert(0, entry)
    if len(activity_log) > 50:
        activity_log.pop()
    return jsonify(result)

# ── Simulate ──────────────────────────────────────────────────
@app.route('/api/simulate', methods=['POST'])
def simulate():
    global eval_counter
    is_mal = random.random() < 0.45
    if is_mal:
        features = [
            random.uniform(80, 200), random.uniform(5, 20),
            random.uniform(-500, 500), random.uniform(-500, 500),
            random.uniform(0, 360), random.randint(30, 100), random.randint(0, 2),
        ]
        td = {k: round(random.uniform(0.0, 0.35), 2) for k in
              ['message_consistency', 'behavior_history', 'neighbor_validation', 'plausibility']}
    else:
        features = [
            random.uniform(5, 60), random.uniform(0, 2),
            random.uniform(100, 400), random.uniform(100, 400),
            random.uniform(0, 360), random.randint(1, 10), random.randint(3, 15),
        ]
        td = {k: round(random.uniform(0.7, 1.0), 2) for k in
              ['message_consistency', 'behavior_history', 'neighbor_validation', 'plausibility']}
    result = make_decision(features, td)
    eval_counter += 1
    vid = f'V{random.randint(1000, 9999)}'
    entry = {'id': eval_counter, 'vehicle_id': vid, 'timestamp': time.strftime('%H:%M:%S'), **result}
    activity_log.insert(0, entry)
    if len(activity_log) > 50:
        activity_log.pop()
    return jsonify({**result, 'vehicle_id': vid, 'features': features, 'trust_data': td})

@app.route('/api/log')
def get_log():
    return jsonify(activity_log[:20])

@app.route('/api/stats')
def get_stats():
    return jsonify({
        'total'   : eval_counter,
        'blocked' : sum(1 for e in activity_log if e['final_decision'] == 'BLOCK'),
        'warned'  : sum(1 for e in activity_log if e['final_decision'] == 'WARN'),
        'accepted': sum(1 for e in activity_log if e['final_decision'] == 'ACCEPT'),
    })

# ── VeReMi Models ─────────────────────────────────────────────
@app.route('/api/models')
def get_models():
    if not metrics_data:
        return jsonify({'error': 'Run scripts/train_model.py first'}), 404
    lst = []
    for name, m in metrics_data['models'].items():
        lst.append({
            'name'            : name,
            'accuracy'        : m['accuracy'],
            'f1_score'        : m['f1_score'],
            'precision'       : m['precision'],
            'recall'          : m['recall'],
            'false_alert_rate': m.get('false_alert_rate', 0),
            'confusion_matrix': m['confusion_matrix'],
            'is_best'         : m.get('is_best', False),
            'loaded'          : name in loaded_models,
        })
    lst.sort(key=lambda x: -x['f1_score'])
    best = metrics_data['best_model']
    return jsonify({
        'models'    : lst,
        'best_model': best,
        'why_best'  : build_why_best(best, metrics_data['models'][best], metrics_data['models']),
    })

# ── VANET Nodes Models ────────────────────────────────────────
@app.route('/api/vanet_models')
def get_vanet_models():
    # Re-read from disk on every request so the dashboard reflects the
    # latest training run without needing a server restart.
    vdata = load_metrics('vanet_model_metrics.json')
    if not vdata:
        return jsonify({'error': 'Run scripts/analyze_vanet_nodes.py first'}), 404
    lst = []
    for name, m in vdata['models'].items():
        lst.append({
            'name'            : name,
            'accuracy'        : m['accuracy'],
            'f1_score'        : m['f1_score'],
            'precision'       : m['precision'],
            'recall'          : m['recall'],
            'false_alert_rate': m.get('false_alert_rate', 0),
            'confusion_matrix': m['confusion_matrix'],
            'is_best'         : m.get('is_best', False),
        })
    lst.sort(key=lambda x: -x['f1_score'])
    best = vdata['best_model']
    fi   = vdata.get('feature_importance', {})
    return jsonify({
        'models'            : lst,
        'best_model'        : best,
        'feature_importance': fi,
        'why_best'          : build_why_best(
            best,
            vdata['models'][best],
            vdata['models'],
        ),
    })

if __name__ == '__main__':
    app.run(debug=False, port=5000, threaded=True)
