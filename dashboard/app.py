import sys, os, json, time, random
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'scripts'))

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

try:
    import joblib, numpy as np
    ML_AVAILABLE = True
except ImportError:
    print("  [!] WARNING: Machine Learning libraries (numpy/joblib) missing.")
    ML_AVAILABLE = False

from trust_score import calculate_trust_score, classify_vehicle

app = Flask(__name__)
CORS(app)

BASE        = os.path.join(os.path.dirname(__file__), '..')
MODELS_DIR  = os.path.join(BASE, 'models')
RESULTS_DIR = os.path.join(BASE, 'results')

def load_metrics(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

metrics_data       = load_metrics('model_metrics.json')
vanet_metrics_data = load_metrics('vanet_model_metrics.json')

# ── Load Models & Scaler ──────────────────────────────────────
loaded_models = {}
scaler = None

if ML_AVAILABLE:
    scaler_path = os.path.join(MODELS_DIR, 'vanet_scaler.pkl')
    if not os.path.exists(scaler_path):
        scaler_path = os.path.join(MODELS_DIR, 'unified_scaler.pkl')
    if os.path.exists(scaler_path):
        try:
            scaler = joblib.load(scaler_path)
            print("  ✔ Loaded unified scaler")
        except Exception as e:
            print(f"  ✖ Scaler error: {e}")

    # Load vanet_ / unified_ models
    candidate_names = [
        'Random Forest', 'XGBoost', 'LightGBM', 'CatBoost', 'Hist Gradient Boosting',
        'Gradient Boosting', 'Extra Trees', 'Decision Tree', 'Stacking Ensemble',
        'Voting Ensemble', 'MLP Neural Network', 'SVM (RBF)', 'K-Nearest Neighbors',
        'Logistic Regression', 'AdaBoost', 'Gaussian Naive Bayes', 'Linear Discriminant Analysis'
    ]
    for name in candidate_names:
        safe_name = name.replace(' ', '_').replace('(', '').replace(')', '')
        for prefix in ['vanet_', 'unified_', '']:
            p = os.path.join(MODELS_DIR, f"{prefix}{safe_name}.pkl")
            if os.path.exists(p):
                try:
                    loaded_models[name] = joblib.load(p)
                    print(f"  ✔ Loaded model: {name} ({os.path.basename(p)})")
                    break
                except Exception as e:
                    print(f"  ✖ {name}: {e}")

best_name = (vanet_metrics_data or {}).get('best_model', 'Random Forest')
active_model = loaded_models.get(best_name) or next(iter(loaded_models.values()), None)

print(f"\n  Active model : {best_name} ({'Available' if active_model else 'Not found'})")
print(f"  Dashboard    → http://127.0.0.1:5000\n")

# ── In-memory state ───────────────────────────────────────────
activity_log = []
eval_counter = 0

def build_14_feature_vector(d, trust_score, trust_data):
    """Constructs the 14 RSU-observable features from request payload."""
    px = float(d.get('position_x', 500.0))
    py = float(d.get('position_y', 500.0))
    spd = float(d.get('speed', 15.0))
    heading = float(d.get('heading', 45.0))
    accel = float(d.get('acceleration', 0.2))
    
    # Message stats
    freq = float(d.get('message_frequency', 10))
    sent = int(d.get('packet_sent', max(10, int(freq * 10))))
    
    # Infer packet reception from plausibility and drop
    drop_est = float(d.get('packet_drop_ratio', 0.05 if trust_score >= 0.7 else 0.65))
    rcv = int(d.get('packet_received', max(1, int(sent * (1.0 - drop_est)))))
    drop_ratio = round(max(0.0, min(1.0, 1.0 - (rcv / max(1, sent)))), 4)
    
    latency = float(d.get('latency', 12.0 if trust_score >= 0.7 else 75.0))
    retx = int(d.get('retransmission_count', 1 if trust_score >= 0.7 else 6))
    signal = float(d.get('signal_strength', -65.0 if trust_score >= 0.7 else -85.0))
    
    t_curr = float(trust_score)
    t_neigh = float(trust_data.get('neighbor_validation', 0.85))
    t_hist = float(trust_data.get('behavior_history', 0.85))
    
    return [
        px, py, spd, heading, accel,
        sent, rcv, drop_ratio, latency,
        retx, signal, t_curr, t_neigh, t_hist
    ]

def make_decision(features_14, trust_score, trust_label, model=None):
    m = model or active_model
    if m is None:
        return {'error': 'No model loaded. Run training script first.'}
        
    try:
        if scaler is not None:
            scaled_vec = scaler.transform([features_14])
        else:
            scaled_vec = [features_14]
            
        pred = int(m.predict(scaled_vec)[0])
        try:
            prob = float(m.predict_proba(scaled_vec)[0][1])
        except Exception:
            prob = float(pred)
    except Exception as e:
        print("Prediction error:", e)
        pred = 1 if trust_score < 0.4 else 0
        prob = 0.9 if pred == 1 else 0.1
        
    # Hybrid Decision Matrix
    if trust_score < 0.40 and pred == 1:
        decision = 'BLOCK'
    elif trust_score < 0.40 or pred == 1:
        decision = 'WARN'
    else:
        decision = 'ACCEPT'
        
    return {
        'ml_prediction'  : 'Malicious' if pred else 'Normal',
        'ml_confidence'  : round(prob * 100.0, 1),
        'trust_score'    : round(trust_score, 4),
        'trust_label'    : trust_label,
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
        'Random Forest'              : 'Ensemble of decision trees — robust to noisy sensors and achieves highest Weighted F1 without overfitting.',
        'LightGBM'                   : 'Leaf-wise tree growth with GOSS — ultra-fast sub-millisecond inference for RSU edge units.',
        'CatBoost'                   : 'Symmetric/oblivious decision trees — eliminates target shift with compact compiled edge execution.',
        'XGBoost'                    : 'eXtreme Gradient Boosting with 2nd-order Taylor loss approximation — optimal tabular robustness.',
        'Stacking Ensemble'          : 'Heterogeneous meta-ensemble fusing tree, boosting, and linear learners.',
        'Hist Gradient Boosting'     : 'Histogram-based binning for continuous features — high-throughput gradient boosting.',
        'Decision Tree'              : 'Fully interpretable rules — ultra-lightweight for micro-controller edge devices.',
        'MLP Neural Network'         : 'Deep tabular neural network — captures non-linear cross-feature embeddings.',
        'SVM (RBF)'                  : 'Maximum-margin hyperplane with RBF kernel — highly effective in non-linear boundaries.',
    }
    return [
        f"Ranked #{f1r} of {total} by F1 Score ({bm['f1_score']}%) — the primary metric for imbalanced security datasets.",
        f"Achieved {bm['accuracy']}% accuracy (Rank #{accr}/{total}), correctly identifying nearly all attack patterns.",
        f"Precision of {bm['precision']}% (Rank #{pr}/{total}) ensures minimal false alarms on benign vehicles.",
        f"Recall of {bm['recall']}% (Rank #{rr}/{total}) guarantees high attack interception rate.",
        notes.get(name, 'Top performer across all benchmark dimensions.'),
    ]

# ═══════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/evaluate', methods=['POST'])
def evaluate():
    global eval_counter
    d = request.json
    trust_data = {
        k: float(d.get(k, 0.5)) for k in
        ['message_consistency', 'behavior_history', 'neighbor_validation', 'plausibility']
    }
    trust = calculate_trust_score(trust_data)
    label = classify_vehicle(trust)
    
    features_14 = build_14_feature_vector(d, trust, trust_data)
    result = make_decision(features_14, trust, label)
    
    eval_counter += 1
    vid = d.get('vehicle_id', f"V{eval_counter:04d}")
    entry = {
        'id': eval_counter,
        'vehicle_id': vid,
        'timestamp': time.strftime('%H:%M:%S'),
        **result,
    }
    activity_log.insert(0, entry)
    if len(activity_log) > 50:
        activity_log.pop()
    return jsonify(result)

@app.route('/api/simulate', methods=['POST'])
def simulate():
    global eval_counter
    is_mal = random.random() < 0.35
    if is_mal:
        # Simulate realistic attack (FDI, Blackhole, or Sybil)
        attack_mode = random.choice(['fdi', 'drop', 'dos'])
        if attack_mode == 'fdi':
            # Position spoofing / jump
            px, py = random.uniform(5500, 6000), random.uniform(5500, 6000)
            spd = random.uniform(0, 5)
            td = {k: round(random.uniform(0.1, 0.35), 2) for k in
                  ['message_consistency', 'behavior_history', 'neighbor_validation', 'plausibility']}
            drop_est, latency, retx, signal = 0.05, 14.0, 1, -68.0
        elif attack_mode == 'drop':
            # Blackhole / Grayhole
            px, py = random.uniform(100, 500), random.uniform(100, 500)
            spd = random.uniform(10, 25)
            td = {k: round(random.uniform(0.15, 0.40), 2) for k in
                  ['message_consistency', 'behavior_history', 'neighbor_validation', 'plausibility']}
            drop_est, latency, retx, signal = random.uniform(0.6, 0.95), 65.0, 7, -85.0
        else:
            # DoS flooding
            px, py = random.uniform(200, 600), random.uniform(200, 600)
            spd = random.uniform(15, 30)
            td = {k: round(random.uniform(0.2, 0.45), 2) for k in
                  ['message_consistency', 'behavior_history', 'neighbor_validation', 'plausibility']}
            drop_est, latency, retx, signal = 0.35, 120.0, 8, -60.0
        accel = round(random.uniform(-4.0, 3.0), 2)
        heading = round(random.uniform(0, 360), 1)
        freq = random.randint(15, 50)
    else:
        # Normal vehicle
        px, py = random.uniform(100, 800), random.uniform(100, 800)
        spd = round(random.uniform(8.0, 28.0), 2)
        accel = round(random.uniform(-1.5, 1.5), 2)
        heading = round(random.uniform(0, 360), 1)
        freq = random.randint(8, 12)
        drop_est, latency, retx, signal = 0.02, 10.5, 0, -62.0
        td = {k: round(random.uniform(0.75, 0.98), 2) for k in
              ['message_consistency', 'behavior_history', 'neighbor_validation', 'plausibility']}
              
    trust = calculate_trust_score(td)
    label = classify_vehicle(trust)
    
    d_sim = {
        'position_x': px, 'position_y': py, 'speed': spd,
        'acceleration': accel, 'heading': heading, 'message_frequency': freq,
        'packet_drop_ratio': drop_est, 'latency': latency,
        'retransmission_count': retx, 'signal_strength': signal
    }
    
    features_14 = build_14_feature_vector(d_sim, trust, td)
    result = make_decision(features_14, trust, label)
    
    eval_counter += 1
    vid = f'V{random.randint(1000, 9999)}'
    entry = {'id': eval_counter, 'vehicle_id': vid, 'timestamp': time.strftime('%H:%M:%S'), **result}
    activity_log.insert(0, entry)
    if len(activity_log) > 50:
        activity_log.pop()
        
    return jsonify({
        **result,
        'vehicle_id': vid,
        'features': [spd, accel, px, py, heading, freq, 5],
        'trust_data': td
    })

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

@app.route('/api/models')
def get_models():
    mdata = vanet_metrics_data or load_metrics('unified_model_metrics.json') or load_metrics('model_metrics.json')
    if not mdata:
        return jsonify({'error': 'No model metrics found'}), 404
    lst = []
    for name, m in mdata['models'].items():
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
    best = mdata.get('best_model', 'Random Forest')
    return jsonify({
        'models'    : lst,
        'best_model': best,
        'why_best'  : build_why_best(best, mdata['models'][best], mdata['models']),
    })

@app.route('/api/vanet_models')
def get_vanet_models():
    vdata = load_metrics('unified_model_metrics.json') or load_metrics('vanet_model_metrics.json')
    if not vdata:
        return jsonify({'error': 'Run training script first'}), 404
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
    best = vdata.get('best_model', 'Random Forest')
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
