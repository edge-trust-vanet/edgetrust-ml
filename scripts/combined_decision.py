import numpy as np
import joblib
from trust_score import calculate_trust_score, classify_vehicle

model = joblib.load('../models/best_model_rf.pkl')

def final_decision(vehicle_features, trust_data):
    """
    Combines ML prediction + Trust Score for final decision.
    vehicle_features: array of sensor features
    trust_data: dict for trust score calculation
    """
    # ML prediction
    ml_pred = model.predict([vehicle_features])[0]
    ml_prob = model.predict_proba([vehicle_features])[0][1]  # prob of malicious

    # Trust score
    trust = calculate_trust_score(trust_data)
    trust_label = classify_vehicle(trust)

    # Combined decision logic
    if trust < 0.4 and ml_pred == 1:
        decision = "BLOCK — High confidence malicious"
    elif trust < 0.4 or ml_pred == 1:
        decision = "WARN — Suspicious, reduce priority"
    else:
        decision = "ACCEPT — Trusted vehicle"

    return {
        'ml_prediction': 'Malicious' if ml_pred == 1 else 'Normal',
        'ml_confidence': round(ml_prob, 3),
        'trust_score': trust,
        'trust_label': trust_label,
        'final_decision': decision
    }

# Test
sample_features = [12.5, 0.3, 100.2, 200.5, 45.0, 10.0, 3.0]
sample_trust    = {'message_consistency': 0.2, 'behavior_history': 0.1,
                   'neighbor_validation': 0.15, 'plausibility': 0.2}

result = final_decision(sample_features, sample_trust)
print("\n=== Combined Decision Engine ===")
for k, v in result.items():
    print(f"  {k}: {v}")