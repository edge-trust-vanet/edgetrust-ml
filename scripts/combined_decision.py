import numpy as np
import joblib
import os
from trust_score import HybridTrustEngine

# Safely resolve paths
BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
model_path = os.path.join(BASE_DIR, 'models', 'Random_Forest_GridSearch.pkl')
scaler_path = os.path.join(BASE_DIR, 'models', 'scaler.pkl')

try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
except FileNotFoundError:
    print("Model or Scaler not found. Please run preprocess.py and train_model.py first.")
    exit(1)

# Initialize the Hybrid Trust Engine (alpha=0.7 as proven)
trust_engine = HybridTrustEngine(alpha=0.7)

def evaluate_packet(vehicle_id, vehicle_features):
    """
    Evaluates a single packet using the Hybrid ML Architecture.
    """
    # 0. Scale the features using the trained StandardScaler!
    scaled_features = scaler.transform([vehicle_features])
    
    # 1. Machine Learning Prediction (Probability of Honest Behavior)
    ml_probs = model.predict_proba(scaled_features)[0]
    
    # Probability of being Honest (Class 0)
    honest_prob = ml_probs[0]
    
    # 2. Historical Trust Update (EMA)
    current_trust = trust_engine.update_trust(vehicle_id, honest_prob)
    
    # 3. Final Classification
    verdict = trust_engine.classify_vehicle(current_trust)
    
    return {
        'ml_honest_prob': round(honest_prob, 4),
        'trust_score': current_trust,
        'verdict': verdict
    }

# --- Test the Hybrid Engine ---
if __name__ == "__main__":
    print("\n=== Hybrid Decision Engine Simulation ===")
    vid = "V_ATTACKER"
    
    # Dummy features representing a spoofed packet 
    # Features: ['speed', 'acceleration', 'position_x', 'position_y', 'direction', 'packet_drop_ratio', 'latency', 'signal_strength']
    # Using exact features from Node 5 in vanet_malicious_nodes.csv (is_malicious = 1)
    dummy_malicious_packet = [14.298, -0.107, 156.018, 869.649, 66.808, 0.224, 41.015, -92.987] 
    
    # If the model expects a different number of features, we handle it safely:
    expected_features = model.n_features_in_
    if len(dummy_malicious_packet) != expected_features:
        dummy_malicious_packet = np.random.rand(expected_features).tolist()
    
    print(f"\nSimulating incoming malicious packets for {vid}...")
    for packet_idx in range(1, 8):
        res = evaluate_packet(vid, dummy_malicious_packet)
        print(f"Packet {packet_idx} | ML Evidence (Honest): {res['ml_honest_prob']} | Trust Score: {res['trust_score']:.4f} -> {res['verdict']}")