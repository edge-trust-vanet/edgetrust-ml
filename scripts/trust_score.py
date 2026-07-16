

def calculate_trust_score(vehicle_data):
    """
    vehicle_data: dict with keys:
      - message_consistency: float (0-1), how consistent messages are
      - behavior_history: float (0-1), past good behavior ratio
      - neighbor_validation: float (0-1), % of neighbors who confirmed the message
      - plausibility: float (0-1), physically possible? (speed, location checks)
    Returns: trust_score between 0 and 1
    """
    # Weights — you can tune these
    w1 = 0.30   # message consistency
    w2 = 0.25   # behavior history
    w3 = 0.25   # neighbor validation
    w4 = 0.20   # plausibility check

    score = (
        w1 * vehicle_data['message_consistency'] +
        w2 * vehicle_data['behavior_history'] +
        w3 * vehicle_data['neighbor_validation'] +
        w4 * vehicle_data['plausibility']
    )
    return round(score, 4)


def classify_vehicle(trust_score, threshold=0.5):
    if trust_score >= 0.75:
        return "Trusted"
    elif trust_score >= threshold:
        return "Neutral"
    else:
        return "Malicious"


# --- Test it with dummy vehicles ---
test_vehicles = [
    {'id': 'V001', 'message_consistency': 0.9, 'behavior_history': 0.85, 
     'neighbor_validation': 0.92, 'plausibility': 0.95},
    {'id': 'V002', 'message_consistency': 0.2, 'behavior_history': 0.1,
     'neighbor_validation': 0.15, 'plausibility': 0.3},
    {'id': 'V003', 'message_consistency': 0.6, 'behavior_history': 0.55,
     'neighbor_validation': 0.5, 'plausibility': 0.65},
]

print("=== Trust Score Evaluation ===")
for v in test_vehicles:
    score = calculate_trust_score(v)
    label = classify_vehicle(score)
    print(f"Vehicle {v['id']}: Trust Score = {score} → {label}")