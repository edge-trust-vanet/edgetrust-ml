class HybridTrustEngine:
    def __init__(self, alpha=0.7):
        """
        Initializes the Hybrid Trust Engine.
        :param alpha: Forgetting factor (0 to 1). Higher means more weight to historical trust.
        """
        self.alpha = alpha
        self.trust_database = {}  # Dictionary to hold historical trust scores

    def update_trust(self, vehicle_id, evidence_score):
        """
        Updates the trust score for a vehicle using Exponential Moving Average.
        :param vehicle_id: Unique identifier for the vehicle
        :param evidence_score: Et (Probability of Honest behavior from ML model)
        :return: Updated Trust Score
        """
        # If vehicle is new, assume initial trust of 1.0
        if vehicle_id not in self.trust_database:
            self.trust_database[vehicle_id] = 1.0 
        
        t_prev = self.trust_database[vehicle_id]
        
        # Exponential Moving Average Formula
        t_new = (self.alpha * t_prev) + ((1 - self.alpha) * evidence_score)
        
        # Save updated trust
        self.trust_database[vehicle_id] = round(t_new, 4)
        return self.trust_database[vehicle_id]

    def classify_vehicle(self, trust_score):
        """
        Classifies a vehicle based on its current trust score.
        """
        if trust_score >= 0.70:
            return "Trusted"
        elif trust_score >= 0.40:
            return "Suspicious"
        else:
            return "Blocked"


def calculate_trust_score(trust_data):
    """
    Computes weighted composite trust score from real-time trust factors:
    - message_consistency (weight: 0.30)
    - behavior_history (weight: 0.30)
    - neighbor_validation (weight: 0.20)
    - plausibility (weight: 0.20)
    """
    weights = {
        'message_consistency': 0.30,
        'behavior_history': 0.30,
        'neighbor_validation': 0.20,
        'plausibility': 0.20,
    }
    total = sum(float(trust_data.get(k, 0.5)) * w for k, w in weights.items())
    return round(float(total), 4)


def classify_vehicle(trust_score):
    """
    Classifies a vehicle based on its current trust score threshold.
    """
    if trust_score >= 0.70:
        return "Trusted"
    elif trust_score >= 0.40:
        return "Suspicious"
    else:
        return "Blocked"

# --- Test functionality ---
if __name__ == "__main__":
    engine = HybridTrustEngine(alpha=0.7)
    test_vehicle = "V001"
    
    print(f"=== Hybrid Trust Engine Test (Vehicle {test_vehicle}) ===")
    print("Simulating a spoofing attack where ML Evidence is consistently 0.1")
    
    # Simulate receiving 5 malicious packets
    for i in range(1, 6):
        evidence_score = 0.1  # Random Forest predicts it is malicious
        current_trust = engine.update_trust(test_vehicle, evidence_score)
        label = engine.classify_vehicle(current_trust)
        print(f"Packet {i}: Evidence = {evidence_score} | Trust = {current_trust:.4f} -> {label}")