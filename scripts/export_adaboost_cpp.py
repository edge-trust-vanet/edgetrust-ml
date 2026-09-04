"""
export_adaboost_cpp.py
─────────────────────────────────────────────────────────────
Exports the trained AdaBoost model and StandardScaler parameters
from edgetrust-ml/models/ into an ultra-fast, self-contained C++
header (AdaBoostPredictor.h) for embedded deployment inside
the OMNeT++ Roadside Unit (RSU).

Usage:
  python scripts/export_adaboost_cpp.py
─────────────────────────────────────────────────────────────
"""

import os
import joblib
import numpy as np

BASE = os.path.join(os.path.dirname(__file__), '..')
MODELS_DIR = os.path.join(BASE, 'models')

adaboost_path = os.path.join(MODELS_DIR, 'AdaBoost.pkl')
scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')

if not os.path.exists(adaboost_path) or not os.path.exists(scaler_path):
    print("[ERROR] Model or Scaler not found in models/. Please run preprocess.py and train_model.py first.")
    exit(1)

model = joblib.load(adaboost_path)
scaler = joblib.load(scaler_path)

mean = scaler.mean_
scale = scaler.scale_
n_features = len(mean)
n_estimators = len(model.estimators_)
weights = model.estimator_weights_
total_w = float(np.sum(weights))

stumps_code = []
for i, (est, w) in enumerate(zip(model.estimators_, weights)):
    feat = int(est.tree_.feature[0])
    thresh = float(est.tree_.threshold[0])
    val = est.tree_.value
    left_class = int(np.argmax(val[1][0]))
    right_class = int(np.argmax(val[2][0]))
    left_val = 1.0 if left_class == 1 else -1.0
    right_val = 1.0 if right_class == 1 else -1.0
    stumps_code.append(f"    {{ {feat}, {thresh:.8f}, {left_val:.1f}, {right_val:.1f}, {float(w):.8f} }}")

stumps_str = ",\n".join(stumps_code)
mean_str = ", ".join(f"{float(m):.8f}" for m in mean)
scale_str = ", ".join(f"{float(s):.8f}" for s in scale)

header_content = f"""#ifndef __ADABOOST_PREDICTOR_H_
#define __ADABOOST_PREDICTOR_H_

/**
 * AdaBoostPredictor.h
 * ─────────────────────────────────────────────────────────────
 * Auto-generated C++ classifier exported from the trained
 * scikit-learn AdaBoost model (AdaBoost.pkl) and scaler (scaler.pkl).
 * Provides sub-microsecond inference directly on RSU edge hardware.
 *
 * Features (8 inputs):
 *   0: speed (m/s)
 *   1: acceleration (m/s^2)
 *   2: position_x (m)
 *   3: position_y (m)
 *   4: direction (degrees)
 *   5: packet_drop_ratio (0.0 - 1.0)
 *   6: latency (ms)
 *   7: signal_strength (dBm)
 * ─────────────────────────────────────────────────────────────
 */

#include <cmath>

namespace veins {{

struct AdaBoostStump {{
    int featureIdx;
    double threshold;
    double leftVal;   // +1.0 for Malicious, -1.0 for Normal
    double rightVal;
    double weight;
}};

class AdaBoostPredictor {{
  public:
    static constexpr int N_FEATURES = {n_features};
    static constexpr int N_ESTIMATORS = {n_estimators};

    // Scaler parameters
    static constexpr double MEAN[{n_features}] = {{ {mean_str} }};
    static constexpr double SCALE[{n_features}] = {{ {scale_str} }};
    static constexpr double TOTAL_WEIGHT = {total_w:.8f};

    /**
     * Scale raw features to mean=0, std=1 using trained scaler parameters.
     */
    static inline void scaleFeatures(const double raw[N_FEATURES], double scaled[N_FEATURES]) {{
        for (int i = 0; i < N_FEATURES; ++i) {{
            scaled[i] = (raw[i] - MEAN[i]) / SCALE[i];
        }}
    }}

    /**
     * Evaluate AdaBoost ensemble on scaled features.
     * Returns: raw score sum(weight_m * pred_m)
     */
    static inline double evaluateRawScore(const double scaled[N_FEATURES]) {{
        static const AdaBoostStump STUMPS[N_ESTIMATORS] = {{
{stumps_str}
        }};

        double score = 0.0;
        for (int i = 0; i < N_ESTIMATORS; ++i) {{
            const AdaBoostStump& s = STUMPS[i];
            double pred = (scaled[s.featureIdx] <= s.threshold) ? s.leftVal : s.rightVal;
            score += s.weight * pred;
        }}
        return score;
    }}

    /**
     * Predict class from raw features:
     * Returns: 1 if Malicious, 0 if Normal
     */
    static inline int predict(const double raw[N_FEATURES]) {{
        double scaled[N_FEATURES];
        scaleFeatures(raw, scaled);
        double score = evaluateRawScore(scaled);
        return (score > 0.0) ? 1 : 0;
    }}

    /**
     * Predict probability of being Malicious (Class 1) [0.0 - 1.0]
     */
    static inline double predictProba(const double raw[N_FEATURES]) {{
        double scaled[N_FEATURES];
        scaleFeatures(raw, scaled);
        double score = evaluateRawScore(scaled);
        double normScore = score / TOTAL_WEIGHT;
        // Logistic sigmoid mapping of normalized margin
        return 1.0 / (1.0 + std::exp(-2.0 * normScore));
    }}
}};

}} // namespace veins

#endif // __ADABOOST_PREDICTOR_H_
"""

# Destination paths
dest_paths = [
    os.path.abspath(os.path.join(BASE, '..', 'edgetrust-test', 'omnetpp', 'veins', 'src', 'veins', 'modules', 'application', 'edgetrust', 'AdaBoostPredictor.h')),
    os.path.abspath(os.path.join(BASE, '..', 'edgetrust-core', 'simulator', 'omnetpp', 'omnetpp-6.4.0', 'samples', 'EdgeTrustSim', 'AdaBoostPredictor.h')),
    os.path.abspath(os.path.join(BASE, '..', 'edgetrust-core', 'simulator', 'omnetpp', 'EdgeTrustSim', 'AdaBoostPredictor.h'))
]

for p in dest_paths:
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, 'w') as f:
        f.write(header_content)
    print(f"[SUCCESS] Exported AdaBoost predictor to: {p}")

print("\nAdaBoost C++ export completed successfully!")
