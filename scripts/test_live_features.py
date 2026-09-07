"""
test_live_features.py
─────────────────────────────────────────────────────────────
Evaluates the trained EdgeTrust ML models against the live
telemetry extracted from the OMNeT++ / SUMO simulation.

Usage:
  python scripts/test_live_features.py [--input data/live_extracted_features.csv]
─────────────────────────────────────────────────────────────
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report
)

BASE = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(BASE, 'data')
MODELS_DIR = os.path.join(BASE, 'models')
RESULTS_DIR = os.path.join(BASE, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Test ML models against live extracted VANET features")
    parser.add_argument(
        '--input',
        default=os.path.join(DATA_DIR, 'live_extracted_features.csv'),
        help='Path to the live extracted CSV file'
    )
    parser.add_argument(
        '--model',
        default=None,
        help='Specific model file name to test (e.g., Random_Forest_GridSearch.pkl)'
    )
    return parser.parse_args()


def main():
    args = parse_args()
    csv_path = args.input

    print("=" * 65)
    print("  EdgeTrust-VANET — Live Simulation ML Evaluation")
    print("=" * 65)

    if not os.path.exists(csv_path):
        print(f"\n[ERROR] Extracted feature file not found at: {csv_path}")
        print("Please run the OMNeT++ / SUMO simulation first to extract live telemetry.")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"\n[INFO] Loaded live simulation features: {df.shape[0]} records, {df.shape[1]} columns")

    # Display dataset statistics
    print("\n" + "─" * 65)
    print("  📊 Live Simulation Telemetry Summary")
    print("─" * 65)

    unique_nodes = df['node_id'].nunique() if 'node_id' in df.columns else 'N/A'
    mal_count = int((df['is_malicious'] == 1).sum()) if 'is_malicious' in df.columns else 0
    benign_count = int((df['is_malicious'] == 0).sum()) if 'is_malicious' in df.columns else 0

    print(f"  • Total Packets Extracted : {len(df)}")
    print(f"  • Unique Vehicles Monitored: {unique_nodes}")
    print(f"  • Benign Telemetry Records: {benign_count} ({(benign_count/len(df)*100):.1f}%)")
    print(f"  • Malicious Telemetry     : {mal_count} ({(mal_count/len(df)*100):.1f}%)")

    if 'false_packet_injection' in df.columns:
        print(f"  • Detected FDI Attempts   : {df['false_packet_injection'].sum()}")
        print(f"  • Blackhole Attack Events : {df['blackhole_attack_attempts'].sum()}")
        print(f"  • Sybil Attack Attempts   : {df['sybil_attack_attempts'].sum()}")
        print(f"  • DoS Flood Records       : {df['denial_of_service'].sum()}")

    # Kinematics & RF summary
    print(f"  • Speed Range (m/s)       : {df['speed'].min():.2f} - {df['speed'].max():.2f}")
    print(f"  • Acceleration Range(m/s²): {df['acceleration'].min():.2f} - {df['acceleration'].max():.2f}")
    print(f"  • RSSI Signal (dBm)       : {df['signal_strength'].min():.2f} - {df['signal_strength'].max():.2f}")
    print(f"  • Avg Trust Score         : {df['trust_score'].mean():.4f}")

    # Load Scaler
    scaler_path = os.path.join(MODELS_DIR, 'scaler.pkl')
    vanet_scaler_path = os.path.join(MODELS_DIR, 'vanet_scaler.pkl')

    scaler = None
    scaler_features = None

    # Model features definitions
    VEREMI_FEATURES = [
        'speed', 'acceleration', 'position_x', 'position_y',
        'direction', 'packet_drop_ratio', 'latency', 'signal_strength'
    ]

    VANET_FEATURES = [
        'position_x', 'position_y', 'speed', 'direction', 'acceleration',
        'packet_sent', 'packet_received', 'packet_drop_ratio', 'latency',
        'message_retransmission_count', 'signal_strength',
        'trust_score', 'neighbor_trust_score_avg', 'historical_trust_score'
    ]

    # Look for available trained models
    model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.pkl') and f != 'scaler.pkl' and f != 'vanet_scaler.pkl'] if os.path.exists(MODELS_DIR) else []

    if not model_files:
        print(f"\n[WARN] No trained models found in {MODELS_DIR}.")
        print("Training Random Forest baseline on available dataset for immediate verification...")
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler

        train_csv = os.path.join(DATA_DIR, 'vanet_malicious_nodes.csv')
        df_train = pd.read_csv(train_csv)
        X_train_raw = df_train[VEREMI_FEATURES].values
        y_train_raw = df_train['is_malicious'].values

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_raw)

        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train_raw)
        joblib.dump(scaler, scaler_path)
        joblib.dump(rf_model, os.path.join(MODELS_DIR, 'Random_Forest.pkl'))
        model_files = ['Random_Forest.pkl']
        print("[SUCCESS] Trained and saved Random Forest model.")

    print("\n" + "─" * 65)
    print("  🤖 Evaluating ML Models on Live SUMO/OMNeT++ Data")
    print("─" * 65)

    # Determine which feature set matches the model
    # First try default scaler
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
        if getattr(scaler, 'n_features_in_', len(VEREMI_FEATURES)) == len(VEREMI_FEATURES):
            FEATURES = VEREMI_FEATURES
        else:
            FEATURES = VANET_FEATURES
    elif os.path.exists(vanet_scaler_path):
        scaler = joblib.load(vanet_scaler_path)
        FEATURES = VANET_FEATURES
    else:
        FEATURES = VEREMI_FEATURES

    X_live = df[FEATURES].values
    y_live = df['is_malicious'].values if 'is_malicious' in df.columns else None

    # Scale live features
    try:
        X_scaled = scaler.transform(X_live)
    except Exception as e:
        print(f"[WARN] Scaler dimension mismatch ({e}), refitting scaler on live data features...")
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_live)

    all_evaluations = {}

    for mf in sorted(model_files):
        mpath = os.path.join(MODELS_DIR, mf)
        try:
            m = joblib.load(mpath)
        except Exception:
            continue

        # Check if feature count matches model expectations
        expected_n = getattr(m, 'n_features_in_', None)
        if expected_n is not None and expected_n != X_scaled.shape[1]:
            # Adjust feature set if needed
            if expected_n == len(VANET_FEATURES) and set(VANET_FEATURES).issubset(df.columns):
                cur_X = df[VANET_FEATURES].values
                from sklearn.preprocessing import StandardScaler
                cur_scaler = StandardScaler()
                cur_X_scaled = cur_scaler.fit_transform(cur_X)
            elif expected_n == len(VEREMI_FEATURES):
                cur_X = df[VEREMI_FEATURES].values
                from sklearn.preprocessing import StandardScaler
                cur_scaler = StandardScaler()
                cur_X_scaled = cur_scaler.fit_transform(cur_X)
            else:
                continue
        else:
            cur_X_scaled = X_scaled

        preds = m.predict(cur_X_scaled)
        try:
            probs = m.predict_proba(cur_X_scaled)[:, 1]
        except Exception:
            probs = preds.astype(float)

        name = mf.replace('.pkl', '').replace('_', ' ')

        if y_live is not None and len(np.unique(y_live)) > 1:
            acc = accuracy_score(y_live, preds) * 100.0
            f1 = f1_score(y_live, preds, average='weighted', zero_division=0) * 100.0
            prec = precision_score(y_live, preds, average='weighted', zero_division=0) * 100.0
            rec = recall_score(y_live, preds, average='weighted', zero_division=0) * 100.0
            cm = confusion_matrix(y_live, preds).tolist()

            all_evaluations[name] = {
                'accuracy': round(acc, 2),
                'f1_score': round(f1, 2),
                'precision': round(prec, 2),
                'recall': round(rec, 2),
                'confusion_matrix': cm
            }

            print(f"  ✓ {name:<26} : Acc={acc:6.2f}% | F1={f1:6.2f}% | Prec={prec:6.2f}% | Rec={rec:6.2f}%")
        else:
            mal_detected = int((preds == 1).sum())
            norm_detected = int((preds == 0).sum())
            print(f"  ✓ {name:<26} : Detected {norm_detected} Normal, {mal_detected} Malicious")

    # Sample table of live decisions (top 15 packets)
    best_model_name = next(iter(all_evaluations.keys())) if all_evaluations else (model_files[0].replace('.pkl', '') if model_files else None)
    if best_model_name:
        active_m = joblib.load(os.path.join(MODELS_DIR, best_model_name.replace(' ', '_') + '.pkl'))
        sample_preds = active_m.predict(X_scaled)
        try:
            sample_probs = active_m.predict_proba(X_scaled)[:, 1]
        except Exception:
            sample_probs = sample_preds.astype(float)

        print("\n" + "─" * 65)
        print(f"  🔍 Live Vehicle Classification Samples (Model: {best_model_name})")
        print("─" * 65)
        print(f"{'Node':<6} {'Speed':<8} {'RSSI(dBm)':<11} {'Trust':<8} {'True':<7} {'Pred':<8} {'Conf':<8} {'Decision'}")
        print("─" * 65)

        indices = np.linspace(0, len(df) - 1, min(15, len(df)), dtype=int)
        for idx in indices:
            row = df.iloc[idx]
            nid = int(row['node_id'])
            spd = f"{row['speed']:.1f} m/s"
            rssi = f"{row['signal_strength']:.1f}"
            trust = f"{row['trust_score']:.2f}"
            y_true = "Mal" if row.get('is_malicious', 0) == 1 else "Norm"
            y_pred = "Mal" if sample_preds[idx] == 1 else "Norm"
            conf = f"{sample_probs[idx]*100:.1f}%" if sample_preds[idx] == 1 else f"{(1-sample_probs[idx])*100:.1f}%"

            # Hybrid Decision Engine Logic
            t_val = row['trust_score']
            if t_val < 0.40 and sample_preds[idx] == 1:
                dec = "BLOCK"
            elif t_val < 0.40 or sample_preds[idx] == 1:
                dec = "WARN"
            else:
                dec = "ACCEPT"

            print(f"V{nid:<5} {spd:<8} {rssi:<11} {trust:<8} {y_true:<7} {y_pred:<8} {conf:<8} {dec}")

    # Save results JSON
    results_out = {
        'total_extracted_records': len(df),
        'unique_nodes': unique_nodes,
        'benign_count': benign_count,
        'malicious_count': mal_count,
        'evaluations': all_evaluations
    }

    results_file = os.path.join(RESULTS_DIR, 'live_evaluation_results.json')
    with open(results_file, 'w') as f:
        json.dump(results_out, f, indent=2)

    print("\n" + "=" * 65)
    print(f"✅ Evaluation complete! Results saved to: results/live_evaluation_results.json")
    print("=" * 65)


if __name__ == '__main__':
    main()
