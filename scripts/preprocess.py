"""
preprocess.py
─────────────────────────────────────────────────────────────
Loads veremi_dataset.csv, scales features, and saves train/test
splits as .npy files for use by train_model.py.

Usage:
  python scripts/preprocess.py
─────────────────────────────────────────────────────────────
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib, os

BASE     = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(BASE, 'data')
MDL_DIR  = os.path.join(BASE, 'models')
os.makedirs(MDL_DIR, exist_ok=True)

print("=" * 55)
print("  VeReMi Dataset — Preprocessing")
print("=" * 55)

csv_path = os.path.join(DATA_DIR, 'vanet_malicious_nodes.csv')
if not os.path.exists(csv_path):
    print(f"\n[ERROR] Dataset not found at {csv_path}")
    print("   Run: python scripts/generate_realistic_dataset.py first")
    exit(1)

df = pd.read_csv(csv_path)
print(f"\n[INFO] Loaded: {df.shape[0]} rows × {df.shape[1]} columns")

FEATURES = ['speed', 'acceleration', 'position_x', 'position_y',
            'direction', 'packet_drop_ratio', 'latency', 'signal_strength']
TARGET   = 'is_malicious'

df = df.dropna(subset=FEATURES + [TARGET])
X  = df[FEATURES].values
y  = df[TARGET].values

print(f"   Normal    : {(y == 0).sum()} ({(y == 0).mean()*100:.1f}%)")
print(f"   Malicious : {(y == 1).sum()} ({(y == 1).mean()*100:.1f}%)")
print(f"   Features  : {FEATURES}")

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Scale
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# Save
np.save(os.path.join(DATA_DIR, 'X_train.npy'), X_train)
np.save(os.path.join(DATA_DIR, 'X_test.npy'),  X_test)
np.save(os.path.join(DATA_DIR, 'y_train.npy'), y_train)
np.save(os.path.join(DATA_DIR, 'y_test.npy'),  y_test)
joblib.dump(scaler, os.path.join(MDL_DIR, 'scaler.pkl'))

print(f"\n   Train : {X_train.shape}")
print(f"   Test  : {X_test.shape}")
print(f"\n[SUCCESS] Saved X_train, X_test, y_train, y_test → data/")
print(f"[SUCCESS] Saved scaler → models/scaler.pkl")
print("\nNext step: python scripts/train_model.py")
