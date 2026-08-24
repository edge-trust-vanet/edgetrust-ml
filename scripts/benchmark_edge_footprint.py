"""
benchmark_edge_footprint.py
─────────────────────────────────────────────────────────────
Benchmarks memory footprint, serialization payload size, and
inference execution latency across candidate classifiers targeted
for Roadside Unit (RSU) and On-Board Unit (OBU) edge deployments.

Target Edge Hardware Profiles:
  - Micro-RSU: ARM Cortex-A72 / STM32F7 (Resource-Constrained)
  - Standard RSU: Raspberry Pi 4 Model B (4 GB RAM)
  - Central Fog: x86-64 Server / NVIDIA Jetson Orin Nano

Usage:
  python scripts/benchmark_edge_footprint.py [--runs 1000]
─────────────────────────────────────────────────────────────
"""

import os
import sys
import time
import argparse
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
import joblib

BASE = os.path.join(os.path.dirname(__file__), '..')
MODELS_DIR = os.path.join(BASE, 'models')
RESULTS_DIR = os.path.join(BASE, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark ML Edge Footprint and Throughput")
    parser.add_argument('--runs', type=int, default=1000, help='Inference timing iterations')
    parser.add_argument(
        '--output',
        default=os.path.join(RESULTS_DIR, 'edge_footprint_benchmark.json'),
        help='Output path for JSON benchmark results'
    )
    return parser.parse_args()


def benchmark_model(name, clf, X_sample, runs=1000):
    """
    Measures serialization size and single-packet inference latency.
    """
    tmp_path = os.path.join(RESULTS_DIR, f'_temp_{name}.pkl')
    joblib.dump(clf, tmp_path, compress=3)
    model_size_bytes = os.path.getsize(tmp_path)
    if os.path.exists(tmp_path):
        os.remove(tmp_path)

    for _ in range(50):
        _ = clf.predict(X_sample)

    latencies_us = []
    for _ in range(runs):
        idx = np.random.randint(0, len(X_sample))
        x = X_sample[idx:idx+1]
        t0 = time.perf_counter()
        _ = clf.predict(x)
        t1 = time.perf_counter()
        latencies_us.append((t1 - t0) * 1e6)

    mean_us = float(np.round(np.mean(latencies_us), 2))
    p95_us = float(np.round(np.percentile(latencies_us, 95), 2))
    p99_us = float(np.round(np.percentile(latencies_us, 99), 2))
    throughput_pps = float(np.round(1e6 / mean_us, 1)) if mean_us > 0 else 0.0

    return {
        'model_name': name,
        'size_bytes': int(model_size_bytes),
        'size_kb': float(np.round(model_size_bytes / 1024, 2)),
        'mean_latency_us': mean_us,
        'p95_latency_us': p95_us,
        'p99_latency_us': p99_us,
        'throughput_pps': throughput_pps,
        'edge_rsu_suitability': 'High' if mean_us < 200 and model_size_bytes < 500000 else 'Moderate'
    }


def main():
    args = parse_args()
    print("=" * 68)
    print("  EdgeTrust-VANET — Edge RSU Memory Footprint & Throughput Suite")
    print("=" * 68)

    np.random.seed(42)
    n_samples = 500
    n_features = 14
    X = np.random.randn(n_samples, n_features)
    y = np.random.choice([0, 1], p=[0.7, 0.3], size=n_samples)

    models = {
        'Logistic_Regression': LogisticRegression(max_iter=500),
        'Decision_Tree': DecisionTreeClassifier(max_depth=6),
        'Gaussian_NB': GaussianNB(),
        'AdaBoost': AdaBoostClassifier(n_estimators=30, random_state=42),
        'Random_Forest': RandomForestClassifier(n_estimators=40, max_depth=10, random_state=42)
    }

    results = []
    print(f"\nBenchmarking {len(models)} models with {args.runs} single-packet iterations...")
    print(f"{'Model':<22} | {'Size (KB)':<10} | {'Latency (us)':<14} | {'P95 (us)':<10} | {'Throughput (pkt/s)':<18}")
    print("-" * 84)

    for name, clf in models.items():
        clf.fit(X, y)
        metrics = benchmark_model(name, clf, X, runs=args.runs)
        results.append(metrics)
        print(f"{name:<22} | {metrics['size_kb']:<10.1f} | {metrics['mean_latency_us']:<14.2f} | {metrics['p95_latency_us']:<10.2f} | {metrics['throughput_pps']:<18.1f}")

    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n[INFO] Edge footprint benchmark saved to: {args.output}")


if __name__ == '__main__':
    main()
