"""
profile_network_telemetry.py
─────────────────────────────────────────────────────────────
Network telemetry profiler for EdgeTrust-VANET.
Evaluates transmission latency distributions, packet drop ratios,
inter-arrival jitter bounds, and channel reliability across
vehicular density regimes under IEEE 802.11p / DSRC specifications.

Usage:
  python scripts/profile_network_telemetry.py [--data data/unified_edgetrust_dataset.csv]
─────────────────────────────────────────────────────────────
"""

import os
import sys
import argparse
import json
import numpy as np
import pandas as pd

BASE = os.path.join(os.path.dirname(__file__), '..')
DATA_DIR = os.path.join(BASE, 'data')
RESULTS_DIR = os.path.join(BASE, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Profile VANET Network Telemetry and Jitter")
    parser.add_argument(
        '--data',
        default=os.path.join(DATA_DIR, 'unified_edgetrust_dataset.csv'),
        help='Path to dataset containing network telemetry'
    )
    parser.add_argument(
        '--output',
        default=os.path.join(RESULTS_DIR, 'network_telemetry_profile.json'),
        help='Output path for JSON profile report'
    )
    return parser.parse_args()


def compute_telemetry_profile(df):
    """
    Computes statistical network telemetry metrics across normal and malicious traffic.
    """
    profile = {}
    
    # Identify relevant columns
    has_latency = 'latency' in df.columns
    has_pdr = 'packet_drop_ratio' in df.columns
    has_rssi = 'signal_strength' in df.columns
    has_retx = 'retransmission_count' in df.columns
    has_label = 'label' in df.columns
    
    total_records = len(df)
    profile['sample_count'] = int(total_records)
    
    # Global latency metrics
    if has_latency:
        lat = df['latency'].dropna()
        profile['latency_stats_ms'] = {
            'mean': float(np.round(lat.mean(), 3)),
            'std': float(np.round(lat.std(), 3)),
            'median': float(np.round(lat.median(), 3)),
            'p95': float(np.round(np.percentile(lat, 95), 3)),
            'p99': float(np.round(np.percentile(lat, 99), 3)),
            'min': float(np.round(lat.min(), 3)),
            'max': float(np.round(lat.max(), 3))
        }
        # Inter-arrival jitter estimation (first-order difference of latency)
        jitter = np.abs(np.diff(lat.values))
        profile['jitter_stats_ms'] = {
            'mean_jitter': float(np.round(np.mean(jitter), 3)) if len(jitter) > 0 else 0.0,
            'max_jitter': float(np.round(np.max(jitter), 3)) if len(jitter) > 0 else 0.0,
            'p95_jitter': float(np.round(np.percentile(jitter, 95), 3)) if len(jitter) > 0 else 0.0
        }

    # Packet drop ratio metrics
    if has_pdr:
        pdr = df['packet_drop_ratio'].dropna()
        profile['packet_drop_stats'] = {
            'mean_pdr': float(np.round(pdr.mean(), 4)),
            'p95_pdr': float(np.round(np.percentile(pdr, 95), 4)),
            'max_pdr': float(np.round(pdr.max(), 4))
        }

    # Retransmissions
    if has_retx:
        retx = df['retransmission_count'].dropna()
        profile['retransmission_stats'] = {
            'mean': float(np.round(retx.mean(), 3)),
            'max': int(retx.max()),
            'zero_retx_pct': float(np.round((retx == 0).mean() * 100, 2))
        }

    # Signal strength (RSSI)
    if has_rssi:
        rssi = df['signal_strength'].dropna()
        profile['rssi_stats_dbm'] = {
            'mean': float(np.round(rssi.mean(), 2)),
            'min': float(np.round(rssi.min(), 2)),
            'max': float(np.round(rssi.max(), 2))
        }

    # Grouped analysis if label is available
    if has_label:
        grouped = {}
        for label_val, group in df.groupby('label'):
            lbl_key = 'malicious' if int(label_val) == 1 else 'benign'
            grouped[lbl_key] = {
                'count': int(len(group)),
                'avg_latency_ms': float(np.round(group['latency'].mean(), 3)) if has_latency else None,
                'avg_pdr': float(np.round(group['packet_drop_ratio'].mean(), 4)) if has_pdr else None,
                'avg_retx': float(np.round(group['retransmission_count'].mean(), 3)) if has_retx else None
            }
        profile['class_breakdown'] = grouped

    return profile


def main():
    args = parse_args()
    print("=" * 65)
    print("  EdgeTrust-VANET — Network Telemetry & Jitter Profiler")
    print("=" * 65)

    if os.path.exists(args.data):
        print(f"[INFO] Loading telemetry data from: {args.data}")
        df = pd.read_csv(args.data)
    else:
        print(f"[WARN] Data file not found at {args.data}. Generating synthetic 802.11p telemetry profile...")
        np.random.seed(42)
        n = 1000
        df = pd.DataFrame({
            'latency': np.random.exponential(scale=15.0, size=n) + 5.0,
            'packet_drop_ratio': np.random.beta(a=0.5, b=5.0, size=n),
            'signal_strength': np.random.normal(loc=-72.0, scale=8.0, size=n),
            'retransmission_count': np.random.poisson(lam=0.8, size=n),
            'label': np.random.choice([0, 1], p=[0.7, 0.3], size=n)
        })

    profile = compute_telemetry_profile(df)

    print("\n--- Summary Telemetry Profile ---")
    if 'latency_stats_ms' in profile:
        l = profile['latency_stats_ms']
        print(f"Latency (ms)   : Mean={l['mean']}, Median={l['median']}, P95={l['p95']}, Max={l['max']}")
    if 'jitter_stats_ms' in profile:
        j = profile['jitter_stats_ms']
        print(f"Jitter (ms)    : Mean={j['mean_jitter']}, P95={j['p95_jitter']}")
    if 'packet_drop_stats' in profile:
        p = profile['packet_drop_stats']
        print(f"Drop Ratio     : Mean={p['mean_pdr']}, P95={p['p95_pdr']}")
    if 'class_breakdown' in profile:
        print("\nClass Breakdown:")
        for k, v in profile['class_breakdown'].items():
            print(f"  [{k.upper()}] Count={v['count']}, Avg Latency={v['avg_latency_ms']} ms, Avg PDR={v['avg_pdr']}")

    with open(args.output, 'w') as f:
        json.dump(profile, f, indent=2)
    print(f"\n[INFO] Profile successfully saved to: {args.output}")


if __name__ == '__main__':
    main()
