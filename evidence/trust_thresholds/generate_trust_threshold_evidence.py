"""Generate evidence for the 0.70/0.40 trust-state thresholds."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
PLOTS_DIR = OUT_DIR / "plots"
ALPHA = 0.7
TRUSTED_THRESHOLD = 0.70
BLOCKED_THRESHOLD = 0.40


def update_trust(previous: float, evidence: float) -> float:
    return round((ALPHA * previous) + ((1 - ALPHA) * evidence), 4)


def label_for(trust: float, trusted: float = TRUSTED_THRESHOLD, blocked: float = BLOCKED_THRESHOLD) -> str:
    if trust >= trusted:
        return "Trusted"
    if trust >= blocked:
        return "Suspicious"
    return "Blocked"


def trajectory(evidence: list[float], trusted: float = TRUSTED_THRESHOLD, blocked: float = BLOCKED_THRESHOLD) -> list[dict]:
    trust = 1.0
    rows = [{"packet": 0, "evidence": None, "trust": trust, "state": label_for(trust, trusted, blocked)}]
    for packet, score in enumerate(evidence, start=1):
        trust = update_trust(trust, score)
        rows.append({"packet": packet, "evidence": score, "trust": trust, "state": label_for(trust, trusted, blocked)})
    return rows


def first_packet(rows: list[dict], state: str) -> int | None:
    matches = [row["packet"] for row in rows if row["state"] == state]
    return min(matches) if matches else None


def recovery_metrics() -> list[dict]:
    rows = []
    attack = [0.10] * 8
    recovery = [0.95] * 30
    alternatives = [
        ("0.70 / 0.40", 0.70, 0.40),
        ("0.80 / 0.50", 0.80, 0.50),
        ("0.60 / 0.30", 0.60, 0.30),
        ("0.75 / 0.35", 0.75, 0.35),
        ("0.65 / 0.45", 0.65, 0.45),
    ]
    for name, trusted, blocked in alternatives:
        attack_rows = trajectory(attack, trusted, blocked)
        recovery_rows = trajectory(attack + recovery, trusted, blocked)
        recovery_start = 8
        recovered = [r["packet"] for r in recovery_rows if r["packet"] >= recovery_start and r["state"] == "Trusted"]
        rows.append({
            "threshold_pair": name,
            "trusted_threshold": trusted,
            "blocked_threshold": blocked,
            "suspicious_packet_under_attack": first_packet(attack_rows, "Suspicious"),
            "blocked_packet_under_attack": first_packet(attack_rows, "Blocked"),
            "trusted_recovery_packet": min(recovered) if recovered else None,
            "recovery_packets_after_attack": (min(recovered) - recovery_start) if recovered else None,
        })
    return rows


def noise_metrics() -> list[dict]:
    scenarios = {
        "Benign stable": [0.95] * 20,
        "Benign noisy": [0.95, 0.60, 0.90, 0.55, 0.95, 0.70, 0.85, 0.50, 0.95, 0.80] * 2,
        "Sustained attack": [0.10] * 20,
    }
    thresholds = [
        ("0.70 / 0.40", 0.70, 0.40),
        ("0.80 / 0.50", 0.80, 0.50),
        ("0.60 / 0.30", 0.60, 0.30),
        ("0.75 / 0.35", 0.75, 0.35),
        ("0.65 / 0.45", 0.65, 0.45),
    ]
    rows = []
    for scenario, evidence in scenarios.items():
        for name, trusted, blocked in thresholds:
            states = trajectory(evidence, trusted, blocked)
            counts = pd.Series([row["state"] for row in states[1:]]).value_counts().to_dict()
            rows.append({
                "scenario": scenario,
                "threshold_pair": name,
                "trusted_count": counts.get("Trusted", 0),
                "suspicious_count": counts.get("Suspicious", 0),
                "blocked_count": counts.get("Blocked", 0),
                "minimum_trust": min(row["trust"] for row in states),
            })
    return rows


def plot_attack_recovery() -> None:
    evidence = [0.10] * 8 + [0.95] * 20
    rows = trajectory(evidence)
    x = [row["packet"] for row in rows]
    y = [row["trust"] for row in rows]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(x, y, marker="o", markersize=3.5, color="#2563eb", linewidth=2, label="Trust score")
    ax.axhline(TRUSTED_THRESHOLD, color="#16a34a", linestyle="--", label="Trusted boundary (0.70)")
    ax.axhline(BLOCKED_THRESHOLD, color="#dc2626", linestyle="--", label="Blocked boundary (0.40)")
    ax.axvline(8, color="#64748b", linestyle=":", label="Evidence changes to benign (packet 9)")
    ax.fill_between(x, TRUSTED_THRESHOLD, 1, color="#16a34a", alpha=0.07)
    ax.fill_between(x, BLOCKED_THRESHOLD, TRUSTED_THRESHOLD, color="#f59e0b", alpha=0.08)
    ax.fill_between(x, 0, BLOCKED_THRESHOLD, color="#dc2626", alpha=0.06)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Packet update")
    ax.set_ylabel("Trust score")
    ax.set_title("Trust Decay Under Attack and Recovery Under Benign Evidence")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "trust_decay_recovery.png", dpi=180)
    plt.close(fig)


def plot_threshold_comparison(metrics: list[dict]) -> None:
    df = pd.DataFrame(metrics)
    x = np.arange(len(df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x - width / 2, df["suspicious_packet_under_attack"], width, label="First Suspicious packet", color="#f59e0b")
    ax.bar(x + width / 2, df["blocked_packet_under_attack"], width, label="First Blocked packet", color="#dc2626")
    ax.set_xticks(x, df["threshold_pair"], rotation=25, ha="right")
    ax.set_ylabel("Packet number")
    ax.set_title("Detection Delay Under Sustained Malicious Evidence (0.10)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "threshold_detection_delay.png", dpi=180)
    plt.close(fig)


def plot_state_counts(rows: list[dict]) -> None:
    df = pd.DataFrame([r for r in rows if r["scenario"] != "Sustained attack"])
    pivot = df[df["scenario"] == "Benign noisy"].set_index("threshold_pair")[["trusted_count", "suspicious_count", "blocked_count"]]
    ax = pivot.plot(kind="bar", figsize=(10, 5.5), color=["#16a34a", "#f59e0b", "#dc2626"])
    ax.set_title("Trust States During a Noisy but Benign Evidence Sequence")
    ax.set_xlabel("Trusted / Blocked thresholds")
    ax.set_ylabel("Packets in state")
    ax.legend(["Trusted", "Suspicious", "Blocked"])
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "benign_noise_threshold_sensitivity.png", dpi=180)
    plt.close()


def main() -> None:
    PLOTS_DIR.mkdir(exist_ok=True)
    attack_recovery = trajectory([0.10] * 8 + [0.95] * 20)
    threshold_metrics = recovery_metrics()
    state_metrics = noise_metrics()
    output = {
        "source_files": ["scripts/trust_score.py", "scripts/combined_decision.py", "dashboard/app.py"],
        "alpha": ALPHA,
        "selected_thresholds": {"trusted": TRUSTED_THRESHOLD, "blocked": BLOCKED_THRESHOLD},
        "formula": "new_trust = alpha * previous_trust + (1 - alpha) * evidence_score",
        "attack_evidence": 0.10,
        "benign_evidence": 0.95,
        "attack_recovery_trajectory": attack_recovery,
        "threshold_comparison": threshold_metrics,
        "noisy_benign_state_counts": state_metrics,
        "dashboard_policy": {
            "block": "trust < 0.40 and ML prediction is malicious",
            "warn": "trust < 0.40 or ML prediction is malicious, unless block applies",
            "accept": "trust >= 0.40 and ML prediction is normal",
        },
        "interpretation": [
            "With alpha=0.7 and malicious evidence=0.1, trust moves from 1.0 to 0.73 after one packet, 0.541 after two, 0.4087 after three, and 0.3161 after four; the selected boundaries therefore create a Trusted -> Suspicious -> Blocked progression within four updates.",
            "The 0.70 boundary creates an early warning state before the stricter 0.40 blocking boundary is crossed.",
            "The dashboard does not block on trust alone; it requires low trust and an ML-malicious prediction, reducing the risk of blocking a vehicle from a single noisy trust signal.",
        ],
    }
    with open(OUT_DIR / "trust_threshold_metrics.json", "w") as f:
        json.dump(output, f, indent=2)
    with open(OUT_DIR / "threshold_comparison.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(threshold_metrics[0].keys()))
        writer.writeheader()
        writer.writerows(threshold_metrics)
    with open(OUT_DIR / "noisy_benign_state_counts.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(state_metrics[0].keys()))
        writer.writeheader()
        writer.writerows(state_metrics)

    plot_attack_recovery()
    plot_threshold_comparison(threshold_metrics)
    plot_state_counts(state_metrics)
    print(f"Wrote evidence to {OUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
