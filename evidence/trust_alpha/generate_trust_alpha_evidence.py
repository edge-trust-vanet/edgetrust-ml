"""
Generate paper evidence for why alpha=0.7 in the Hybrid Trust EMA.

Formula (scripts/trust_score.py HybridTrustEngine):
    T_t = alpha * T_{t-1} + (1 - alpha) * E_t
    T_0 = 1.0

Higher alpha keeps more history (stable, slow). Lower alpha reacts faster
to new ML evidence (responsive, noisy). This script compares alpha values
and shows 0.7 is the fastest setting that still stays Trusted after a
single malicious-looking packet.

Run from the project root:
    python evidence/trust_alpha/generate_trust_alpha_evidence.py

Outputs:
    evidence/trust_alpha/alpha_comparison.csv
    evidence/trust_alpha/alpha_trajectories.csv
    evidence/trust_alpha/trust_alpha_metrics.json
    evidence/trust_alpha/plots/trust_alpha_sensitivity.png
    results/trust_alpha_sensitivity.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
PLOTS_DIR = OUT_DIR / "plots"
RESULTS_DIR = ROOT / "results"

sys.path.insert(0, str(ROOT / "scripts"))
from trust_score import HybridTrustEngine  # noqa: E402

CHOSEN = 0.7
CURVE_ALPHAS = [0.3, 0.5, 0.7, 0.9]
BAR_ALPHAS = [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]
TRUSTED = 0.70
BLOCKED = 0.40
ATTACK_E = 0.10
BENIGN_E = 0.95
ATTACK_PACKETS = 8
RECOVERY_PACKETS = 20
NOISE_WINDOW = 12

CURVE_COLORS = {
    0.3: "#dc2626",
    0.5: "#7c3aed",
    0.7: "#d97706",
    0.9: "#475569",
}
CHOSEN_COLOR = "#d97706"
OTHER_COLOR = "#64748b"


def label_for(trust: float) -> str:
    if trust >= TRUSTED:
        return "Trusted"
    if trust >= BLOCKED:
        return "Suspicious"
    return "Blocked"


def simulate(alpha: float, evidence: list[float]) -> list[dict]:
    engine = HybridTrustEngine(alpha=alpha)
    vid = "V"
    rows = [{
        "packet": 0,
        "evidence": None,
        "trust": 1.0,
        "state": label_for(1.0),
        "alpha": alpha,
    }]
    for i, score in enumerate(evidence, start=1):
        trust = engine.update_trust(vid, score)
        rows.append({
            "packet": i,
            "evidence": score,
            "trust": trust,
            "state": engine.classify_vehicle(trust),
            "alpha": alpha,
        })
    return rows


def first_state(rows: list[dict], state: str) -> int | None:
    hits = [r["packet"] for r in rows if r["packet"] > 0 and r["state"] == state]
    return min(hits) if hits else None


def recovery_packet(rows: list[dict], attack_len: int) -> int | None:
    hits = [
        r["packet"] for r in rows
        if r["packet"] > attack_len and r["state"] == "Trusted"
    ]
    return min(hits) if hits else None


def metrics_for(alpha: float) -> dict:
    attack = simulate(alpha, [ATTACK_E] * NOISE_WINDOW)
    decay_recovery = simulate(
        alpha, [ATTACK_E] * ATTACK_PACKETS + [BENIGN_E] * RECOVERY_PACKETS
    )
    glitch = simulate(alpha, [ATTACK_E] + [BENIGN_E] * 10)
    noisy = simulate(
        alpha,
        [0.95, 0.60, 0.90, 0.55, 0.95, 0.70, 0.85, 0.50, 0.95, 0.80] * 2,
    )
    noisy_states = [r["state"] for r in noisy if r["packet"] > 0]
    rec = recovery_packet(decay_recovery, ATTACK_PACKETS)
    sus_pkt = first_state(attack, "Suspicious")
    blk_pkt = first_state(attack, "Blocked")
    return {
        "alpha": alpha,
        "is_chosen": alpha == CHOSEN,
        "suspicious_packet": sus_pkt,
        "blocked_packet": blk_pkt,
        "trust_after_1_attack": attack[1]["trust"],
        "trust_after_4_attack": attack[4]["trust"] if len(attack) > 4 else None,
        "stays_trusted_after_one_glitch": glitch[1]["state"] == "Trusted",
        "min_trust_single_glitch": min(r["trust"] for r in glitch),
        "recovery_packet": rec,
        "recovery_packets_after_attack": (rec - ATTACK_PACKETS) if rec else None,
        "noisy_trusted": noisy_states.count("Trusted"),
        "noisy_suspicious": noisy_states.count("Suspicious"),
        "noisy_blocked": noisy_states.count("Blocked"),
        "noisy_min_trust": min(r["trust"] for r in noisy),
    }


def style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.28, linestyle="--")
    ax.set_axisbelow(True)


def plot_sensitivity(metrics: pd.DataFrame, trajectories: pd.DataFrame, path: Path) -> None:
    fig = plt.figure(figsize=(16.0, 10.2))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(2, 2, height_ratios=[1.18, 1.0], hspace=0.38, wspace=0.28)

    fig.suptitle(
        "Why α = 0.7 in the Hybrid Trust Update",
        fontsize=15,
        fontweight="bold",
        y=0.98,
    )
    fig.text(
        0.5,
        0.935,
        r"$T_t = \alpha\,T_{t-1} + (1-\alpha)\,E_t$   ·   $T_0=1.0$   ·   "
        r"α=0.7 is the fastest value that stays Trusted after one noisy packet",
        ha="center",
        fontsize=10,
        color="#475569",
    )

    # ── A. Decay + recovery overlay ────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    style_axes(ax)
    ax.axhspan(TRUSTED, 1.02, color="#16a34a", alpha=0.07, zorder=0)
    ax.axhspan(BLOCKED, TRUSTED, color="#f59e0b", alpha=0.08, zorder=0)
    ax.axhspan(0.0, BLOCKED, color="#dc2626", alpha=0.06, zorder=0)
    ax.axhline(TRUSTED, color="#16a34a", linestyle="--", linewidth=1.1, zorder=1)
    ax.axhline(BLOCKED, color="#dc2626", linestyle="--", linewidth=1.1, zorder=1)
    ax.axvline(ATTACK_PACKETS, color="#94a3b8", linestyle=":", linewidth=1.2, zorder=1)

    for alpha in CURVE_ALPHAS:
        sub = trajectories[trajectories["alpha"] == alpha]
        lw = 2.8 if alpha == CHOSEN else 1.7
        z = 4 if alpha == CHOSEN else 3
        ax.plot(
            sub["packet"],
            sub["trust"],
            color=CURVE_COLORS[alpha],
            linewidth=lw,
            marker="o",
            markersize=3.6 if alpha == CHOSEN else 2.8,
            label=f"α = {alpha:.1f}" + ("  (chosen)" if alpha == CHOSEN else ""),
            zorder=z,
        )

    ax.set_ylim(0.0, 1.05)
    ax.set_xlim(0, ATTACK_PACKETS + RECOVERY_PACKETS)
    ax.set_xlabel("Packet update")
    ax.set_ylabel("Trust score")
    ax.set_title("A. Attack decay and benign recovery")
    ax.text(
        ATTACK_PACKETS / 2,
        1.02,
        "Attack  E=0.10",
        ha="center",
        va="top",
        fontsize=8,
        color="#64748b",
    )
    ax.text(
        ATTACK_PACKETS + RECOVERY_PACKETS / 2,
        1.02,
        "Recovery  E=0.95",
        ha="center",
        va="top",
        fontsize=8,
        color="#64748b",
    )
    ax.legend(loc="lower right", fontsize=8, frameon=True)

    # ── B. Detection delay ─────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    style_axes(ax)
    x = np.arange(len(BAR_ALPHAS))
    width = 0.36
    def pkt(val) -> int:
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return 0
        return int(val)

    sus = [pkt(metrics.loc[a, "suspicious_packet"]) for a in BAR_ALPHAS]
    blk = [pkt(metrics.loc[a, "blocked_packet"]) for a in BAR_ALPHAS]
    bars_s = ax.bar(x - width / 2, sus, width, color="#fbbf24", label="First Suspicious", zorder=3)
    bars_b = ax.bar(x + width / 2, blk, width, color="#ef4444", label="First Blocked", zorder=3)
    for bars in (bars_s, bars_b):
        for bar, a in zip(bars, BAR_ALPHAS):
            if a == CHOSEN:
                bar.set_edgecolor("#7c2d12")
                bar.set_linewidth(1.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a:.1f}" for a in BAR_ALPHAS])
    ax.set_xlabel("α")
    ax.set_ylabel("Packets until state change")
    ax.set_title("B. Detection delay under sustained attack")
    ax.legend(fontsize=8)
    ymax = max(blk) * 1.22
    ax.set_ylim(0, ymax)
    for i, (s, b, a) in enumerate(zip(sus, blk, BAR_ALPHAS)):
        if s:
            ax.text(i - width / 2, s + ymax * 0.02, str(s), ha="center", fontsize=8,
                    fontweight="bold" if a == CHOSEN else "normal")
        ax.text(i + width / 2, b + ymax * 0.02, str(b), ha="center", fontsize=8,
                fontweight="bold" if a == CHOSEN else "normal")

    # ── C. Recovery delay + single-glitch stability ────────────────────
    ax = fig.add_subplot(gs[1, 0])
    style_axes(ax)
    rec = [metrics.loc[a, "recovery_packets_after_attack"] or 0 for a in BAR_ALPHAS]
    colors = [CHOSEN_COLOR if a == CHOSEN else OTHER_COLOR for a in BAR_ALPHAS]
    bars = ax.bar(x, rec, color=colors, width=0.62, zorder=3)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{a:.1f}" for a in BAR_ALPHAS])
    ax.set_xlabel("α")
    ax.set_ylabel("Benign packets to return to Trusted")
    ax.set_title("C. Recovery delay after 8-packet attack")
    ymax = max(rec) * 1.22
    ax.set_ylim(0, ymax)
    for bar, val, a in zip(bars, rec, BAR_ALPHAS):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            val + ymax * 0.02,
            str(val),
            ha="center",
            fontsize=8.5,
            fontweight="bold" if a == CHOSEN else "normal",
            color="#7c2d12" if a == CHOSEN else "#334155",
        )

    # ── D. Stability vs speed ──────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    style_axes(ax)
    ax.axhline(TRUSTED, color="#16a34a", linestyle="--", linewidth=1.1, zorder=1)
    ax.fill_between(
        [0.5, 4.6], TRUSTED, 1.02,
        color="#dcfce7", alpha=0.45, zorder=0,
    )
    ax.text(
        2.55,
        0.985,
        "Stays Trusted after one glitch",
        ha="center",
        va="top",
        fontsize=8,
        color="#166534",
    )

    for a in BAR_ALPHAS:
        row = metrics.loc[a]
        ax.scatter(
            [row["blocked_packet"]],
            [row["trust_after_1_attack"]],
            s=220 if a == CHOSEN else 90,
            marker="*" if a == CHOSEN else "o",
            color=CHOSEN_COLOR if a == CHOSEN else OTHER_COLOR,
            edgecolors="#7c2d12" if a == CHOSEN else "#334155",
            linewidths=0.8,
            zorder=5 if a == CHOSEN else 4,
        )
        dy = 8 if a != 0.8 else -12
        ax.annotate(
            f"α={a:.1f}",
            (row["blocked_packet"], row["trust_after_1_attack"]),
            textcoords="offset points",
            xytext=(7, dy),
            fontsize=8,
            fontweight="bold" if a == CHOSEN else "normal",
            color="#7c2d12" if a == CHOSEN else "#0f172a",
        )

    ax.set_xlabel("Packets until Blocked (lower = faster detection)")
    ax.set_ylabel("Trust after 1 malicious packet (higher = more stable)")
    ax.set_title("D. Speed–stability tradeoff  ·  lower-left overreacts, upper-right is slow")
    ax.set_xlim(0.5, max(metrics["blocked_packet"]) + 1.2)
    ax.set_ylim(0.28, 1.02)

    legend_handles = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor=CHOSEN_COLOR,
               markeredgecolor="#7c2d12", markersize=14, label="α = 0.7 (chosen)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=OTHER_COLOR,
               markersize=8, label="Other α"),
        Line2D([0], [0], color="#16a34a", linestyle="--", label="Trusted boundary 0.70"),
        Line2D([0], [0], color="#dc2626", linestyle="--", label="Blocked boundary 0.40"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
        fontsize=9,
    )

    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    metric_rows = [metrics_for(a) for a in BAR_ALPHAS]
    metrics = pd.DataFrame(metric_rows).set_index("alpha")

    traj_rows = []
    evidence = [ATTACK_E] * ATTACK_PACKETS + [BENIGN_E] * RECOVERY_PACKETS
    for alpha in CURVE_ALPHAS:
        traj_rows.extend(simulate(alpha, evidence))
    trajectories = pd.DataFrame(traj_rows)

    chosen = metrics.loc[CHOSEN].to_dict()
    payload = {
        "formula": "T_t = alpha * T_{t-1} + (1 - alpha) * E_t",
        "source": "scripts/trust_score.py HybridTrustEngine",
        "chosen_alpha": CHOSEN,
        "thresholds": {"trusted": TRUSTED, "blocked": BLOCKED},
        "attack_evidence": ATTACK_E,
        "benign_evidence": BENIGN_E,
        "selection_rule": (
            "Choose the smallest alpha that still stays Trusted after a single "
            "malicious-looking packet (E=0.10), while reaching Blocked under "
            "sustained attack in at most 4 packets."
        ),
        "chosen_metrics": {
            "suspicious_packet": chosen["suspicious_packet"],
            "blocked_packet": chosen["blocked_packet"],
            "trust_after_1_attack": chosen["trust_after_1_attack"],
            "stays_trusted_after_one_glitch": chosen["stays_trusted_after_one_glitch"],
            "recovery_packets_after_attack": chosen["recovery_packets_after_attack"],
        },
        "all_alphas": metric_rows,
        "why": [
            "α=0.3 drops to 0.37 after one malicious packet and Blocks immediately — too reactive for noisy V2X evidence.",
            "α=0.5 and α=0.6 leave Trusted after a single glitch (trust 0.55 / 0.64) and over-warn.",
            "α=0.7 is the first value that stays Trusted after one E=0.10 packet (trust=0.73), then Suspicious at packet 2 and Blocked at packet 4.",
            "α=0.8 / 0.9 are more stable but slow: Blocked only at packet 5 / 11, which delays isolation of a sustained attacker.",
            "After an 8-packet attack, α=0.7 returns to Trusted in 4 benign packets — fast enough to forgive a recovered node.",
        ],
    }

    metrics.reset_index().to_csv(OUT_DIR / "alpha_comparison.csv", index=False)
    trajectories.to_csv(OUT_DIR / "alpha_trajectories.csv", index=False)
    with open(OUT_DIR / "trust_alpha_metrics.json", "w") as f:
        json.dump(payload, f, indent=2)

    plot_path = PLOTS_DIR / "trust_alpha_sensitivity.png"
    plot_sensitivity(metrics, trajectories, plot_path)

    import shutil
    shutil.copy2(plot_path, RESULTS_DIR / "trust_alpha_sensitivity.png")

    print(f"Wrote {plot_path.relative_to(ROOT)}")
    print(f"Chosen α={CHOSEN}: "
          f"Suspicious@{chosen['suspicious_packet']}  "
          f"Blocked@{chosen['blocked_packet']}  "
          f"trust after 1 attack={chosen['trust_after_1_attack']}  "
          f"recovery={chosen['recovery_packets_after_attack']} packets")
    print("Blocked packet by alpha:",
          {a: int(metrics.loc[a, "blocked_packet"]) for a in BAR_ALPHAS})


if __name__ == "__main__":
    main()
