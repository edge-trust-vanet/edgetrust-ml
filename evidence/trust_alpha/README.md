# Trust Alpha Evidence: Why α = 0.7

This folder justifies the forgetting factor in the Hybrid Trust Engine (`scripts/trust_score.py`):

```text
T_t = α · T_{t-1} + (1 − α) · E_t
T_0 = 1.0
```

`E_t` is the ML honest-probability (evidence) for packet `t`. Larger α keeps more history; smaller α reacts faster to new evidence. The implemented default is **α = 0.7**.

Thresholds are held at the operational values 0.70 (Trusted) and 0.40 (Blocked). Those boundaries are justified separately in [`evidence/trust_thresholds/`](../trust_thresholds/README.md).

## Files

- [`generate_trust_alpha_evidence.py`](generate_trust_alpha_evidence.py): Replays `HybridTrustEngine` across α ∈ {0.3, 0.5, 0.6, 0.7, 0.8, 0.9}.
- [`alpha_comparison.csv`](alpha_comparison.csv): Detection delay, recovery delay, and noise robustness per α.
- [`alpha_trajectories.csv`](alpha_trajectories.csv): Full decay/recovery curves for the overlay plot.
- [`trust_alpha_metrics.json`](trust_alpha_metrics.json): Selection rule and per-α metrics.
- [`plots/trust_alpha_sensitivity.png`](plots/trust_alpha_sensitivity.png): Main 4-panel figure.

A copy is also written to [`results/trust_alpha_sensitivity.png`](../../results/trust_alpha_sensitivity.png).

## Selection Rule

Choose the **smallest α that still stays Trusted after one malicious-looking packet** (`E = 0.10`), while reaching **Blocked under sustained attack in at most 4 packets**.

Only **α = 0.7** hits both constraints.

## Results (start at T=1.0)

| α | Trust after 1 attack packet | Stays Trusted after 1 glitch? | First Suspicious | First Blocked | Benign packets to recover |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 0.3 | 0.37 | No (Blocks immediately) | — (skips) | **1** | 2 |
| 0.5 | 0.55 | No (already Suspicious) | 1 | 2 | 2 |
| 0.6 | 0.64 | No (already Suspicious) | 1 | 3 | 3 |
| **0.7 (chosen)** | **0.73** | **Yes** | **2** | **4** | **4** |
| 0.8 | 0.82 | Yes | 2 | 5 | 5 |
| 0.9 | 0.91 | Yes | 4 | 11 | 6 |

Under a 20-packet noisy-but-benign sequence, α = 0.7 never leaves Trusted (min trust 0.704). α ≤ 0.6 spends 2–5 packets in Suspicious.

## Why 0.7, Not The Rest

- **α = 0.3 / 0.5 / 0.6** overreact. A single degraded beacon is enough to leave Trusted. In V2X that is a GPS glitch or a one-off drop, not an attack. α = 0.3 even Blocks on packet 1.
- **α = 0.7** is the first value that absorbs one `E = 0.10` packet (trust = 0.73, still Trusted), then stages the response: Suspicious at packet 2, Blocked at packet 4 if the attack continues. After the attack stops it is Trusted again in 4 benign packets.
- **α = 0.8 / 0.9** are calmer still, but they delay isolation. α = 0.9 needs 11 packets to Block — too slow for a roadside unit that sees attackers for only a few beacons.

## Paper Wording

> The hybrid trust engine updates node reputation with an exponential moving average, \(T_t = \alpha T_{t-1} + (1-\alpha) E_t\), where \(E_t\) is the classifier’s honest-class probability. Sensitivity analysis over \(\alpha \in \{0.3, 0.5, 0.6, 0.7, 0.8, 0.9\}\) shows that \(\alpha = 0.7\) is the smallest value that remains Trusted after a single malicious-looking packet (trust = 0.73) while still reaching the Blocked boundary within four sustained-attack updates. Smaller \(\alpha\) over-punishes one-off V2X noise; larger \(\alpha\) delays isolation of a persistent attacker.

## Reproduce

```bash
python evidence/trust_alpha/generate_trust_alpha_evidence.py
```
