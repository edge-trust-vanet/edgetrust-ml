# Trust Threshold Evidence

This folder supports the paper question:

> Why are the trust-state thresholds 0.70 and 0.40?

## Code basis

The thresholds are implemented in `scripts/trust_score.py`:

- `trust >= 0.70`: `Trusted`
- `0.40 <= trust < 0.70`: `Suspicious`
- `trust < 0.40`: `Blocked`

The trust update is an exponential moving average:

```text
new_trust = 0.7 * previous_trust + 0.3 * evidence_score
```

The dashboard applies a stricter final policy in `dashboard/app.py`: it returns `BLOCK` only when trust is below `0.40` and the ML prediction is malicious. Low trust alone produces `WARN`, which makes the threshold a staged response rather than an automatic block.

## Generated evidence

- `trust_threshold_metrics.json`: full trajectories, threshold comparisons, and dashboard policy.
- `threshold_comparison.csv`: detection and recovery delay for alternative threshold pairs.
- `noisy_benign_state_counts.csv`: state counts under stable, noisy benign, and sustained attack evidence.
- `plots/trust_decay_recovery.png`: selected thresholds over attack and recovery phases.
- `plots/threshold_detection_delay.png`: comparison of attack detection delays for alternative threshold pairs.
- `plots/benign_noise_threshold_sensitivity.png`: effect of thresholds under noisy benign evidence.

## Main result

Starting from trust `1.0` and receiving malicious evidence `0.1`, alpha `0.7` produces trust values of `0.73` after one update, `0.541` after two, `0.4087` after three, and `0.3161` after four. Thus, the selected thresholds produce a staged progression:

```text
Trusted (packet 1) -> Suspicious (packets 2-3) -> Blocked (packet 4)
```

This supports the operational interpretation that `0.70` is an early-warning boundary and `0.40` is a stronger isolation boundary. The spacing between them gives the system time to observe continued malicious behavior before blocking.

## Paper wording

The current code supports a design-rationale claim, not a statistically optimized threshold claim. A defensible statement is:

> The trust boundaries were selected as a two-stage safety policy. A score below 0.70 indicates that confidence has degraded enough to require monitoring, while a score below 0.40 indicates sustained low trust and qualifies the node for blocking only when the ML classifier also reports malicious behavior. Under the implemented alpha=0.7 update, repeated malicious evidence moves a new node through Trusted, Suspicious, and Blocked states within three updates, while the intermediate state avoids immediate blocking from one degraded observation.

To claim that `0.70` and `0.40` are optimal, the paper would still need real labeled trust traces or a validation experiment with explicit false-warning and false-blocking costs. The generated sensitivity analysis shows behavior of alternatives, but it does not prove global optimality.

## Reproduce

From the project root:

```bash
.venv/bin/python evidence/trust_thresholds/generate_trust_threshold_evidence.py
```
