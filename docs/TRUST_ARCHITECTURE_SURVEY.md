# Decentralized Trust Management Architectures for VANETs: Synthesis & Taxonomy

**Author:** Abhinav Juturu  
**Affiliation:** EdgeTrust-VANET Research Group, Amrita Vishwa Vidyapeetham  
**Date:** September 2026  

---

## 1. Executive Summary

This document synthesizes comparative architectural paradigms for trust management and insider misbehavior detection in Vehicular Ad-Hoc Networks (VANETs). As vehicular communication evolves under IEEE 802.11p (DSRC) and 3GPP Cellular-V2X (C-V2X), conventional Public Key Infrastructure (PKI) signatures verify vehicle identity but cannot detect legitimate insiders broadcasting physically fabricated kinematic telemetry.

Decentralized edge-assisted trust frameworks bridge this vulnerability by distributing reputation calculations across local Roadside Units (RSUs) and vehicular On-Board Units (OBUs), eliminating the single point of failure and high communication latency of centralized cloud trust authorities.

---

## 2. Architectural Comparison Matrix

| Paradigm | Trust Aggregation Point | Consensus Latency | Scalability | Single Point of Failure | Attack Resilience |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Centralized Cloud** | Remote Fog / Cloud DC | High (> 500 ms) | Low ($O(N)$ channel load) | Yes | Poor against local burst attacks |
| **Pure Ad-Hoc (P2P)** | Vehicle OBU Only | Low (< 20 ms) | Moderate | No | Vulnerable to Sybil & collusion |
| **Edge-Hierarchical (EdgeTrust)** | 4-Tier: Sensor $\to$ OBU $\to$ RSU $\to$ Fog | Sub-millisecond at Tier 2 | High (Local cluster scaling) | No | High (Multi-angle consensus) |
| **Blockchain-Assisted** | Distributed Ledger on RSUs | High (Block mining delay) | Low-to-Medium | No | High for audit, poor for live braking |

---

## 3. Attack Taxonomy & Detection Capabilities

| Attack Type | Physical Manifestation | Observable Indicator | EdgeTrust Defense Mechanism |
| :--- | :--- | :--- | :--- |
| **Position Forgery (Constant / Random)** | Vehicle transmits fake GPS coordinates | Spatial deviation from RSU Doppler / Kalman Filter | Kinematic filter + Neighbor consensus |
| **Speed / Acceleration Spoofing** | Telemetry reports unrealistic velocity steps | $|\Delta v / \Delta t| > a_{\text{max}}$ | Kinematic bounds check ($a \in [-8, +5]\, \text{m/s}^2$) |
| **Sybil Attack** | Single node claims multiple virtual identities | Identical RSSI / MAC sequence patterns | Signal strength correlation + MAC sequence audit |
| **DoS / Beacon Flooding** | High-frequency beacon transmission | PDR spike, transmission counter surge | Leaky bucket rate limiter + MAC retry penalty |
| **On-Off Reputation Attack** | Oscillating good/bad behavior to evade penalty | High trust variance, rapid score drop | Exponential Moving Average ($\alpha=0.70$) penalty |

---

## 4. Key Takeaways for EdgeTrust-VANET Implementation

1. **Reputation Dominance:** Machine learning models trained on pure kinematic features remain vulnerable to stealthy attacks that respect physical speed bounds. Incorporating dynamic trust features (historical trust, direct trust, and spatial neighbor consensus) provides over 55% of the total predictive gain.
2. **Sub-Millisecond Edge Budget:** In automated driving scenarios, emergency brake notification decisions must be computed within 10 ms. Deploying lightweight classifiers (Random Forest, AdaBoost, or regularized linear models) directly at the RSU satisfies this strict timing budget.
3. **Leakage Elimination:** Trust systems must strictly decouple training features from retrospective counters (e.g., ground-truth attack logs) to prevent real-time prediction failure in deployment environments.
