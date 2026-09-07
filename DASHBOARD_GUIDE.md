# EdgeTrust-VANET Dashboard: Complete User & Demonstration Guide 🚗💻

Welcome to the **EdgeTrust-VANET Security Dashboard**. This guide explains what the dashboard is, how it works, what each button and slider does, and how to use it for demonstrations, testing, or research evaluation.

---

## 🎯 What is the Dashboard?

In a real-world Vehicular Ad-Hoc Network (VANET), **Roadside Units (RSUs)** are physical towers mounted along highways and intersections. As connected vehicles drive by, they broadcast **Basic Safety Messages (BSMs)** 10 times per second over 802.11p DSRC / C-V2X radio.

An RSU must make an **instant decision** for every vehicle:
- ✅ **ACCEPT**: The vehicle is behaving honestly; forward its safety messages to traffic control and nearby cars.
- ⚠️ **WARN**: Minor anomaly or low trust detected; flag the vehicle, monitor closely, and deprioritize its messages.
- 🚫 **BLOCK**: Confirmed malicious attacker (e.g. spoofing fake accidents, performing a blackhole attack, or flooding the channel); quarantine and block all messages to protect nearby autonomous vehicles.

The **EdgeTrust-VANET Dashboard** is an interactive web interface that simulates an **RSU edge node** in real time, connecting the frontend directly to our trained Machine Learning models and the EdgeTrust heuristic engine.

---

## 🚀 How to Start the Dashboard

### Step 1: Start the Backend Server
Open your terminal in the project folder and run:

```bash
python dashboard/app.py
```

You will see output similar to:
```text
  ✔ Loaded unified scaler
  ✔ Loaded model: Random Forest (vanet_Random_Forest.pkl)
  ✔ Loaded model: XGBoost (vanet_XGBoost.pkl)
  ✔ Loaded model: LightGBM (vanet_LightGBM.pkl)
  ✔ Loaded model: CatBoost (vanet_CatBoost.pkl)
  ...
  Active model : Random Forest (Available)
  Dashboard    → http://127.0.0.1:5000
```

### Step 2: Open in Your Browser
Open Google Chrome, Edge, Safari, or Firefox and navigate to:
```
http://127.0.0.1:5000
```

> **Tip**: If you ever see cached older results, press **Ctrl + Shift + R** (or **Cmd + Shift + R** on Mac) to force-refresh your browser cache.

---

## 🧭 Dashboard Layout & Navigation

At the top right of the dashboard, you will find three main navigation tabs:

```text
[ 🚗 Evaluator ]    [ 🏆 VeReMi Arena ]    [ 🌐 VANET Nodes ]
```

---

## 1. 🚗 Tab 1: The Evaluator (Live RSU Simulation)

This is the main interactive tab where you can test individual vehicles and watch the AI make real-time decisions.

```text
+-------------------------------------------------------------------------------+
|  STAT COUNTERS:                                                               |
|  [ ✅ Vehicles Accepted: 14 ]   [ ⚠️ Warnings: 3 ]   [ 🚫 Blocked: 5 ]        |
+-------------------------------------------------------+-----------------------+
|  LEFT CARD: Vehicle Evaluator Form                    | RIGHT CARD: Verdict   |
|  - Speed, Acceleration, Position X/Y, Heading         | - [ BLOCK / ACCEPT ]  |
|  - Message Frequency, Neighbor Count                  | - ML Prediction (99%) |
|  - Trust Sliders (Consistency, History, Plausibility) | - Trust Score (0.21)  |
|  [ Simulate Random ]   [ Evaluate Vehicle ]           | - Radar Breakdown     |
+-------------------------------------------------------+-----------------------+
|  ACTIVITY LOG: Real-time table of recent evaluations                          |
+-------------------------------------------------------------------------------+
```

### A. The Stat Counters (Top Row)
- **Vehicles Accepted** (Green): Total legitimate vehicles accepted during this session.
- **Warnings Issued** (Yellow): Vehicles that triggered caution flags.
- **Vehicles Blocked** (Red): Malicious nodes quarantined by the RSU.
- **Total Evaluated** (Blue): Total vehicle evaluations performed.

### B. Two Ways to Evaluate a Vehicle

#### Option 1: "Simulate Random" Button (Fastest & Easiest!)
Click the purple **"Simulate Random"** button at the top right of the Evaluator card.
- The system randomly creates either a normal vehicle or a realistic attack (Position Spoofing, Blackhole packet drop, or DoS flood).
- It automatically populates all sensor fields and trust sliders.
- It immediately sends the vehicle through the AI and displays the verdict!
- Click it 5–10 times in a row to watch the stats increment and see the Activity Log update in real time!

#### Option 2: Manual Form Entry
You can manually type values and drag sliders to simulate specific scenarios:
1. **Sensor Features**:
   - **Speed (m/s)**: Vehicle speed (normal highway speed is ~15–30 m/s; >50 m/s is suspicious).
   - **Acceleration (m/s²)**: Rate of speed change (normal braking/accel is -4 to +3 m/s²; sudden jumps like +15 m/s² indicate spoofed GPS).
   - **Position X & Y (m)**: Coordinates in the simulated city grid.
   - **Heading (°)**: Direction angle from 0° to 360°.
   - **Msg Frequency (Hz)**: How many BSMs per second the vehicle is sending (standard is 10 Hz; 50 Hz indicates a DoS flood).
   - **Neighbor Count**: Number of nearby vehicles within 300m radio range.
2. **Trust Factors (Sliders 0.00 to 1.00)**:
   - **Message Consistency**: Are consecutive messages physically logical?
   - **Behavior History**: Does this vehicle have a clean historical track record?
   - **Neighbor Validation**: Do nearby cars agree with this vehicle's claimed location?
   - **Plausibility Check**: Does the vehicle obey physics (no teleporting)?
3. Click **"Evaluate Vehicle"** to submit.

---

## ⚖️ How Does the AI Make the Final Decision?

EdgeTrust-VANET uses a **Hybrid Defense Policy**. It does not rely on Machine Learning alone, nor does it rely on Trust scores alone. It combines both:

| Real-Time Trust Score ($T_t$) | ML Model Prediction | Final Verdict | What the RSU Does |
| :---: | :---: | :---: | :--- |
| **High ($\ge 0.70$)** | **Normal** | <span style="color:#10b981; font-weight:bold">✅ ACCEPT</span> | Message forwarded; full network privileges. |
| **Suspicious ($0.40 - 0.70$)** | **Normal** | <span style="color:#f59e0b; font-weight:bold">⚠️ WARN</span> | Monitored closely; safety beacons accepted with lower priority. |
| **High ($\ge 0.70$)** | **Malicious** | <span style="color:#f59e0b; font-weight:bold">⚠️ WARN</span> | Discrepancy flagged; RSU challenges the vehicle before blocking. |
| **Low ($< 0.40$)** | **Malicious** | <span style="color:#ef4444; font-weight:bold">🚫 BLOCK</span> | **Attacker Confirmed!** Vehicle quarantined; packets dropped. |

### Why This Policy?
- In connected vehicles, **false blocks are dangerous** (you don't want to accidentally disconnect an honest ambulance or braking car).
- Requiring **both** low trust and an ML malicious prediction for `BLOCK` prevents false alarms while ensuring malicious attackers are promptly neutralized.

---

## 2. 🌐 Tab 2: VANET Nodes (Research & Dataset Insights)

Click the **"🌐 VANET Nodes"** tab in the navigation bar to inspect the dataset statistics and research results:

1. **Dataset Overview Cards**:
   - **23,970 Total Nodes**: The unified research dataset.
   - **70.6% Normal vs 29.4% Malicious**: Realistic attack penetration matching the LuST Luxembourg simulation.
   - **14 Leakage-Free Features**: Proves that post-hoc attack counters are excluded.
2. **🏆 Best Model Spotlight (Random Forest)**:
   - Shows the champion classifier: **97.74% F1 Score, 97.77% Accuracy**.
   - Bullet points explain why it won (highest weighted F1, zero overfitting, low false alarms).
3. **📊 Feature Importance Chart**:
   - Horizontal bar chart showing which signals the AI relies on most:
     - `historical_trust_score` (27.2%)
     - `position_y` and `position_x` (28.5% combined spatial consistency)
     - `neighbor_trust_score_avg` (14.4%)
     - `trust_score` (14.3%)
     - `speed`, `packet_drop_ratio`, `latency`, `signal_strength`
4. **⚔️ All 17 Model Comparison Chart**:
   - Side-by-side visualization comparing Accuracy and F1 score across all 17 algorithms (Random Forest, LightGBM, CatBoost, XGBoost, Stacking, Neural Network, etc.).
5. **Detailed Model Leaderboard Table**:
   - Full rankings with Accuracy, F1 Score, Precision, Recall, and False Alert Rate.

---

## 3. 🏆 Tab 3: VeReMi Arena (Benchmark Leaderboard)

Click the **"🏆 VeReMi Arena"** tab to compare model performance metrics under the standard VeReMi reference benchmark. You will see:
- Interactive ranking table sorted by Weighted F1 score.
- Breakdown of model strengths and computational overheads.

---

## 🎬 Step-by-Step Demo Script (What to Say to an Evaluator or Panel)

If you are presenting this project to a panel, professor, or interviewer, follow this 2-minute flow:

1. **Introduce the Problem (30 seconds)**:
   > *"In connected vehicles, standard security uses digital certificates to verify identity. But if an attacker compromises a valid certificate, they can transmit fake GPS locations or perform blackhole packet drops. Our project, **EdgeTrust-VANET**, solves this at Roadside Units by combining heuristic trust tracking with machine learning."*

2. **Demonstrate the Evaluator Tab (45 seconds)**:
   > *"Let me show you our live RSU simulation. I'll click 'Simulate Random'. Here, an honest vehicle enters with high consistency sliders and normal speed. The AI predicts Normal with 93% confidence, trust is 0.85, and the RSU issues an **ACCEPT**."*
   > *(Click Simulate Random again until an attack occurs)*
   > *"Now an attacker enters broadcasting falsified positions with high drop ratio. The trust score drops to 0.22, the Random Forest model detects the malicious pattern with 99% confidence, and the RSU immediately triggers **BLOCK**, quarantining the vehicle."*

3. **Showcase the Research & Anti-Leakage Rigor (45 seconds)**:
   > *"Clicking on the 'VANET Nodes' tab shows our research dataset of 23,970 records. 72.9% comes from the authentic VeReMi LuST simulation. Notice we strictly exclude retrospective attack counters to eliminate data leakage. Tested on over 3,500 completely unseen vehicles, our Random Forest achieves 97.74% F1, while our LightGBM model offers an ultra-fast 329 microsecond latency suitable for real-time edge microcontrollers."*

---

## ❓ Frequently Asked Questions & Troubleshooting

### Q: Why does the terminal say `* Running on http://127.0.0.1:5000`?
That is the local web address. Open your browser and type `http://127.0.0.1:5000` in the URL address bar.

### Q: What if port 5000 is already in use?
If another program is using port 5000 (such as macOS AirPlay Receiver):
- On macOS: Go to *System Settings > General > AirDrop & Handoff > AirPlay Receiver* and temporarily toggle it OFF.
- Or specify a different port in `dashboard/app.py` (e.g. `port=5001`).

### Q: How do I stop the dashboard server?
In your terminal, press **`Ctrl + C`**.
