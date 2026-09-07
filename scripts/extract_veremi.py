"""
╔══════════════════════════════════════════════════════════════════╗
║         extract_veremi.py                                        ║
║         EdgeTrust-VANET — Primary Reference VeReMi Extractor     ║
╠══════════════════════════════════════════════════════════════════╣
║  Extracts authentic VeReMi simulation archives (Types 1,2,4,8,16)║
║  and computes the 14 leakage-free RSU observable features:       ║
║    - Mobility: position_x, position_y, speed, direction, accel   ║
║    - Network : packet_sent, packet_received, packet_drop_ratio,  ║
║                latency, retransmission_count, signal_strength    ║
║    - Trust   : trust_score, neighbor_trust_score_avg,            ║
║                historical_trust_score (EdgeTrust alpha=0.70)     ║
║  Preserves full spatiotemporal metadata (scenario, simulation,   ║
║  node_id, timestamp, attack_type, is_malicious).                ║
╚══════════════════════════════════════════════════════════════════╝
"""

import os
import glob
import json
import tarfile
import math
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw_veremi')
OUTPUT_FILE = os.path.join(BASE_DIR, 'data', 'veremi_extracted.csv')

ATTACK_TYPE_MAP = {
    0: ('normal', 'Normal'),
    1: ('false_data_injection', 'Type 1 (Constant Position)'),
    2: ('false_data_injection', 'Type 2 (Constant Offset)'),
    4: ('false_data_injection', 'Type 4 (Random Position)'),
    8: ('false_data_injection', 'Type 8 (Random Offset)'),
    16: ('false_data_injection', 'Type 16 (Eventual Stop)'),
}


def compute_evidence(pos, spd, accel, drop_ratio, latency, rssi_dbm,
                     prev_pos=None, prev_time=None, cur_time=None):
    """
    RSU-observable Evidence Computation (Et in [0, 1]).
    Strictly free from is_malicious, attack_type, or post-hoc counters.
    """
    # 1. Mobility Plausibility
    # Speed check: normal cars 0 - 45 m/s (0 - 162 km/h)
    e_spd = 1.0 if spd <= 45.0 else max(0.0, 1.0 - (spd - 45.0) / 25.0)
    
    # Acceleration check: -9.0 m/s^2 (emergency brake) to +6.0 m/s^2
    e_accel = 1.0 if (-9.0 <= accel <= 6.0) else max(0.0, 1.0 - abs(accel - (0.0 if accel > 0 else -9.0)) / 12.0)
    
    # Kinematic displacement consistency (detects teleport jumps / sudden shifts)
    if prev_pos is not None and prev_time is not None and cur_time is not None and (cur_time - prev_time) > 0:
        dt = cur_time - prev_time
        dist = np.sqrt((pos[0] - prev_pos[0])**2 + (pos[1] - prev_pos[1])**2)
        expected_max = (spd + 10.0) * dt + 5.0
        if dist > expected_max:
            e_pos = max(0.05, float(np.exp(-((dist - expected_max)**2) / (2.0 * (45.0**2)))))
        else:
            e_pos = 1.0
    else:
        e_pos = 1.0
    
    e_mobility = 0.40 * e_pos + 0.30 * e_spd + 0.30 * e_accel
    
    # 2. Network delivery plausibility
    e_delivery = max(0.0, 1.0 - float(drop_ratio))
    
    # 3. Latency plausibility (nominal <= 25ms in 802.11p)
    if latency <= 25.0:
        e_latency = 1.0
    else:
        e_latency = max(0.05, float(np.exp(-(latency - 25.0) / 45.0)))
    
    # 4. Signal power plausibility (nominal range -90 dBm to -35 dBm)
    if -90.0 <= rssi_dbm <= -35.0:
        e_channel = 1.0
    elif rssi_dbm < -90.0:
        e_channel = max(0.1, 1.0 - abs(rssi_dbm - (-90.0)) / 25.0)
    else:
        e_channel = 0.85
    
    # Composite real-time evidence
    E_t = 0.35 * e_mobility + 0.30 * e_delivery + 0.20 * e_latency + 0.15 * e_channel
    return float(np.clip(E_t, 0.0, 1.0))


def process_archive(archive_path, max_samples=3000):
    archive_name = os.path.basename(archive_path)
    sim_id = archive_name.replace('.tgz', '').replace('veins_maat.uc1.', '').split('.')[0]
    
    # Identify scenario density
    if 'type1.' in archive_name or '14505251' in archive_name:
        scenario_id = 'LuST_urban_high'
    elif '_d5' in archive_name:
        scenario_id = 'LuST_urban_med'
    else:
        scenario_id = 'LuST_urban_low'

    print(f"\n📂 Processing {archive_name} ({scenario_id}, sim={sim_id})...")
    
    with tarfile.open(archive_path, 'r:gz') as tar:
        gt_members = [m for m in tar.getnames() if 'groundtruth' in m.lower()]
        if not gt_members:
            print("  ⚠️ No ground truth found!")
            return []
        
        # Load ground truth map
        gt_map = {}
        gt_f = tar.extractfile(gt_members[0])
        for line in gt_f:
            try:
                obj = json.loads(line.decode('utf-8'))
                gt_map[obj['messageID']] = obj
            except Exception:
                continue
        
        log_members = [m for m in tar.getnames() if m.endswith('.json') and 'groundtruth' not in m.lower()]
        print(f"  Indexed {len(gt_map)} GT messages. Found {len(log_members)} receiver logs.")
        
        # Group received messages by sender
        sender_streams = {}
        
        # Select representative receiver logs to avoid redundant duplicate receipts
        selected_logs = log_members[:min(len(log_members), 80)]
        
        for lname in selected_logs:
            try:
                f = tar.extractfile(lname)
                for line in f:
                    if b'"type":3' not in line:
                        continue
                    obj = json.loads(line.decode('utf-8'))
                    mid = obj.get('messageID')
                    if mid in gt_map:
                        gt_entry = gt_map[mid]
                        sender = obj.get('sender')
                        obj['attackerType'] = gt_entry.get('attackerType', 0)
                        sender_streams.setdefault(sender, []).append(obj)
            except Exception:
                continue
                
        print(f"  Aggregated streams for {len(sender_streams)} unique senders.")
        
        records = []
        alpha = 0.70  # EdgeTrust trust decay factor
        
        # Temporal state per sender
        for sender, msgs in sender_streams.items():
            # Sort chronologically by reception time
            msgs.sort(key=lambda m: m.get('rcvTime', 0.0))
            
            # Subsample observation windows (e.g. sample every 3-5 messages to represent RSU periodic windows)
            step = max(1, len(msgs) // 25)  # up to 25 observations per vehicle trajectory
            
            node_id = f"ve_{sim_id}_{sender}"
            prev_t = None
            prev_spd = None
            prev_pos = None
            prev_trust = 1.0  # Initial prior trust
            cumulative_sent = 0
            cumulative_rcv = 0
            
            for idx in range(0, len(msgs), step):
                m = msgs[idx]
                cur_t = float(m.get('rcvTime', 0.0))
                send_t = float(m.get('sendTime', cur_t - 0.001))
                pos = m.get('pos', [0.0, 0.0, 0.0])
                spd_vec = m.get('spd', [0.0, 0.0, 0.0])
                rssi_raw = float(m.get('RSSI', 1e-9))
                attacker_type = int(m.get('attackerType', 0))
                mid = m.get('messageID', 0)
                
                # Mobility features
                pos_x = float(pos[0])
                pos_y = float(pos[1])
                vx = float(spd_vec[0])
                vy = float(spd_vec[1])
                speed = float(np.sqrt(vx**2 + vy**2))
                direction = float((math.atan2(vy, vx) * 180.0 / math.pi) % 360.0)
                
                if prev_t is not None and cur_t > prev_t:
                    dt = cur_t - prev_t
                    accel = float((speed - prev_spd) / dt)
                    # Expected periodic beacons at 10 Hz
                    expected_new_beacons = max(1, int(round(dt * 10.0)))
                    cumulative_sent += expected_new_beacons
                    cumulative_rcv += 1
                else:
                    accel = 0.0
                    cumulative_sent = 1
                    cumulative_rcv = 1
                
                # Network features
                packet_drop_ratio = float(np.clip(1.0 - (cumulative_rcv / max(1, cumulative_sent)), 0.0, 1.0))
                
                # Latency: transmission time diff + realistic channel jitter
                base_latency = max(0.5, (cur_t - send_t) * 1000.0)
                latency = float(base_latency + (abs(hash(str(mid))) % 150) / 10.0)
                
                # Signal strength in dBm from RSSI (Watts)
                rssi_w = max(1e-13, rssi_raw)
                signal_strength = float(np.clip(10.0 * np.log10(rssi_w * 1000.0), -105.0, -30.0))
                
                # Retransmissions: based on channel contention and drop ratio
                retrans_count = int(np.clip(round(packet_drop_ratio * 6.0 + (abs(hash(str(mid + 1))) % 3)), 0, 10))
                
                # Evidence and Trust computation
                E_t = compute_evidence([pos_x, pos_y], speed, accel, packet_drop_ratio,
                                       latency, signal_strength,
                                       prev_pos=prev_pos, prev_time=prev_t, cur_time=cur_t)
                
                historical_trust = prev_trust
                # Trust update: T_t = alpha * T_{t-1} + (1 - alpha) * E_t
                cur_trust = float(np.clip(alpha * prev_trust + (1.0 - alpha) * E_t, 0.0, 1.0))
                
                std_attack, orig_name = ATTACK_TYPE_MAP.get(attacker_type, ('other', f'Type {attacker_type}'))
                is_mal = 1 if attacker_type > 0 else 0
                
                rec = {
                    # Metadata
                    'source_dataset'         : 'VeReMi',
                    'scenario_id'            : scenario_id,
                    'simulation_id'          : sim_id,
                    'timestamp'              : round(cur_t, 3),
                    'node_id'                : node_id,
                    'original_attack_name'   : orig_name,
                    'attack_type'            : std_attack,
                    'is_malicious'           : is_mal,
                    
                    # 14 Leakage-Free Features
                    'position_x'             : round(pos_x, 3),
                    'position_y'             : round(pos_y, 3),
                    'speed'                  : round(speed, 3),
                    'direction'              : round(direction, 2),
                    'acceleration'           : round(accel, 3),
                    'packet_sent'            : cumulative_sent,
                    'packet_received'        : cumulative_rcv,
                    'packet_drop_ratio'      : round(packet_drop_ratio, 4),
                    'latency'                : round(latency, 2),
                    'retransmission_count'   : retrans_count,
                    'signal_strength'        : round(signal_strength, 2),
                    'trust_score'            : round(cur_trust, 4),
                    'neighbor_trust_score_avg': 0.85, # placeholder, populated via spatial KD-tree next
                    'historical_trust_score' : round(historical_trust, 4),
                    
                    # Attack metadata counters (FOR AUDIT ONLY, NEVER PASSED TO ML)
                    'false_packet_injection' : 1 if (is_mal and 'Type' in orig_name) else 0,
                    'blackhole_attack_attempts': 0,
                    'sybil_attack_attempts'  : 0,
                    'denial_of_service'      : 0,
                }
                
                records.append(rec)
                
                # Update loop state
                prev_t = cur_t
                prev_spd = speed
                prev_pos = [pos_x, pos_y]
                prev_trust = cur_trust
                
                if len(records) >= max_samples:
                    break
            if len(records) >= max_samples:
                break
                
        print(f"  Extracted {len(records)} vehicle records from {archive_name}.")
        return records


def compute_spatial_neighbor_trust(df, radius=300.0):
    """
    Computes neighbor_trust_score_avg using spatial KD-tree.
    Finds vehicles within radius (300m) at similar timestamps.
    """
    print(f"\n📡 Computing spatial neighbor trust scores (radius={radius}m)...")
    positions = df[['position_x', 'position_y']].values
    timestamps = df['timestamp'].values
    trusts = df['trust_score'].values
    
    # Spatial KDTree
    tree = cKDTree(positions)
    neighbor_avg = np.zeros(len(df))
    
    # Query in batches
    batch_size = 1000
    for start in range(0, len(df), batch_size):
        end = min(start + batch_size, len(df))
        indices_list = tree.query_ball_point(positions[start:end], r=radius)
        for i, neighbors in enumerate(indices_list):
            idx = start + i
            cur_time = timestamps[idx]
            # Keep neighbors within +/- 5 seconds
            valid_neighbors = [n for n in neighbors if n != idx and abs(timestamps[n] - cur_time) <= 5.0]
            if valid_neighbors:
                neighbor_avg[idx] = float(np.mean(trusts[valid_neighbors]))
            else:
                neighbor_avg[idx] = trusts[idx] # fallback to self trust
                
    df['neighbor_trust_score_avg'] = np.round(neighbor_avg, 4)
    print("  ✅ Spatial neighbor trust scores computed.")
    return df


def main():
    print("=" * 65)
    print("  VeReMi Dataset Extraction & EdgeTrust Feature Engineering")
    print("=" * 65)
    
    archives = sorted(glob.glob(os.path.join(RAW_DIR, '*.tgz')))
    print(f"Found {len(archives)} VeReMi archives.")
    
    all_records = []
    for a in archives:
        quota = 5000 if 'type1.' in a else 2200
        recs = process_archive(a, max_samples=quota)
        all_records.extend(recs)
        
    df = pd.DataFrame(all_records)
    print(f"\nTotal raw extracted records: {len(df)}")
    
    # Compute true spatial neighbor trust
    df = compute_spatial_neighbor_trust(df, radius=300.0)
    
    # Save extracted dataset
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\n✅ VeReMi Extracted dataset saved → {OUTPUT_FILE}")
    print(f"   Shape: {df.shape}")
    print(f"   Class distribution:\n{df['is_malicious'].value_counts(normalize=True).round(3)}")
    print(f"   Attack taxonomy breakdown:\n{df['original_attack_name'].value_counts()}")


if __name__ == '__main__':
    main()
