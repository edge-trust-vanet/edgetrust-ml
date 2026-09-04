/* ──────────────────────────────────────────────────────────
   EdgeTrust-VANET Dashboard JS
   ────────────────────────────────────────────────────────── */

// ── Clock ─────────────────────────────────────────────────
function updateClock() { document.getElementById('clock').textContent = new Date().toLocaleTimeString(); }
setInterval(updateClock, 1000); updateClock();

// ── Slider ────────────────────────────────────────────────
function updateSlider(sid, vid) {
  document.getElementById(vid).textContent = parseFloat(document.getElementById(sid).value).toFixed(2);
}

// ── Section Nav ───────────────────────────────────────────
function showSection(name, btn) {
  ['evaluator', 'arena', 'vanet', 'live-demo'].forEach(s => {
    const el = document.getElementById('sec-' + s);
    if (el) el.style.display = 'none';
  });
  const target = document.getElementById('sec-' + name);
  if (target) target.style.display = 'flex';
  document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
  if (btn) btn.classList.add('active');
  if (name === 'arena') loadArena();
  if (name === 'vanet') loadVanet();
  if (name === 'live-demo') initLiveDemo();
}

// ── Stats ─────────────────────────────────────────────────
async function refreshStats() {
  try {
    const d = await (await fetch('/api/stats')).json();
    animateCount('s-accepted', d.accepted);
    animateCount('s-warned', d.warned);
    animateCount('s-blocked', d.blocked);
    animateCount('s-total', d.total);
  } catch (e) { }
}

function animateCount(id, target) {
  const el = document.getElementById(id);
  const cur = parseInt(el.textContent) || 0;
  if (cur === target) return;
  const step = Math.sign(target - cur);
  let v = cur;
  const iv = setInterval(() => { el.textContent = v += step; if (v === target) clearInterval(iv); }, 40);
}

// ── Log ───────────────────────────────────────────────────
async function refreshLog() {
  try { renderLog(await (await fetch('/api/log')).json()); } catch (e) { }
}

function renderLog(items) {
  const c = document.getElementById('log-container');
  if (!items.length) { c.innerHTML = '<p class="log-empty">No evaluations yet.</p>'; return; }
  c.innerHTML = items.map(i => `
    <div class="log-item">
      <span class="log-verdict ${i.final_decision}">${i.final_decision}</span>
      <span class="log-vehicle">${i.vehicle_id}</span>
      <span class="log-detail">Trust: <b>${i.trust_score.toFixed(2)}</b> · ML: ${i.ml_prediction} (${i.ml_confidence}%)</span>
      <span class="log-time">${i.timestamp}</span>
    </div>`).join('');
}

// ── Evaluate ──────────────────────────────────────────────
async function evaluate(e) {
  e.preventDefault();
  const btn = document.getElementById('btn-eval');
  btn.disabled = true; btn.textContent = 'Evaluating…';
  const payload = {
    vehicle_id: document.getElementById('f-vid').value,
    speed: document.getElementById('f-speed').value,
    acceleration: document.getElementById('f-accel').value,
    position_x: document.getElementById('f-px').value,
    position_y: document.getElementById('f-py').value,
    heading: document.getElementById('f-heading').value,
    message_frequency: document.getElementById('f-freq').value,
    neighbor_count: document.getElementById('f-neighbors').value,
    message_consistency: document.getElementById('sl-mc').value,
    behavior_history: document.getElementById('sl-bh').value,
    neighbor_validation: document.getElementById('sl-nv').value,
    plausibility: document.getElementById('sl-pl').value,
  };
  try {
    const data = await (await fetch('/api/evaluate', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    })).json();
    showVerdict(data);
    showRadar(payload);
    refreshStats();
    refreshLog();
  } catch (err) {
    alert('Server error — is the Flask server running?');
  } finally {
    btn.disabled = false; btn.textContent = 'Evaluate Vehicle';
  }
}

// ── Simulate ──────────────────────────────────────────────
async function simulate() {
  try {
    const data = await (await fetch('/api/simulate', { method: 'POST' })).json();
    document.getElementById('f-vid').value = data.vehicle_id;
    document.getElementById('f-speed').value = data.features[0].toFixed(2);
    document.getElementById('f-accel').value = data.features[1].toFixed(2);
    document.getElementById('f-px').value = data.features[2].toFixed(2);
    document.getElementById('f-py').value = data.features[3].toFixed(2);
    document.getElementById('f-heading').value = data.features[4].toFixed(2);
    document.getElementById('f-freq').value = Math.round(data.features[5]);
    document.getElementById('f-neighbors').value = Math.round(data.features[6]);
    ['mc', 'bh', 'nv', 'pl'].forEach((k, i) => {
      const key = ['message_consistency', 'behavior_history', 'neighbor_validation', 'plausibility'][i];
      document.getElementById(`sl-${k}`).value = data.trust_data[key];
      updateSlider(`sl-${k}`, `sv-${k}`);
    });
    showVerdict(data);
    showRadar(data.trust_data);
    refreshStats();
    refreshLog();
  } catch (err) {
    alert('Simulation error');
  }
}

// ── Verdict ───────────────────────────────────────────────
function showVerdict(data) {
  const icons = { ACCEPT: '✅', WARN: '⚠️', BLOCK: '🚫' };
  const desc = {
    ACCEPT: 'Vehicle is trusted. Messages accepted.',
    WARN: 'Vehicle is suspicious. Priority reduced.',
    BLOCK: 'Vehicle blocked from the network.',
  };
  document.getElementById('verdict-panel').innerHTML = `
    <div class="verdict-result">
      <div class="verdict-badge ${data.final_decision}">${icons[data.final_decision]} ${data.final_decision}</div>
      <p style="color:var(--text-2);font-size:0.82rem;margin-bottom:16px">${desc[data.final_decision]}</p>
      <div class="verdict-meta">
        <div class="meta-chip"><span>ML:</span>${data.ml_prediction}</div>
        <div class="meta-chip"><span>Confidence:</span>${data.ml_confidence}%</div>
        <div class="meta-chip"><span>Trust Score:</span>${data.trust_score.toFixed(3)}</div>
        <div class="meta-chip"><span>Trust Label:</span>${data.trust_label}</div>
      </div>
    </div>`;
}

// ── Trust Radar ───────────────────────────────────────────
let radarChart = null;
function showRadar(td) {
  document.getElementById('trust-gauge-section').style.display = 'block';
  const ctx = document.getElementById('radarChart').getContext('2d');
  if (radarChart) radarChart.destroy();
  radarChart = new Chart(ctx, {
    type: 'radar',
    data: {
      labels: ['Msg Consistency', 'Behavior History', 'Neighbor Valid.', 'Plausibility'],
      datasets: [{
        label: 'Trust Factors',
        data: [+td.message_consistency, +td.behavior_history, +td.neighbor_validation, +td.plausibility],
        backgroundColor: 'rgba(59,130,246,0.15)',
        borderColor: '#3b82f6',
        borderWidth: 2,
        pointBackgroundColor: '#3b82f6',
        pointRadius: 4,
      }],
    },
    options: {
      responsive: true,
      scales: {
        r: {
          min: 0, max: 1,
          ticks: { display: false },
          grid: { color: 'rgba(255,255,255,0.06)' },
          angleLines: { color: 'rgba(255,255,255,0.06)' },
          pointLabels: { color: '#94a3b8', font: { size: 11, family: 'Inter' } },
        },
      },
      plugins: { legend: { display: false } },
    },
  });
}

// ── Baseline Perf Chart ───────────────────────────────────
function buildPerfChart() {
  const ctx = document.getElementById('perfChart').getContext('2d');
  const labels = ['PKI Only', 'ML Only', 'Trust Only', 'EdgeTrust-VANET'];
  const accent = i => i === 3 ? '#3b82f6' : 'rgba(148,163,184,0.3)';
  new Chart(ctx, {
    type: 'bar',
    data: {
      labels,
      datasets: [
        { label: 'Accuracy (%)', data: [72.1, 85.3, 78.6, 93.4], backgroundColor: labels.map((_, i) => accent(i)), borderRadius: 6, borderSkipped: false },
        { label: 'F1 Score (%)', data: [68.0, 83.1, 76.2, 91.8], backgroundColor: labels.map((_, i) => i === 3 ? '#7c3aed' : 'rgba(148,163,184,0.15)'), borderRadius: 6, borderSkipped: false },
      ],
    },
    options: {
      responsive: true,
      plugins: {
        legend: { labels: { color: '#94a3b8', font: { family: 'Inter', size: 12 } } },
        tooltip: { mode: 'index' },
      },
      scales: {
        x: { grid: { color: 'rgba(255,255,255,0.04)' }, ticks: { color: '#94a3b8', font: { family: 'Inter', size: 11 } } },
        y: { min: 60, max: 100, grid: { color: 'rgba(255,255,255,0.06)' }, ticks: { color: '#94a3b8', callback: v => v + '%', font: { family: 'Inter', size: 11 } } },
      },
    },
  });
}

function clearLog() { document.getElementById('log-container').innerHTML = '<p class="log-empty">No evaluations yet.</p>'; }

// ════════════════════════════════════════════════════════
//  SHARED MODEL RENDERING HELPERS
// ════════════════════════════════════════════════════════

function metricColor(v) { return v >= 90 ? 'good' : v >= 75 ? 'ok' : 'poor'; }
function barColor(v) { return v >= 90 ? '#22c55e' : v >= 75 ? '#f59e0b' : '#ef4444'; }

function modelType(name) {
  return ({
    'Random Forest': 'Ensemble · Trees',
    'SVM (RBF)': 'Kernel Method',
    'K-Nearest Neighbors': 'Instance-Based',
    'Logistic Regression': 'Linear · Fast',
    'Decision Tree': 'Tree · Interpretable',
    'Gaussian Naive Bayes': 'Probabilistic · Tiny',
    'AdaBoost': 'Boosting Ensemble',
    'Gradient Boosting': 'Boosting · Iterative',
    'Extra Trees': 'Randomized Trees',
    'XGBoost': 'Regularized GBDT',
    'LightGBM': 'Fast Histogram GBDT',
    'CatBoost': 'Symmetric Tree GBDT',
    'Hist Gradient Boosting': 'Histogram-Binned GBDT',
    'MLP Neural Network': 'Deep Tabular Neural Net',
    'Linear Discriminant Analysis': 'Generative · Linear Bayes',
    'Stacking Ensemble': 'Meta-Learner Ensemble',
    'Voting Ensemble': 'Soft Blend Ensemble',
  })[name] || 'ML Classifier';
}

function renderModelCards(models, gridId) {
  document.getElementById(gridId).innerHTML = models.map((m, i) => {
    const cm = m.confusion_matrix;
    const tp = cm[1]?.[1] ?? 0, fp = cm[0]?.[1] ?? 0, fn = cm[1]?.[0] ?? 0, tn = cm[0]?.[0] ?? 0;
    const rankClass = i < 3 ? `rank-${i + 1}` : '';
    return `
    <div class="model-card ${m.is_best ? 'best-card' : ''}">
      <span class="model-card-rank ${rankClass}">#${i + 1}</span>
      ${m.is_best ? '<div class="best-badge">🏆 Best Model</div>' : ''}
      <div class="model-name">${m.name}</div>
      <div class="model-type">${modelType(m.name)}</div>
      <div class="model-metrics">
        <div class="metric-box"><div class="m-label">Accuracy</div><div class="m-val ${metricColor(m.accuracy)}">${m.accuracy}%</div></div>
        <div class="metric-box"><div class="m-label">F1 Score</div><div class="m-val ${metricColor(m.f1_score)}">${m.f1_score}%</div></div>
        <div class="metric-box"><div class="m-label">Precision</div><div class="m-val ${metricColor(m.precision)}">${m.precision}%</div></div>
        <div class="metric-box"><div class="m-label">Recall</div><div class="m-val ${metricColor(m.recall)}">${m.recall}%</div></div>
      </div>
      ${['Accuracy', 'F1 Score', 'Precision', 'Recall'].map((label, j) => {
      const v = [m.accuracy, m.f1_score, m.precision, m.recall][j];
      return `<div class="mini-bar-wrap">
          <span class="ml">${label}</span>
          <div class="mini-bar-track"><div class="mini-bar-fill" style="width:${v}%;background:${barColor(v)}"></div></div>
          <span class="mr">${v}%</span>
        </div>`;
    }).join('')}
      <div class="cm-mini">
        <div class="cm-label">Confusion Matrix</div>
        <div class="cm-cell cm-tp" title="True Positive">TP: ${tp}</div>
        <div class="cm-cell cm-fp" title="False Positive">FP: ${fp}</div>
        <div class="cm-cell cm-fn" title="False Negative">FN: ${fn}</div>
        <div class="cm-cell cm-tn" title="True Negative">TN: ${tn}</div>
      </div>
    </div>`;
  }).join('');
}

function buildMegaChart(models, metric, canvasId, instanceKey) {
  if (window[instanceKey]) window[instanceKey].destroy();
  const ctx = document.getElementById(canvasId).getContext('2d');
  const labels = models.map(m => m.name.length > 18 ? m.name.substring(0, 16) + '…' : m.name);
  const values = models.map(m => m[metric]);
  const bgColors = models.map(m => m.is_best ? '#3b82f6' : 'rgba(148,163,184,0.25)');
  const metricLabels = {
    accuracy: 'Accuracy (%)', f1_score: 'F1 Score (%)',
    precision: 'Precision (%)', recall: 'Recall (%)', false_alert_rate: 'False Alert Rate (%)',
  };
  window[instanceKey] = new Chart(ctx, {
    type: 'bar',
    data: { labels, datasets: [{ label: metricLabels[metric] || metric, data: values, backgroundColor: bgColors, borderRadius: 8, borderSkipped: false }] },
    options: {
      indexAxis: 'y', responsive: true,
      plugins: { legend: { display: false }, tooltip: { callbacks: { label: c => ` ${c.parsed.x.toFixed(2)}%` } } },
      scales: {
        x: { min: metric === 'false_alert_rate' ? 0 : 50, max: 100, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#94a3b8', callback: v => v + '%', font: { family: 'Inter', size: 11 } } },
        y: { grid: { display: false }, ticks: { color: '#cbd5e1', font: { family: 'Inter', size: 11 } } },
      },
    },
  });
}

function switchMetric(metric, btn, canvasId, section) {
  btn.closest('.chart-tabs').querySelectorAll('.chart-tab').forEach(t => t.classList.remove('active'));
  btn.classList.add('active');
  const data = section === 'arena' ? arenaData : vanetData;
  if (data) buildMegaChart(data.models, metric, canvasId, section === 'arena' ? 'megaChartInst' : 'vanetChartInst');
}

function buildRadar(top5, canvasId, instanceKey) {
  if (window[instanceKey]) window[instanceKey].destroy();
  const ctx = document.getElementById(canvasId).getContext('2d');
  const palette = ['#3b82f6', '#22c55e', '#f59e0b', '#ef4444', '#a855f7'];
  window[instanceKey] = new Chart(ctx, {
    type: 'radar',
    data: {
      labels: ['Accuracy', 'F1 Score', 'Precision', 'Recall'],
      datasets: top5.map((m, i) => ({
        label: m.name,
        data: [m.accuracy, m.f1_score, m.precision, m.recall].map(v => v / 100),
        backgroundColor: palette[i] + '18',
        borderColor: palette[i],
        borderWidth: m.is_best ? 2.5 : 1.5,
        pointBackgroundColor: palette[i],
        pointRadius: m.is_best ? 5 : 3,
      })),
    },
    options: {
      responsive: true,
      scales: {
        r: {
          min: 0.5, max: 1.0,
          ticks: { display: false },
          grid: { color: 'rgba(255,255,255,0.06)' },
          angleLines: { color: 'rgba(255,255,255,0.06)' },
          pointLabels: { color: '#94a3b8', font: { size: 12, family: 'Inter', weight: '500' } },
        },
      },
      plugins: {
        legend: { position: 'bottom', labels: { color: '#94a3b8', font: { family: 'Inter', size: 11 }, boxWidth: 12, padding: 16 } },
      },
    },
  });
}

function buildSpotlight(data, elId) {
  const best = data.models.find(m => m.is_best);
  const reasons = data.why_best;
  document.getElementById(elId).innerHTML = `
    <div class="best-badge" style="margin-bottom:10px">🏆 Best Model — Ranked #1 by F1 Score</div>
    <div class="spotlight-name">${best.name}</div>
    <div style="color:var(--text-3);font-size:0.78rem;margin-bottom:16px">${modelType(best.name)}</div>
    <div class="spotlight-score">
      <div class="sc-pill acc" ><span class="sp-val">${best.accuracy}%</span><span class="sp-key">Accuracy</span></div>
      <div class="sc-pill f1"  ><span class="sp-val">${best.f1_score}%</span><span class="sp-key">F1 Score</span></div>
      <div class="sc-pill prec"><span class="sp-val">${best.precision}%</span><span class="sp-key">Precision</span></div>
      <div class="sc-pill rec" ><span class="sp-val">${best.recall}%</span><span class="sp-key">Recall</span></div>
    </div>
    <div class="why-title">Why This Model Wins</div>
    <ul class="why-list">${reasons.map(r => `<li>${r}</li>`).join('')}</ul>`;
}

// ════════════════════════════════════════════════════════
//  VEREMI ARENA
// ════════════════════════════════════════════════════════
let arenaData = null;

async function loadArena() {
  if (arenaData) return;
  document.getElementById('arena-loading').classList.remove('hidden');
  try {
    const res = await fetch('/api/models');
    if (!res.ok) {
      document.getElementById('arena-loading').innerHTML =
        '<span style="color:var(--yellow)">⚠️ Run <code>python scripts/train_model.py</code> in the terminal first</span>';
      return;
    }
    arenaData = await res.json();
    document.getElementById('arena-loading').classList.add('hidden');
    renderModelCards(arenaData.models, 'model-cards-grid');
    buildMegaChart(arenaData.models, 'accuracy', 'megaChart', 'megaChartInst');
    buildRadar(arenaData.models.slice(0, 5), 'arenaRadar', 'arenaRadarInst');
    buildSpotlight(arenaData, 'spotlight-body');
  } catch (e) {
    document.getElementById('arena-loading').innerHTML = '<span style="color:var(--red)">Error loading model data.</span>';
  }
}

// ════════════════════════════════════════════════════════
//  VANET NODES
// ════════════════════════════════════════════════════════
let vanetData = null;

async function loadVanet() {
  if (vanetData) return;
  document.getElementById('vanet-loading').classList.remove('hidden');
  try {
    const res = await fetch('/api/vanet_models');
    if (!res.ok) {
      document.getElementById('vanet-loading').innerHTML =
        '<span style="color:var(--yellow)">⚠️ Run <code>python scripts/analyze_vanet_nodes.py</code> in the terminal first</span>';
      return;
    }
    vanetData = await res.json();
    document.getElementById('vanet-loading').classList.add('hidden');
    renderModelCards(vanetData.models, 'vanet-cards-grid');
    buildMegaChart(vanetData.models, 'accuracy', 'vanetChart', 'vanetChartInst');
    buildRadar(vanetData.models.slice(0, 5), 'vanetRadar', 'vanetRadarInst');
    buildSpotlight(vanetData, 'vanet-spotlight');
    renderFeatureImportance(vanetData.feature_importance);
  } catch (e) {
    document.getElementById('vanet-loading').innerHTML = '<span style="color:var(--red)">Error loading VANET data.</span>';
  }
}

function renderFeatureImportance(fi) {
  if (!fi) return;
  const el = document.getElementById('fi-bars');
  const sorted = Object.entries(fi).sort((a, b) => b[1] - a[1]);
  const max = sorted[0][1];
  el.innerHTML = sorted.map(([name, val]) => `
    <div class="fi-bar-row">
      <span class="fi-name">${name}</span>
      <div class="fi-bar-track"><div class="fi-bar-fill" style="width:${(val / max * 100).toFixed(1)}%"></div></div>
      <span class="fi-val">${val.toFixed(2)}%</span>
    </div>`).join('');
}

// ── Init ──────────────────────────────────────────────────
buildPerfChart();
refreshStats();

// ══════════════════════════════════════════════════════════
// ── SECTION 4: LIVE PANEL DEMO CONTROLLER & VISUALIZER ───
// ══════════════════════════════════════════════════════════
let simFeed = null;
let currentPktIdx = 0;
let isPlaying = false;
let playTimer = null;
let simSpeed = 800;
let feedFilter = 'all';

async function initLiveDemo() {
  if (!simFeed) {
    await fetchSimFeed();
  } else {
    displayPacket(currentPktIdx);
  }
}

async function fetchSimFeed() {
  try {
    const res = await fetch('/api/live_sim_feed');
    if (!res.ok) {
      alert('Could not fetch simulation feed. Please ensure simulation has run.');
      return;
    }
    simFeed = await res.json();
    renderFeedTable();
    if (simFeed.packets && simFeed.packets.length > 0) {
      displayPacket(0);
    }
  } catch (e) {
    console.error('Error fetching live simulation feed:', e);
  }
}

function togglePlay() {
  const btn = document.getElementById('btn-play-pause');
  if (isPlaying) {
    clearInterval(playTimer);
    isPlaying = false;
    btn.textContent = '▶ Play';
    btn.style.background = '#2563eb';
  } else {
    if (!simFeed || !simFeed.packets.length) return;
    isPlaying = true;
    btn.textContent = '⏸ Pause';
    btn.style.background = '#f59e0b';
    playTimer = setInterval(() => {
      stepNext();
    }, simSpeed);
  }
}

function changeSpeed(val) {
  simSpeed = parseInt(val);
  if (isPlaying) {
    clearInterval(playTimer);
    playTimer = setInterval(stepNext, simSpeed);
  }
}

function stepNext() {
  if (!simFeed || !simFeed.packets.length) return;
  currentPktIdx = (currentPktIdx + 1) % simFeed.packets.length;
  displayPacket(currentPktIdx);
}

function stepPrev() {
  if (!simFeed || !simFeed.packets.length) return;
  currentPktIdx = (currentPktIdx - 1 + simFeed.packets.length) % simFeed.packets.length;
  displayPacket(currentPktIdx);
}

function jumpToFirstAttack() {
  if (!simFeed || !simFeed.packets.length) return;
  const attackIdx = simFeed.packets.findIndex(p => p.is_malicious === 1 || p.verdict === 'BLOCK');
  if (attackIdx !== -1) {
    if (isPlaying) togglePlay(); // Pause for panel explanation
    currentPktIdx = attackIdx;
    displayPacket(currentPktIdx);
  } else {
    alert('No attack packets found in current feed.');
  }
}

function selectPacket(idx) {
  if (isPlaying) togglePlay();
  currentPktIdx = idx;
  displayPacket(currentPktIdx);
}

function displayPacket(idx) {
  if (!simFeed || !simFeed.packets[idx]) return;
  const p = simFeed.packets[idx];

  // Update badge & title
  document.getElementById('pkt-counter-badge').textContent = `Packet ${p.packet_id} / ${simFeed.total_packets} (Node ${p.node_id})`;

  // Decision Card & Colors
  const decTitle = document.getElementById('demo-decision-title');
  const decCard  = document.getElementById('demo-decision-card');
  const atkTag   = document.getElementById('demo-attack-type');

  decTitle.textContent = p.verdict;
  atkTag.textContent   = p.attack_name;

  if (p.verdict === 'BLOCK') {
    decTitle.style.color = '#ef4444';
    decCard.style.background = 'rgba(239, 68, 68, 0.15)';
    decCard.style.borderColor = '#ef4444';
    atkTag.style.color = '#f87171';
    atkTag.style.background = 'rgba(239, 68, 68, 0.25)';
  } else if (p.verdict === 'WARN') {
    decTitle.style.color = '#f59e0b';
    decCard.style.background = 'rgba(245, 158, 11, 0.15)';
    decCard.style.borderColor = '#f59e0b';
    atkTag.style.color = '#fbbf24';
    atkTag.style.background = 'rgba(245, 158, 11, 0.25)';
  } else {
    decTitle.style.color = '#10b981';
    decCard.style.background = 'rgba(16, 185, 129, 0.12)';
    decCard.style.borderColor = '#10b981';
    atkTag.style.color = '#34d399';
    atkTag.style.background = 'rgba(16, 185, 129, 0.2)';
  }

  // AdaBoost Meter
  const adaPct = (p.ml_confidence * 100).toFixed(1);
  document.getElementById('demo-adaboost-pct').textContent = adaPct + '%';
  const adaFill = document.getElementById('demo-adaboost-fill');
  adaFill.style.width = Math.min(100, Math.max(5, adaPct)) + '%';
  adaFill.style.background = p.ml_confidence > 0.5 ? '#ef4444' : (p.ml_confidence > 0.3 ? '#f59e0b' : '#10b981');
  document.getElementById('demo-adaboost-pct').style.color = adaFill.style.background;

  // Trust Score Meter
  document.getElementById('demo-trust-score').textContent = p.trust_score.toFixed(4);
  const trustFill = document.getElementById('demo-trust-fill');
  const trustPct = Math.min(100, Math.max(5, p.trust_score * 100));
  trustFill.style.width = trustPct + '%';
  trustFill.style.background = p.trust_score < 0.40 ? '#ef4444' : (p.trust_score < 0.70 ? '#f59e0b' : '#10b981');
  document.getElementById('demo-trust-score').style.color = trustFill.style.background;

  // Telemetry Grid
  document.getElementById('demo-v-id').textContent    = 'Vehicle ' + p.node_id;
  document.getElementById('demo-v-speed').textContent = p.speed.toFixed(1) + ' m/s (' + (p.speed * 3.6).toFixed(1) + ' km/h)';
  document.getElementById('demo-v-pos').textContent   = `(${p.position_x.toFixed(1)}, ${p.position_y.toFixed(1)})`;
  document.getElementById('demo-v-dist').textContent  = p.dist_to_rsu.toFixed(1) + ' m ' + (p.in_coverage ? '✓ (In Range)' : '⚠ (Out)');
  document.getElementById('demo-v-rssi').textContent  = p.signal_strength.toFixed(1) + ' dBm';
  document.getElementById('demo-v-drop').textContent  = (p.packet_drop_ratio * 100).toFixed(1) + '%';

  // Highlight active table row
  document.querySelectorAll('#demo-table-body tr').forEach(tr => tr.style.background = 'transparent');
  const activeRow = document.getElementById('row-pkt-' + p.packet_id);
  if (activeRow) {
    activeRow.style.background = 'rgba(59, 130, 246, 0.25)';
    activeRow.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
  }

  // Draw 2D Simulation Canvas
  renderSimCanvas(p);
}

function renderSimCanvas(p) {
  const canvas = document.getElementById('simCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);

  // Coordinate projection from OMNeT++ (X: 10..140, Y: 10..120) to Canvas (680x500)
  const minX = 15, maxX = 145;
  const minY = 15, maxY = 120;
  function toCanvasX(x) { return 40 + ((x - minX) / (maxX - minX)) * (w - 80); }
  function toCanvasY(y) { return (h - 40) - ((y - minY) / (maxY - minY)) * (h - 80); }

  const rsuX = toCanvasX(58.0);
  const rsuY = toCanvasY(49.0);

  // 1. Draw Road Network (4-way intersection)
  ctx.fillStyle = '#1e293b';
  // Horizontal Road (East-West)
  ctx.fillRect(0, rsuY - 32, w, 64);
  // Vertical Road (North-South)
  ctx.fillRect(rsuX - 32, 0, 64, h);

  // Road markings (yellow dashed lines)
  ctx.strokeStyle = '#eab308';
  ctx.lineWidth = 2;
  ctx.setLineDash([8, 8]);

  // Horizontal lane center
  ctx.beginPath();
  ctx.moveTo(0, rsuY);
  ctx.lineTo(rsuX - 32, rsuY);
  ctx.moveTo(rsuX + 32, rsuY);
  ctx.lineTo(w, rsuY);
  ctx.stroke();

  // Vertical lane center
  ctx.beginPath();
  ctx.moveTo(rsuX, 0);
  ctx.lineTo(rsuX, rsuY - 32);
  ctx.moveTo(rsuX, rsuY + 32);
  ctx.lineTo(rsuX, h);
  ctx.stroke();
  ctx.setLineDash([]); // Reset line dash

  // Intersection boundary box
  ctx.strokeStyle = 'rgba(255, 255, 255, 0.15)';
  ctx.strokeRect(rsuX - 32, rsuY - 32, 64, 64);

  // 2. Draw 85m Radio Coverage Zone
  const coverageRadiusPx = ((85.0) / (maxX - minX)) * (w - 80);
  ctx.beginPath();
  ctx.arc(rsuX, rsuY, coverageRadiusPx, 0, Math.PI * 2);
  ctx.fillStyle = 'rgba(6, 182, 212, 0.06)';
  ctx.fill();
  ctx.strokeStyle = 'rgba(6, 182, 212, 0.4)';
  ctx.lineWidth = 1.5;
  ctx.setLineDash([6, 6]);
  ctx.stroke();
  ctx.setLineDash([]);

  // 3. Draw All Background Vehicles in the Stream
  const latestVehicles = {};
  simFeed.packets.slice(0, currentPktIdx + 1).forEach(pkt => {
    latestVehicles[pkt.node_id] = pkt;
  });

  Object.values(latestVehicles).forEach(v => {
    if (v.node_id === p.node_id) return; // Drawn as active later
    const vx = toCanvasX(v.position_x);
    const vy = toCanvasY(v.position_y);

    ctx.fillStyle = (v.is_malicious === 1) ? '#ef4444' : '#10b981';
    ctx.beginPath();
    ctx.arc(vx, vy, 7, 0, Math.PI * 2);
    ctx.fill();

    ctx.fillStyle = '#ffffff';
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.fillText('V' + v.node_id, vx - 7, vy - 10);
  });

  // 4. Draw Active Vehicle & Glowing Halo
  const curX = toCanvasX(p.position_x);
  const curY = toCanvasY(p.position_y);
  const isAttack = (p.is_malicious === 1 || p.verdict === 'BLOCK');

  // Halo pulse
  ctx.beginPath();
  ctx.arc(curX, curY, 16, 0, Math.PI * 2);
  ctx.fillStyle = isAttack ? 'rgba(239, 68, 68, 0.35)' : 'rgba(16, 185, 129, 0.35)';
  ctx.fill();

  // Vehicle Body
  ctx.beginPath();
  ctx.arc(curX, curY, 9, 0, Math.PI * 2);
  ctx.fillStyle = isAttack ? '#ef4444' : '#10b981';
  ctx.fill();
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 1.8;
  ctx.stroke();

  // Active Vehicle Label
  ctx.fillStyle = '#ffffff';
  ctx.font = 'bold 11px JetBrains Mono, monospace';
  ctx.fillText('V' + p.node_id, curX - 8, curY - 14);

  // 5. Draw DIRECTION ARROWS (From Vehicle -> RSU)
  const arrowColor = isAttack ? '#ef4444' : (p.verdict === 'WARN' ? '#f59e0b' : '#10b981');
  ctx.strokeStyle = arrowColor;
  ctx.fillStyle   = arrowColor;
  ctx.lineWidth   = isAttack ? 3.5 : 2.5;

  // Draw arrow line
  ctx.beginPath();
  ctx.moveTo(curX, curY);
  ctx.lineTo(rsuX, rsuY);
  ctx.stroke();

  // Arrowhead calculation pointing at RSU
  const angle = Math.atan2(rsuY - curY, rsuX - curX);
  const headLen = 14;
  const arrowTipX = rsuX - 16 * Math.cos(angle);
  const arrowTipY = rsuY - 16 * Math.sin(angle);

  ctx.beginPath();
  ctx.moveTo(arrowTipX, arrowTipY);
  ctx.lineTo(arrowTipX - headLen * Math.cos(angle - Math.PI / 6), arrowTipY - headLen * Math.sin(angle - Math.PI / 6));
  ctx.lineTo(arrowTipX - headLen * Math.cos(angle + Math.PI / 6), arrowTipY - headLen * Math.sin(angle + Math.PI / 6));
  ctx.closePath();
  ctx.fill();

  // If BLOCK, draw radiating blue safety advisory waves from RSU
  if (p.verdict === 'BLOCK') {
    ctx.strokeStyle = 'rgba(59, 130, 246, 0.8)';
    ctx.lineWidth = 2.0;
    ctx.beginPath();
    ctx.arc(rsuX, rsuY, 35, 0, Math.PI * 2);
    ctx.stroke();
    ctx.beginPath();
    ctx.arc(rsuX, rsuY, 60, 0, Math.PI * 2);
    ctx.stroke();
  }

  // 6. Draw Central RSU Tower Icon
  ctx.fillStyle = '#06b6d4';
  ctx.beginPath();
  ctx.arc(rsuX, rsuY, 12, 0, Math.PI * 2);
  ctx.fill();
  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 2;
  ctx.stroke();

  ctx.fillStyle = '#0891b2';
  ctx.beginPath();
  ctx.arc(rsuX, rsuY, 6, 0, Math.PI * 2);
  ctx.fill();

  ctx.fillStyle = '#06b6d4';
  ctx.font = 'bold 11px Inter, sans-serif';
  ctx.fillText('RSU 0 [Edge AI]', rsuX - 44, rsuY + 24);

  // 7. Render Lingering Speech Bubbles on Canvas
  // Vehicle Bubble
  const vMsg = isAttack ? `🚨 FDI: Falsified Accident (+28m, 0 m/s)` : `💬 V2V: Routine Telemetry (${p.speed.toFixed(1)} m/s)`;
  drawCanvasBubble(ctx, curX, curY - 26, vMsg, isAttack ? '#ef4444' : '#10b981');

  // RSU Decision Bubble
  const rsuMsg = (p.verdict === 'BLOCK') ? `🛡 AdaBoost: BLOCK [Malicious ${(p.ml_confidence * 100).toFixed(0)}%]` :
                 (p.verdict === 'WARN'  ? `⚠️ AdaBoost: WARN [Suspicious]` : `✓ AdaBoost: ACCEPT [Verified]`);
  drawCanvasBubble(ctx, rsuX, rsuY - 26, rsuMsg, arrowColor);
}

function drawCanvasBubble(ctx, x, y, text, color) {
  ctx.font = '11px Inter, sans-serif';
  const textWidth = ctx.measureText(text).width;
  const bubbleW = textWidth + 16;
  const bubbleH = 22;
  const bx = x - bubbleW / 2;
  const by = y - bubbleH;

  // Bubble background
  ctx.fillStyle = 'rgba(15, 23, 42, 0.92)';
  ctx.strokeStyle = color;
  ctx.lineWidth = 1.5;
  ctx.beginPath();
  ctx.roundRect(bx, by, bubbleW, bubbleH, 6);
  ctx.fill();
  ctx.stroke();

  // Pointer triangle
  ctx.fillStyle = 'rgba(15, 23, 42, 0.92)';
  ctx.beginPath();
  ctx.moveTo(x - 5, by + bubbleH);
  ctx.lineTo(x, by + bubbleH + 5);
  ctx.lineTo(x + 5, by + bubbleH);
  ctx.closePath();
  ctx.fill();

  // Text
  ctx.fillStyle = '#ffffff';
  ctx.fillText(text, bx + 8, by + 15);
}

function renderFeedTable() {
  const tbody = document.getElementById('demo-table-body');
  if (!simFeed || !simFeed.packets.length) {
    tbody.innerHTML = '<tr><td colspan="9" class="log-empty">No telemetry records available.</td></tr>';
    return;
  }

  let list = simFeed.packets;
  if (feedFilter === 'malicious') list = list.filter(p => p.is_malicious === 1);
  if (feedFilter === 'blocked')   list = list.filter(p => p.verdict === 'BLOCK');

  tbody.innerHTML = list.map(p => `
    <tr id="row-pkt-${p.packet_id}" onclick="selectPacket(${p.packet_id - 1})" style="cursor:pointer; border-bottom:1px solid rgba(255,255,255,0.04);">
      <td style="padding:6px 8px; font-family:'JetBrains Mono';">${p.packet_id}</td>
      <td style="padding:6px 8px;"><strong>V${p.node_id}</strong></td>
      <td style="padding:6px 8px;">${p.speed.toFixed(1)} m/s</td>
      <td style="padding:6px 8px; font-size:0.72rem; color:#94a3b8;">(${p.position_x.toFixed(0)}, ${p.position_y.toFixed(0)})</td>
      <td style="padding:6px 8px;">${p.signal_strength.toFixed(1)} dBm</td>
      <td style="padding:6px 8px;"><span style="color:${p.trust_score < 0.4 ? '#ef4444' : '#10b981'}">${p.trust_score.toFixed(3)}</span></td>
      <td style="padding:6px 8px;"><span style="color:${p.is_malicious ? '#f87171' : '#cbd5e1'}">${p.attack_name}</span></td>
      <td style="padding:6px 8px;"><strong>${(p.ml_confidence * 100).toFixed(1)}%</strong></td>
      <td style="padding:6px 8px;"><span class="table-badge ${p.verdict.toLowerCase()}">${p.verdict}</span></td>
    </tr>
  `).join('');
}

function filterFeed(filter, btn) {
  feedFilter = filter;
  document.querySelectorAll('.btn-demo').forEach(b => b.classList.remove('active'));
  if (btn) btn.classList.add('active');
  renderFeedTable();
}
