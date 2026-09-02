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
  ['evaluator', 'arena', 'vanet'].forEach(s => {
    document.getElementById('sec-' + s).style.display = 'none';
  });
  document.getElementById('sec-' + name).style.display = 'flex';
  document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
  if (btn) btn.classList.add('active');
  if (name === 'arena') loadArena();
  if (name === 'vanet') loadVanet();
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
