/* ============================================================
   Sentry — AI Fraud Detection Dashboard
   Talks to the Flask API at localhost:5000. If it isn't running,
   falls back to an in-browser scoring model so the demo still works.
   ============================================================ */

const API_BASE = "http://localhost:5000/api";
let apiOnline = false;

const FEATURES = [
  "amount", "distance_from_home", "distance_from_last_transaction",
  "ratio_to_median_purchase", "repeat_retailer", "used_chip",
  "used_pin", "online_order", "hour_of_day",
];

// Importances taken from the actual trained RandomForest (model/train_model.py output)
const IMPORTANCE = [
  { key: "distance_from_last_transaction", label: "distance_from_last_transaction", value: 0.3389 },
  { key: "distance_from_home", label: "distance_from_home", value: 0.2811 },
  { key: "amount", label: "amount", value: 0.1841 },
  { key: "ratio_to_median_purchase", label: "ratio_to_median_purchase", value: 0.1260 },
  { key: "repeat_retailer", label: "repeat_retailer", value: 0.0396 },
  { key: "used_chip", label: "used_chip", value: 0.0128 },
  { key: "hour_of_day", label: "hour_of_day", value: 0.0075 },
  { key: "used_pin", label: "used_pin", value: 0.0060 },
  { key: "online_order", label: "online_order", value: 0.0040 },
];

/* ---------------- Offline fallback scoring model ----------------
   A logistic-style approximation of the trained model's behaviour,
   built from the same generative statistics used in data/generate_data.py.
------------------------------------------------------------------- */
function offlineScore(t) {
  let z = -5.2;
  z += 2.6 * sigmoidNorm(t.distance_from_last_transaction, 30, 25);
  z += 2.2 * sigmoidNorm(t.distance_from_home, 40, 30);
  z += 1.6 * sigmoidNorm(t.amount, 300, 220);
  z += 1.3 * sigmoidNorm(t.ratio_to_median_purchase, 3, 2);
  z += 0.9 * (1 - t.repeat_retailer);
  z += 0.7 * (1 - t.used_chip);
  z += 0.5 * (1 - t.used_pin);
  z += 0.9 * t.online_order;
  const nightBoost = (t.hour_of_day <= 5 || t.hour_of_day >= 22) ? 0.6 : 0;
  z += nightBoost;
  const p = 1 / (1 + Math.exp(-z));
  return Math.min(0.995, Math.max(0.004, p));
}
function sigmoidNorm(x, mid, scale) {
  return 1 / (1 + Math.exp(-((x - mid) / scale)));
}

function riskLevel(p) {
  if (p >= 0.75) return "high";
  if (p >= 0.4) return "medium";
  return "low";
}
function riskColor(level) {
  return { high: "var(--red)", medium: "var(--amber)", low: "var(--green)" }[level];
}

/* ---------------- API health check ---------------- */
async function checkApi() {
  try {
    const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(1500) });
    apiOnline = res.ok;
  } catch (e) {
    apiOnline = false;
  }
  document.getElementById("apiStatusText").textContent = apiOnline ? "live api connected" : "offline demo mode";
  document.getElementById("offlineNote").style.display = apiOnline ? "none" : "inline";
}

/* ---------------- Pulse strip (signature element) ---------------- */
const canvas = document.getElementById("pulseCanvas");
const ctx = canvas.getContext("2d");
let pulseData = Array.from({ length: 90 }, () => Math.random() * 0.15);
let statMonitored = 0, statBlocked = 0;

function resizeCanvas() {
  const rect = canvas.parentElement.getBoundingClientRect();
  canvas.width = rect.width * devicePixelRatio;
  canvas.height = rect.height * devicePixelRatio;
  canvas.style.width = rect.width + "px";
  canvas.style.height = rect.height + "px";
}
window.addEventListener("resize", resizeCanvas);
resizeCanvas();

function drawPulse() {
  const w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);

  // gridlines
  ctx.strokeStyle = "rgba(255,255,255,0.04)";
  ctx.lineWidth = 1;
  for (let i = 1; i < 4; i++) {
    const y = (h / 4) * i;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
  }

  const stepX = w / (pulseData.length - 1);
  ctx.beginPath();
  pulseData.forEach((v, i) => {
    const x = i * stepX;
    const y = h - v * h * 0.92 - h * 0.04;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  });
  const grad = ctx.createLinearGradient(0, 0, w, 0);
  grad.addColorStop(0, "#4C8DFF");
  grad.addColorStop(1, "#4C8DFF");
  ctx.strokeStyle = grad;
  ctx.lineWidth = 2 * devicePixelRatio;
  ctx.lineJoin = "round";
  ctx.stroke();

  // fill under curve
  ctx.lineTo(w, h); ctx.lineTo(0, h); ctx.closePath();
  const fillGrad = ctx.createLinearGradient(0, 0, 0, h);
  fillGrad.addColorStop(0, "rgba(76,141,255,0.18)");
  fillGrad.addColorStop(1, "rgba(76,141,255,0)");
  ctx.fillStyle = fillGrad;
  ctx.fill();

  // spike markers for high risk
  pulseData.forEach((v, i) => {
    if (v > 0.75) {
      const x = i * stepX;
      const y = h - v * h * 0.92 - h * 0.04;
      ctx.beginPath();
      ctx.arc(x, y, 3 * devicePixelRatio, 0, Math.PI * 2);
      ctx.fillStyle = "#FF5D5D";
      ctx.fill();
    }
  });
}

function tickPulse() {
  const isSpike = Math.random() < 0.06;
  const v = isSpike ? 0.7 + Math.random() * 0.3 : Math.random() * 0.35;
  pulseData.push(v);
  pulseData.shift();
  document.getElementById("pulseVal").textContent = v.toFixed(2);
  document.getElementById("pulseVal").style.color = riskColor(riskLevel(v));

  statMonitored += Math.floor(Math.random() * 4) + 1;
  if (isSpike) statBlocked += 1;
  document.getElementById("statMonitored").textContent = statMonitored.toLocaleString();
  document.getElementById("statBlocked").textContent = statBlocked.toLocaleString();

  drawPulse();
}
setInterval(tickPulse, 700);
drawPulse();

/* ---------------- Live transaction feed ---------------- */
function badgeHtml(level, prob) {
  return `<span class="badge ${level}"><i></i>${(prob * 100).toFixed(1)}%</span>`;
}
function barHtml(prob, level) {
  return `<span class="bar-track"><span class="bar-fill" style="width:${prob * 100}%; background:${riskColor(level)}"></span></span>`;
}

function sampleOfflineTransactions(n = 20) {
  const rows = [];
  for (let i = 0; i < n; i++) {
    const isFraud = Math.random() < 0.15;
    const t = isFraud ? {
      amount: +(Math.random() * 900 + 100).toFixed(2),
      distance_from_home: +(Math.random() * 250 + 40).toFixed(1),
      distance_from_last_transaction: +(Math.random() * 200 + 30).toFixed(1),
      ratio_to_median_purchase: +(Math.random() * 8 + 2).toFixed(2),
      repeat_retailer: Math.random() < 0.2 ? 1 : 0,
      used_chip: Math.random() < 0.15 ? 1 : 0,
      used_pin: Math.random() < 0.1 ? 1 : 0,
      online_order: Math.random() < 0.85 ? 1 : 0,
      hour_of_day: Math.floor(Math.random() * 24),
    } : {
      amount: +(Math.random() * 180 + 5).toFixed(2),
      distance_from_home: +(Math.random() * 15).toFixed(1),
      distance_from_last_transaction: +(Math.random() * 8).toFixed(1),
      ratio_to_median_purchase: +(Math.random() * 1.2 + 0.3).toFixed(2),
      repeat_retailer: Math.random() < 0.85 ? 1 : 0,
      used_chip: Math.random() < 0.7 ? 1 : 0,
      used_pin: Math.random() < 0.6 ? 1 : 0,
      online_order: Math.random() < 0.3 ? 1 : 0,
      hour_of_day: Math.floor(Math.random() * 24),
    };
    const prob = offlineScore(t);
    rows.push({
      transaction_id: "TXN" + (200000 + Math.floor(Math.random() * 99999)),
      amount: t.amount,
      distance_from_home: t.distance_from_home,
      online_order: !!t.online_order,
      used_chip: !!t.used_chip,
      predicted_fraud_probability: prob,
      predicted_fraud: prob >= 0.5,
    });
  }
  rows.sort((a, b) => b.predicted_fraud_probability - a.predicted_fraud_probability);
  return rows;
}

async function loadFeed() {
  let rows;
  if (apiOnline) {
    try {
      const res = await fetch(`${API_BASE}/transactions?limit=20`);
      rows = await res.json();
    } catch (e) {
      rows = sampleOfflineTransactions();
    }
  } else {
    rows = sampleOfflineTransactions();
  }

  const body = document.getElementById("feedBody");
  body.innerHTML = "";
  rows.forEach(r => {
    const level = riskLevel(r.predicted_fraud_probability);
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td class="id">${r.transaction_id}</td>
      <td class="amount">$${Number(r.amount).toFixed(2)}</td>
      <td>${Number(r.distance_from_home).toFixed(1)} km</td>
      <td>${r.online_order ? "Online" : (r.used_chip ? "Chip" : "Swipe")}</td>
      <td>${barHtml(r.predicted_fraud_probability, level)}${(r.predicted_fraud_probability * 100).toFixed(1)}%</td>
      <td>${badgeHtml(level, r.predicted_fraud_probability)}</td>
    `;
    body.appendChild(tr);
  });
  document.getElementById("feedCount").textContent = rows.length;
}
document.getElementById("refreshFeed").addEventListener("click", loadFeed);

/* ---------------- Check-a-transaction form ---------------- */
const sliders = {
  amount: document.getElementById("inAmount"),
  distance_from_home: document.getElementById("inDistHome"),
  distance_from_last_transaction: document.getElementById("inDistLast"),
  ratio_to_median_purchase: document.getElementById("inRatio"),
  hour_of_day: document.getElementById("inHour"),
};
const labels = {
  amount: document.getElementById("lblAmount"),
  distance_from_home: document.getElementById("lblDistHome"),
  distance_from_last_transaction: document.getElementById("lblDistLast"),
  ratio_to_median_purchase: document.getElementById("lblRatio"),
  hour_of_day: document.getElementById("lblHour"),
};
function refreshLabels() {
  labels.amount.textContent = "$" + Number(sliders.amount.value).toFixed(2);
  labels.distance_from_home.textContent = Number(sliders.distance_from_home.value).toFixed(1);
  labels.distance_from_last_transaction.textContent = Number(sliders.distance_from_last_transaction.value).toFixed(1);
  labels.ratio_to_median_purchase.textContent = Number(sliders.ratio_to_median_purchase.value).toFixed(1) + "\u00d7";
  const h = Number(sliders.hour_of_day.value);
  labels.hour_of_day.textContent = String(h).padStart(2, "0") + ":00";
}
Object.values(sliders).forEach(s => s.addEventListener("input", refreshLabels));
refreshLabels();

const toggles = {};
document.querySelectorAll(".toggle").forEach(el => {
  const key = el.dataset.key;
  toggles[key] = el.classList.contains("on") ? 1 : 0;
  el.addEventListener("click", () => {
    el.classList.toggle("on");
    toggles[key] = el.classList.contains("on") ? 1 : 0;
  });
});

function currentTransaction() {
  return {
    amount: Number(sliders.amount.value),
    distance_from_home: Number(sliders.distance_from_home.value),
    distance_from_last_transaction: Number(sliders.distance_from_last_transaction.value),
    ratio_to_median_purchase: Number(sliders.ratio_to_median_purchase.value),
    repeat_retailer: toggles.repeat_retailer,
    used_chip: toggles.used_chip,
    used_pin: toggles.used_pin,
    online_order: toggles.online_order,
    hour_of_day: Number(sliders.hour_of_day.value),
  };
}

function buildReasons(t, prob) {
  const reasons = [];
  if (t.distance_from_last_transaction > 40) reasons.push("Far from the location of the previous transaction");
  if (t.distance_from_home > 50) reasons.push("Far from the cardholder's home");
  if (t.ratio_to_median_purchase > 3) reasons.push("Amount is much higher than this cardholder's usual purchase");
  if (t.amount > 400) reasons.push("Unusually large transaction amount");
  if (!t.used_chip && !t.used_pin) reasons.push("No chip or PIN used to verify the card");
  if (t.online_order) reasons.push("Card-not-present online order");
  if (!t.repeat_retailer) reasons.push("First time at this retailer");
  if (t.hour_of_day <= 5 || t.hour_of_day >= 22) reasons.push("Occurred late at night");
  if (reasons.length === 0) {
    reasons.push("Matches the cardholder's typical spending pattern");
    reasons.push("Verified in person with chip or PIN");
  }
  return reasons.slice(0, 5);
}

async function analyzeTransaction() {
  const t = currentTransaction();
  let prob;
  if (apiOnline) {
    try {
      const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(t),
      });
      const data = await res.json();
      prob = data.fraud_probability;
    } catch (e) {
      prob = offlineScore(t);
    }
  } else {
    prob = offlineScore(t);
  }

  const level = riskLevel(prob);
  document.getElementById("resultEmpty").style.display = "none";
  document.getElementById("resultBody").style.display = "block";
  document.getElementById("resultPct").textContent = (prob * 100).toFixed(1) + "%";
  document.getElementById("resultPct").style.color = riskColor(level);

  const circumference = 540;
  const arc = document.getElementById("gaugeArc");
  arc.style.stroke = riskColor(level);
  arc.style.transition = "stroke-dashoffset 0.7s ease, stroke 0.3s";
  requestAnimationFrame(() => {
    arc.style.strokeDashoffset = circumference - circumference * prob;
  });

  const badge = document.getElementById("resultBadge");
  const verdictText = level === "high" ? "High risk — recommend blocking" : level === "medium" ? "Medium risk — recommend review" : "Low risk — likely legitimate";
  badge.className = "badge verdict-badge " + level;
  badge.innerHTML = `<i></i>${verdictText}`;

  const list = document.getElementById("reasonsList");
  list.innerHTML = "";
  buildReasons(t, prob).forEach(r => {
    const li = document.createElement("li");
    li.innerHTML = `<span class="ico">&#9679;</span><span>${r}</span>`;
    list.appendChild(li);
  });
}
document.getElementById("analyzeBtn").addEventListener("click", analyzeTransaction);

/* ---------------- Feature importance bars ---------------- */
function renderImportance() {
  const container = document.getElementById("importanceList");
  container.innerHTML = "";
  const max = IMPORTANCE[0].value;
  IMPORTANCE.forEach(item => {
    const row = document.createElement("div");
    row.className = "imp-row";
    row.innerHTML = `
      <div class="fname">${item.label}</div>
      <div class="imp-track"><div class="imp-fill" style="width:${(item.value / max) * 100}%"></div></div>
      <div class="imp-val">${(item.value * 100).toFixed(1)}%</div>
    `;
    container.appendChild(row);
  });
}
renderImportance();

/* ---------------- Init ---------------- */
(async function init() {
  await checkApi();
  await loadFeed();
})();
