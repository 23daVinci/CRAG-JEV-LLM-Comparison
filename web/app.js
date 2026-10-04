const state = {
  totals: { gemini: { latency: 0, cost: 0 }, jev: { latency: 0, cost: 0 } },
  results: { gemini: null, jev: null },
};

async function loadQueries() {
  const res = await fetch("/api/queries");
  const queries = await res.json();
  const select = document.getElementById("query-select");
  select.innerHTML = queries
    .map((q) => `<option value="${q.id}">[${q.type}] ${q.question}</option>`)
    .join("");
}

function resetColumn(variant) {
  document.getElementById(`nodes-${variant}`).innerHTML = "";
  document.getElementById(`totals-${variant}`).textContent = "";
  document.getElementById(`result-${variant}`).innerHTML = "";
  state.totals[variant] = { latency: 0, cost: 0 };
  state.results[variant] = null;
}

function appendNode(variant, payload) {
  const list = document.getElementById(`nodes-${variant}`);
  const li = document.createElement("li");
  li.innerHTML = `
    <span class="node-name">${payload.node}</span>
    <span>${payload.summary}</span>
    <span class="node-meta">${payload.latency_ms}ms · $${payload.cost_usd.toFixed(6)}</span>
  `;
  list.appendChild(li);

  state.totals[variant].latency += payload.latency_ms;
  state.totals[variant].cost += payload.cost_usd;
  document.getElementById(`totals-${variant}`).textContent =
    `total: ${state.totals[variant].latency.toFixed(1)}ms · $${state.totals[variant].cost.toFixed(6)}`;
}

function renderResult(variant, payload) {
  state.results[variant] = payload;
  const goldSet = new Set(payload.gold_titles);
  const selectedSet = new Set(payload.selected_titles);
  const rows = [...new Set([...payload.gold_titles, ...payload.selected_titles])]
    .sort()
    .map((title) => {
      const inGold = goldSet.has(title);
      const inSelected = selectedSet.has(title);
      const cls = inGold && inSelected ? "title-ok" : inGold ? "title-missing" : "";
      const marker = inGold && inSelected ? "✓" : inGold ? "✗ (missed)" : "+ (extra)";
      return `<div class="${cls}">${marker} ${title}</div>`;
    })
    .join("");
  document.getElementById(`result-${variant}`).innerHTML = `
    <div>precision ${payload.precision.toFixed(2)} · recall ${payload.recall.toFixed(2)} ·
    f1 ${payload.f1.toFixed(2)} · attempts ${payload.attempts}</div>
    ${rows}
  `;
  maybeRenderCompare();
}

function maybeRenderCompare() {
  if (!state.results.gemini || !state.results.jev) return;
  const maxLatency = Math.max(state.totals.gemini.latency, state.totals.jev.latency, 1);
  const maxCost = Math.max(state.totals.gemini.cost, state.totals.jev.cost, 1e-9);
  const bar = (label, variant, value, max, fmt) => `
    <div class="bar-row">
      <span>${label}</span>
      <div class="bar-track"><div class="bar-fill ${variant}" style="width:${(value / max) * 100}%"></div></div>
      <span>${fmt(value)}</span>
    </div>`;
  document.getElementById("compare").innerHTML = `
    <h3>This run</h3>
    ${bar("Gemini", "gemini", state.totals.gemini.latency, maxLatency, (v) => v.toFixed(0) + "ms")}
    ${bar("Jev", "jev", state.totals.jev.latency, maxLatency, (v) => v.toFixed(0) + "ms")}
    ${bar("Gemini", "gemini", state.totals.gemini.cost, maxCost, (v) => "$" + v.toFixed(6))}
    ${bar("Jev", "jev", state.totals.jev.cost, maxCost, (v) => "$" + v.toFixed(6))}
  `;
}

async function runQuery() {
  const queryId = document.getElementById("query-select").value;
  const status = document.getElementById("status");
  resetColumn("gemini");
  resetColumn("jev");
  document.getElementById("compare").innerHTML = "";
  status.textContent = "starting…";

  const res = await fetch("/api/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query_id: queryId }),
  });
  const { run_id } = await res.json();

  const source = new EventSource(`/api/runs/${run_id}/events`);
  status.textContent = "running…";

  source.addEventListener("node_finished", (e) => {
    const data = JSON.parse(e.data);
    appendNode(data.variant, data);
  });
  source.addEventListener("variant_finished", (e) => {
    const data = JSON.parse(e.data);
    renderResult(data.variant, data);
  });
  source.addEventListener("decision_error", (e) => appendError(e));
  source.addEventListener("run_finished", () => {
    status.textContent = "done";
    source.close();
  });
  source.onerror = () => {
    status.textContent = "connection closed";
    source.close();
  };
}

function appendError(e) {
  const data = JSON.parse(e.data);
  document.getElementById("status").textContent = `error: ${data.message}`;
}

document.getElementById("run-button").addEventListener("click", runQuery);
loadQueries();
