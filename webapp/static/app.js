const $ = (s) => document.querySelector(s);
const state = { docs: [], busy: false, engine: "jev_okf" };
try { state.engine = localStorage.getItem("engine") || "jev_okf"; } catch (e) {}
const ENGINE = { jev_okf: "JEV + OKF", rag: "Agentic RAG" };
const ROUTE = { doc_question: "Read as a question about your document", review: "Read as a document review",
  law_question: "Read as a general question of law", out_of_scope: "Outside what LexAgent covers" };
const STARTERS = [
  "Can my company enforce a non-compete after I resign?",
  "How soon must my final settlement be paid?",
  "Am I eligible for gratuity on a fixed-term contract?",
  "How long is maternity leave under the law?",
];

const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
function toast(msg) { const t = $("#toast"); t.textContent = msg; t.hidden = false; clearTimeout(t._h); t._h = setTimeout(() => (t.hidden = true), 4000); }
let accessCode = "";
try { accessCode = sessionStorage.getItem("accessCode") || ""; } catch (e) {}
function askCode(wrong) {
  return new Promise((resolve) => {
    const d = $("#code-dlg"), inp = $("#code-input");
    $("#code-msg").textContent = wrong ? "That code didn't work. Check it and try again." : "This app is shared privately. Enter the access code you were given.";
    inp.value = ""; d.showModal(); inp.focus();
    inp.onkeydown = (e) => { if (e.key === "Enter") { e.preventDefault(); d.close("ok"); } };
    d.onclose = () => resolve(d.returnValue === "ok" ? inp.value.trim() : null);
  });
}
async function api(path, opts = {}, retry = true) {
  const headers = Object.assign({}, opts.headers || {}, accessCode ? { "X-Access-Code": accessCode } : {});
  const r = await fetch(path, Object.assign({}, opts, { headers }));
  let data = {}; try { data = await r.json(); } catch (e) {}
  if (r.status === 401 && data.needs_code && retry) {
    const code = await askCode(!!accessCode);
    if (!code) throw new Error("Enter the access code to use this app.");
    accessCode = code; try { sessionStorage.setItem("accessCode", code); } catch (e) {}
    return api(path, opts, true);
  }
  if (!r.ok) throw new Error(data.error || `The server replied ${r.status}.`);
  return data;
}

/* ---------- documents ---------- */
function renderChips() {
  const box = $("#attached");
  box.innerHTML = state.docs.map((d) => d.loading
    ? `<span class="chip loading"><span class="nm">${esc(d.name)}</span><span class="sub">reading…</span></span>`
    : `<span class="chip"><span class="nm" title="${esc(d.name)}">${esc(d.name)}</span><span class="sub">${d.clauses} clauses</span>
       <button type="button" class="rv" data-review="${d.id}">Review</button>
       <button type="button" data-remove="${d.id}" aria-label="Remove ${esc(d.name)}">✕</button></span>`).join("");
  $("#q").placeholder = state.docs.some((d) => !d.loading) ? "Ask about this document or the law" : "Ask about your document or the law";
}
async function addDoc(fileOrSample) {
  const isFile = fileOrSample instanceof File;
  const tmp = { id: "tmp" + Date.now(), name: isFile ? fileOrSample.name : fileOrSample, loading: true };
  state.docs.push(tmp); renderChips();
  try {
    let d;
    if (isFile) { const fd = new FormData(); fd.append("file", fileOrSample); d = await api("/api/upload", { method: "POST", body: fd }); }
    else d = await api(`/api/samples/${encodeURIComponent(fileOrSample)}`, { method: "POST" });
    Object.assign(tmp, d, { loading: false });
    toast(`${d.name} added. Ask a question or choose Review.`);
    $("#q").focus();
  } catch (e) { state.docs = state.docs.filter((x) => x !== tmp); toast(e.message); }
  renderChips();
}
for (const id of ["#file", "#file2"]) $(id).addEventListener("change", (e) => { const f = e.target.files[0]; if (f) addDoc(f); e.target.value = ""; });
const drop = $("#drop");
drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("over"); });
drop.addEventListener("dragleave", () => drop.classList.remove("over"));
drop.addEventListener("drop", (e) => { e.preventDefault(); drop.classList.remove("over"); const f = e.dataTransfer.files[0]; if (f) addDoc(f); });
$("#sample-btn").addEventListener("click", async () => {
  try { const names = await api("/api/samples"); const pick = names.find((n) => n.includes("noncompete")) || names[0];
    if (pick) addDoc(pick); else toast("No sample documents were found in data\\samples."); } catch (e) { toast(e.message); }
});
$("#attached").addEventListener("click", (e) => {
  const rm = e.target.closest("[data-remove]"), rv = e.target.closest("[data-review]");
  if (rm) { const id = rm.dataset.remove; state.docs = state.docs.filter((d) => d.id !== id); api(`/api/docs/${id}`, { method: "DELETE" }).catch(() => {}); renderChips(); }
  if (rv) { const d = state.docs.find((x) => x.id === rv.dataset.review); ask(`Review ${d.name}`, { review: true, docs: [d.id] }); }
});

/* ---------- answer rendering ---------- */
const CITE = /\[((?:[LD]\d+)(?:\s*[,;]\s*[LD]\d+)*)([^\]]*)\]/g;
function inline(text) {
  return esc(text).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(CITE, (m, labels) => labels.match(/[LD]\d+/g)
      .map((l) => `<button type="button" class="cite${l[0] === "D" ? " doc" : ""}" data-cite="${l}" aria-pressed="false">${l}</button>`).join(""));
}
function renderAnswer(md) {
  md = md.replace(/\n?\s*This is general information, not legal advice\.?\s*$/i, "");
  const out = []; let list = null;
  const close = () => { if (list) { out.push(`</${list}>`); list = null; } };
  for (const raw of md.split(/\n/)) {
    const line = raw.trim();
    if (!line) { close(); continue; }
    let m;
    if ((m = line.match(/^[-*•]\s+(.*)/))) { if (list !== "ul") { close(); out.push("<ul>"); list = "ul"; } out.push(`<li><span class="claim">${inline(m[1])}</span></li>`); continue; }
    if ((m = line.match(/^\d+[.)]\s+(.*)/))) { if (list !== "ol") { close(); out.push("<ol>"); list = "ol"; } out.push(`<li><span class="claim">${inline(m[1])}</span></li>`); continue; }
    close();
    if ((m = line.match(/^#{1,6}\s+(.*)/))) { out.push(`<p><strong>${inline(m[1])}</strong></p>`); continue; }
    out.push(`<p><span class="claim">${inline(line)}</span></p>`);
  }
  close();
  return out.join("");
}
function plain(el) { const c = el.cloneNode(true); c.querySelectorAll(".cite").forEach((b) => b.remove()); return c.textContent.replace(/\s+/g, " ").replace(/\s+([.,;:])/g, "$1").replace(/[,\s]+\./g, ".").trim(); }

function sourceRow(s) {
  const isDoc = s.kind === "document";
  const st = isDoc ? "your document" : s.status === "active" ? "in force" : "repealed";
  return `<li><button type="button" data-cite="${esc(s.label)}"><span class="lbl${isDoc ? " doc" : ""}">${esc(s.label)}</span>
    <span class="ttl">${esc(s.title)}</span><span class="st">${st}</span></button></li>`;
}
function howTable(r) {
  const rows = r.steps.map((s) => `<tr><td>${esc(s.step)}</td><td>${s.secs != null ? Number(s.secs).toFixed(1) + " s" : ""}</td><td>${esc(s.info).slice(0, 220)}</td></tr>`).join("");
  const m = r.metrics || {};
  return `<details class="how"><summary>How this answer was reached</summary>
    <p>${ENGINE[r.engine]} engine, ${m.llm_calls ?? 0} language-model call${m.llm_calls === 1 ? "" : "s"}${r.engine === "jev_okf" ? `, ${m.decision_calls ?? 0} decisions` : ""}, ${(m.llm_tokens ?? 0).toLocaleString()} tokens.</p>
    <table>${rows}</table></details>`;
}
function renderResult(li, r) {
  const law = r.sources.filter((s) => s.kind !== "document").length;
  li.querySelector(".slot").innerHTML = `<article class="answer">
      ${renderAnswer(r.answer)}
      ${r.sources.length ? `<ul class="sources" aria-label="Sources">${r.sources.map(sourceRow).join("")}</ul>` : ""}
      <div class="meta"><span>${esc(ROUTE[r.route] || r.route)}. Answered in ${Math.round(r.seconds)} seconds${law ? `, citing ${law} section${law > 1 ? "s" : ""} of law` : ""}.</span></div>
      ${r.issues.length ? `<p class="meta">The self-check questioned ${r.issues.length} statement${r.issues.length > 1 ? "s" : ""}; read the cited sources before relying on ${r.issues.length > 1 ? "them" : "it"}.</p>` : ""}
      ${r.steps.length ? howTable(r) : ""}
    </article>`;
  li._result = r;
}

/* ---------- asking ---------- */
async function ask(question, extra = {}) {
  if (state.busy) return;
  if (state.docs.some((d) => d.loading)) { toast("Wait until the document has been read."); return; }
  state.busy = true; $("#send").disabled = true; $("#welcome").hidden = true;
  const docs = extra.docs || state.docs.map((d) => d.id);
  const names = state.docs.filter((d) => docs.includes(d.id)).map((d) => d.name);
  const li = document.createElement("li");
  li.innerHTML = `<p class="turn-q">${esc(question)}${names.length ? `<small>About ${esc(names.join(", "))}</small>` : ""}</p>
    <div class="slot"><div class="working"><span class="pulse"></span><span class="wtxt">Reading the law. On this laptop an answer takes about one to two minutes.</span></div></div>`;
  $("#thread").append(li); li.scrollIntoView({ behavior: "smooth", block: "start" });
  const t0 = Date.now(); const tick = setInterval(() => { const w = li.querySelector(".wtxt"); if (w) w.textContent = `Reading the law. ${Math.round((Date.now() - t0) / 1000)} seconds so far.`; }, 1000);
  try {
    const r = await api("/api/ask", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question, docs, engine: state.engine, review: !!extra.review }) });
    renderResult(li, r);
  } catch (e) {
    li.querySelector(".slot").innerHTML = `<div class="error">${esc(e.message)}</div>`;
  } finally { clearInterval(tick); state.busy = false; $("#send").disabled = false; }
}
$("#composer").addEventListener("submit", (e) => { e.preventDefault(); const q = $("#q").value.trim(); if (!q) return; $("#q").value = ""; grow(); ask(q); });
const grow = () => { const t = $("#q"); t.style.height = "auto"; t.style.height = Math.min(t.scrollHeight, 160) + "px"; };
$("#q").addEventListener("input", grow);
$("#q").addEventListener("keydown", (e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); $("#composer").requestSubmit(); } });
$("#starters").innerHTML = STARTERS.map((q) => `<button type="button" class="starter">${esc(q)}</button>`).join("");
$("#starters").addEventListener("click", (e) => { const b = e.target.closest(".starter"); if (b) ask(b.textContent); });

/* ---------- source drawer ---------- */
let lit = [];
function openSource(btn) {
  const li = btn.closest("li.turn") || btn.closest("#thread > li"); const r = li && li._result; if (!r) return;
  const label = btn.dataset.cite; const s = r.sources.find((x) => x.label === label);
  if (!s) { toast(`${label} isn't among the sources given to the engine.`); return; }
  lit.forEach((el) => { el.classList.remove("lit"); el.setAttribute?.("aria-pressed", "false"); }); lit = [];
  const claim = btn.classList.contains("cite") ? btn.closest(".claim") : null;
  if (claim) { claim.classList.add("lit"); lit.push(claim); }
  li.querySelectorAll(`.cite[data-cite="${label}"]`).forEach((c) => { c.setAttribute("aria-pressed", "true"); lit.push(c); });
  const isDoc = s.kind === "document";
  $("#drawer-kind").innerHTML = isDoc ? `${esc(label)}, from your document`
    : `${esc(label)}, statute <span class="badge ${s.status === "active" ? "inforce" : "repealed"}">${s.status === "active" ? "in force" : "repealed"}</span>`;
  $("#drawer-title").textContent = s.title;
  $("#drawer-body").innerHTML = (claim ? `<p class="for-claim"><span>Cited for</span>${esc(plain(claim))}</p>` : "")
    + `<p class="lawtext">${esc(s.text)}</p>`;
  $("#drawer").hidden = false; document.body.classList.add("drawer-open"); $("#drawer-close").focus();
}
function closeDrawer() { $("#drawer").hidden = true; document.body.classList.remove("drawer-open");
  lit.forEach((el) => { el.classList.remove("lit"); el.setAttribute?.("aria-pressed", "false"); }); lit = []; }
$("#thread").addEventListener("click", (e) => { const b = e.target.closest("[data-cite]"); if (b) openSource(b); });
$("#drawer-close").addEventListener("click", closeDrawer);
document.addEventListener("keydown", (e) => { if (e.key === "Escape" && !$("#drawer").hidden) closeDrawer(); });

/* ---------- settings and status ---------- */
const dlg = $("#settings");
document.querySelectorAll('input[name="engine"]').forEach((i) => { i.checked = i.value === state.engine;
  i.addEventListener("change", () => { state.engine = i.value; try { localStorage.setItem("engine", i.value); } catch (e) {} }); });
$("#settings-btn").addEventListener("click", () => { dlg.showModal(); checkStatus(); });
dlg.addEventListener("close", () => { if (dlg.returnValue === "clear") { $("#thread").innerHTML = ""; $("#welcome").hidden = false; closeDrawer(); } });
async function checkStatus() {
  try {
    const s = await api("/api/status");
    const w = $("#sys-writer"), d = $("#sys-dec");
    w.className = "sys " + (s.writer.ok ? "ok" : "bad");
    w.innerHTML = s.writer.ok ? `Language model ready: ${esc(s.writer.info)}` : "Language model not reachable. Start Ollama, then try again.";
    d.className = "sys " + (s.decisions.ok ? "ok" : "bad");
    d.innerHTML = s.decisions.ok ? `Decision model ready: ${esc(s.decisions.info)}` : `Decision model not running. Start it with <code>${esc(s.decisions.info)}</code>, or choose Agentic RAG.`;
    const all = s.writer.ok && (s.decisions.ok || state.engine === "rag");
    $("#status-dot").className = "status-dot " + (all ? "ok" : "bad");
    $("#settings-btn").setAttribute("aria-label", all ? "Settings. Models ready." : "Settings. A model is not running.");
  } catch (e) { $("#status-dot").className = "status-dot bad"; }
}
checkStatus();
