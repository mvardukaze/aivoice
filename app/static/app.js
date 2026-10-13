const $ = (s) => document.querySelector(s);
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
const api = async (url, opts) => {
  const r = await fetch(url, opts);
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || r.statusText);
  return body;
};

const SLIDERS = [
  { key: "length_scale", label: "ტემპი (სიგრძე)", step: 0.01 },
  { key: "noise_scale", label: "ინტონაცია", step: 0.01 },
  { key: "noise_w_scale", label: "რიტმის ცოცხლობა", step: 0.01 },
  { key: "volume", label: "ხმამაღლობა", step: 0.01 },
];
const PHRASES = [
  "შენ აქ რას აკეთებ? ხომ გითხარი, არ მოსულიყავი!",
  "ბოდიში... არ მინდოდა, რომ ასე მომხდარიყო.",
  "ჩვენ მოვიგეთ! წარმოგიდგენია? მოვიგეთ!",
  "ჩუმად, ვიღაც მოდის. არ გაინძრე.",
  "ოდესღაც, ძალიან დიდი ხნის წინ, ერთ პატარა სოფელში ცხოვრობდა მოხუცი მეთევზე.",
  "13.10.2026-დან iPhone 15 ღირს 2 499.99 ₾ — 15%-ით იაფად, ვიდრე მე-3 კვარტალში.",
];
const STATUS = { ready: "მზადაა", testing: "სატესტო", hidden: "დამალული", training: "იწვრთნება", selected: "არჩეულია", planned: "იქმნება" };

let emotions = { presets: {}, limits: {} };
let voices = [];
let current = "neutral";

// ---------- tabs ----------
document.querySelectorAll(".tabs button").forEach((b) =>
  b.addEventListener("click", () => {
    document.querySelectorAll(".tabs button").forEach((x) => (x === b ? x.setAttribute("aria-current", "page") : x.removeAttribute("aria-current")));
    document.querySelectorAll(".tab").forEach((t) => (t.hidden = t.id !== `tab-${b.dataset.tab}`));
    try { localStorage.setItem("tab", b.dataset.tab); } catch {}
  })
);
try {
  const saved = localStorage.getItem("tab");
  if (saved) document.querySelector(`.tabs button[data-tab="${saved}"]`)?.click();
} catch {}

// ---------- overview ----------
async function loadRoadmap() {
  const rm = await api("/api/roadmap");
  $("#roadmap").innerHTML = rm.stages.map((s, i) => `
    <li class="${s.status}">
      <span class="dot">${s.status === "done" ? "✓" : i + 1}</span>
      <div><div class="title">${esc(s.title)}</div><div class="note">${esc(s.note)}</div></div>
    </li>`).join("");
  $("#roadmap-updated").textContent = rm.updated ? `განახლდა: ${rm.updated}` : "";
  return rm;
}

const RUN_STATUS = { training: "იწვრთნება", exporting: "ექსპორტი", done: "დასრულდა", failed: "შეცდომა",
                     paused: "შეჩერებულია", partial: "შემდეგ სესიას ელის", archived: "არქივი" };
const DEVICE = { cpu: "CPU (ეს კომპიუტერი)", gpu: "GPU (ეს კომპიუტერი)", kaggle: "Kaggle GPU", gcp: "Google Cloud · L4" };
// ეპოქის ნომერი ეკრანზე: Kaggle-ის status-ში — ბოლო დასრულებული; ლოკალურ/Google Cloud-ზე — მიმდინარე (0-დან)
const epochShown = (r) => (r.epoch != null ? r.epoch + 1 : 0);
const hm = (h) => (h == null ? "—" : h >= 24 ? `${Math.floor(h / 24)} დღე ${Math.round(h % 24)} სთ` : `${h} სთ`);
const etaClock = (h) => {
  if (h == null) return "";
  const d = new Date(Date.now() + h * 3600e3);
  return d.toLocaleString("ka-GE", { weekday: "short", hour: "2-digit", minute: "2-digit" });
};

function renderStats(ov, base, tj) {
  const ready = voices.filter((v) => v.status === "ready").length;
  const testing = voices.filter((v) => v.status === "testing").length;
  const run = base.run || {};
  const busy = (tj?.jobs || []).filter(jobActive).length;
  const items = [
    [`${ready}${testing ? ` + ${testing}` : ""}`, testing ? "მზა ხმა + სატესტო" : "მზა ხმა", "wave", "voices"],
    [busy ? `${busy}` : "—", "მიმდინარე სამუშაო", "activity", "train"],
    [run.epoch != null ? `${epochShown(run)} ეპოქა` : "—", "საბაზო მოდელი", "layers", "base"],
    run.device === "gcp" && run.status === "training"
      ? [`~${hm(run.eta_hours)}`, "დარჩა (Google Cloud L4)", "sparkle", "base"]
      : [`${(+ov.kaggle_week_hours).toFixed(1)} / ${ov.kaggle_week_limit} სთ`, "Kaggle GPU ბოლო 7 დღეში", "sparkle", "train"],
  ];
  $("#stats").innerHTML = items.map(([b, s, ic, page]) => `<button type="button" class="stat" data-stat-go="${page}">
    <span class="stat-ico">${UI.icon(ic, 18)}</span><span class="stat-body"><b>${esc(b)}</b><span>${esc(s)}</span></span></button>`).join("");
  $("#stats").querySelectorAll("[data-stat-go]").forEach((x) => x.addEventListener("click", () => UI.go(x.dataset.statGo)));
}

function progressPct(r) {
  const n = r.steps_per_epoch || 1;
  const e = r.epoch != null ? r.epoch + (r.device === "kaggle" ? 1 : 0) : 0;
  return Math.min(100, ((e * n + (r.step_in_epoch || 0)) / ((r.max_epochs || 1) * n)) * 100);
}

// აქტიური = რამე ეშვება (მომზადება, ატვირთვა, წვრთნა, ლოდინი) ან ყურადღებას ითხოვს (შეცდომა, სესია დასრულდა)
const jobActive = (j) => j.busy || ["prepare", "uploading", "training", "partial", "failed"].includes(j.stage);
const STAGE_KA = { new: "შექმნილია", prepare: "მზადდება", prepared: "მზადაა გასაშვებად", uploading: "იტვირთება", training: "იწვრთნება",
  partial: "სესია დასრულდა", done: "დასრულდა", failed: "შეცდომა", stopped: "გაჩერებულია" };
const TARGET_KA = { kaggle: "Kaggle", local: "ეს კომპიუტერი", gcp: "Google Cloud" };

function renderJobs(tj) {
  const jobs = (tj?.jobs || []).filter(jobActive);
  const running = jobs.filter((j) => !["failed", "partial"].includes(j.stage)).length, failed = jobs.filter((j) => j.stage === "failed").length;
  UI.setStatus("jobs", jobs.length ? { text: running ? `${running} მიმდინარე` : `${jobs.length} სამუშაო`, kind: failed ? "bad" : "busy",
    title: `აქტიური სამუშაოები: ${jobs.length}${failed ? ` (შეცდომა: ${failed})` : ""}`, go: ["train"] } : null);
  $("#jobs").innerHTML = jobs.length ? jobs.map((j) => `
    <div class="job">
      <div class="run-head">
        <div><b>${esc(j.name)}</b> <span class="muted small">${j.kind === "base" ? "საბაზო მოდელი" : esc(j.slot)}${j.target ? " · " + esc(TARGET_KA[j.target] || j.target) : ""}</span></div>
        <span class="badge ${j.stage === "failed" ? "bad" : "training"}">${esc(STAGE_KA[j.stage] || j.stage || "")}${j.busy ? " …" : ""}</span>
      </div>
      <div class="bar"><span style="width:${j.max_epochs ? Math.min(100, (j.epoch / j.max_epochs) * 100).toFixed(1) : 0}%"></span></div>
      <div class="muted small">ეპოქა ${j.epoch} / ${j.max_epochs ?? "—"}${j.note ? " · " + esc(j.note) : ""}${j.error ? ` · <span class="err">${esc(j.error)}</span>` : ""}</div>
    </div>`).join("") + `<div class="actions"><button class="ghost small" data-goto-tab="train">დეტალები — „წვრთნა“ →</button></div>`
    : `<div class="empty">${UI.icon("check", 20)}<div>ამჟამად არაფერი მიმდინარეობს.</div></div>`;
  $("#jobs").querySelector("[data-goto-tab]")?.addEventListener("click", () => document.querySelector('.tabs button[data-tab="train"]').click());
}

function renderLibrarySummary() {
  const shown = voices.filter((v) => v.status !== "hidden");
  $("#ov-voices").innerHTML = shown.length ? shown.map((v) => `
    <div class="lib-row"><span class="avatar sm" style="--h:${hue(v.id)}">${esc(initial(v.name))}</span>
      <div class="lib-main"><b>${esc(v.name)}</b><span class="muted small">${esc([v.gender, v.character].filter(Boolean).join(" · "))}</span></div>
      <span class="badge ${v.status}">${esc(STATUS[v.status] || v.status)}</span></div>`).join("")
    : `<div class="empty">ჯერ არცერთი ხმა — „პარამეტრები → მოდელები → ჩამოტვირთვა“.</div>`;
}

// ---------- voices ----------
// ემოციები ფარდობითია: თითო ხმას თავისი ნორმა აქვს (backend: /api/voices → defaults, emotions)
let showHidden = false, editing = null;
const curVoice = () => voices.find((v) => v.id === $("#voice-select").value);

const hue = (s) => [...String(s)].reduce((a, c) => (a * 31 + c.charCodeAt(0)) % 360, 7);
const initial = (s) => [...String(s || "?").trim()][0] || "?";

function voiceCard(v) {
  if (editing === v.id) return voiceEditCard(v);
  const d = v.defaults;
  const canListen = v.available && ["ready", "testing"].includes(v.status);
  return `
    <div class="voice ${v.status}" data-id="${esc(v.id)}">
      <div class="voice-top">
        <span class="avatar" style="--h:${hue(v.id)}" aria-hidden="true">${esc(initial(v.name))}</span>
        <div class="voice-title"><span class="name">${esc(v.name)}</span><span class="id">${esc(v.id)}</span></div>
        ${UI.dd([
          UI.ddItem("data-edit", "edit", "რედაქტირება"),
          v.available ? UI.ddItem("data-check", "shield", "შემოწმება") : "",
          !["hidden", "planned"].includes(v.status) ? UI.ddItem("data-hide", "eyeoff", "დამალვა") : "",
          `<div class="dd-sep" role="separator"></div>`,
          UI.ddItem("data-remove", "trash", "წაშლა", true),
        ])}
      </div>
      <div class="voice-tags"><span class="badge ${v.status}">${STATUS[v.status] || v.status}</span>
        ${[v.gender, v.age].filter(Boolean).map((x) => `<span class="tag">${esc(x)}</span>`).join("")}</div>
      ${v.character ? `<div class="meta">${esc(v.character)}</div>` : ""}
      <div class="meta small">${esc(v.source)}${v.size_mb ? ` · ${v.sample_rate} Hz, ${v.size_mb} MB` : ""}</div>
      ${d ? `<div class="norm-line small" title="ნორმა">ტემპი <b>${d.length_scale}</b> · ინტონაცია <b>${d.noise_scale}</b> · რიტმი <b>${d.noise_w_scale}</b> · ხმამაღლობა <b>${d.volume}</b></div>` : ""}
      ${v.model && !v.available ? `<div class="meta err">მოდელი ამ კომპიუტერზე არ არის — „პარამეტრები → მოდელები“</div>` : ""}
      ${canListen ? `<div class="voice-foot"><button class="ghost small" data-listen>${UI.icon("play", 12)} მოსმენა</button></div>` : ""}
    </div>`;
}

function voiceEditCard(v) {
  const opt = (val, label) => `<option value="${val}" ${v.status === val ? "selected" : ""}>${label}</option>`;
  return `
    <form class="voice editing" data-id="${esc(v.id)}">
      <div class="edit-head"><b>რედაქტირება</b><span class="id">${esc(v.id)}</span></div>
      <label class="lbl">სახელი <input name="name" value="${esc(v.name)}" required maxlength="60"></label>
      <label class="lbl">სქესი <input name="gender" value="${esc(v.gender)}" maxlength="20"></label>
      <label class="lbl">ასაკი <input name="age" value="${esc(v.age)}" maxlength="20" placeholder="მაგ. 25–40"></label>
      <label class="lbl">ხასიათი <input name="character" value="${esc(v.character)}" maxlength="200"></label>
      <label class="lbl">სტატუსი <select name="status">
        ${opt("ready", "მზადაა — ბიბლიოთეკაშია")}${opt("testing", "სატესტო — ჯერ მოწმდება")}${opt("hidden", "დამალული")}
        ${v.status === "planned" ? opt("planned", "იქმნება") : ""}</select></label>
      <div class="actions">
        <button class="primary small" type="submit">შენახვა</button>
        <button class="ghost small" type="button" data-cancel>გაუქმება</button>
        <button class="ghost small danger" type="button" data-delete>წაშლა</button>
      </div>
    </form>`;
}

function renderVoices() {
  const vis = voices.filter((v) => showHidden || v.status !== "hidden");
  $("#voice-list").innerHTML = vis.map(voiceCard).join("") || `<div class="empty">${UI.icon("wave", 22)}<div>ბიბლიოთეკა ცარიელია.</div></div>`;
  const nHidden = voices.filter((v) => v.status === "hidden").length;
  $("#hidden-toggle").hidden = !nHidden;
  $("#hidden-toggle").textContent = showHidden ? "დამალულების დამალვა" : `დამალულების ჩვენება (${nHidden})`;
  const sel = $("#voice-select"), prev = sel.value;
  const usable = voices.filter((v) => v.available && ["ready", "testing"].includes(v.status));
  sel.innerHTML = usable.map((v) => `<option value="${esc(v.id)}">${esc(v.name)}${v.status === "testing" ? " (სატესტო)" : ""}</option>`).join("");
  if (usable.some((v) => v.id === prev)) sel.value = prev;
  bindVoiceCards();
  if (Object.keys(emotions.presets).length && $("#sliders").innerHTML) applyPreset(current || "neutral");
}

function bindVoiceCards() {
  $("#voice-list").querySelectorAll("[data-edit]").forEach((b) => b.addEventListener("click", () => {
    editing = b.closest("[data-id]").dataset.id; renderVoices();
  }));
  $("#voice-list").querySelectorAll("[data-check]").forEach((b) => b.addEventListener("click", () => openCheck(b.closest("[data-id]").dataset.id)));
  $("#voice-list").querySelectorAll("[data-remove]").forEach((b) => b.addEventListener("click", () => deleteVoice(b.closest("[data-id]").dataset.id)));
  $("#voice-list").querySelectorAll("[data-hide]").forEach((b) => b.addEventListener("click", async () => {
    const id = b.closest("[data-id]").dataset.id;
    try { await postJSON(`/api/voices/${id}`, { status: "hidden" }); await reloadVoices(); UI.toast("ხმა დაიმალა", "ok"); }
    catch (e) { UI.toast(e.message, "error"); }
  }));
  $("#voice-list").querySelectorAll("[data-listen]").forEach((b) => b.addEventListener("click", () => {
    $("#voice-select").value = b.closest("[data-id]").dataset.id;
    $("#voice-select").dispatchEvent(new Event("change"));
    UI.setSub("voices", "listen");
    window.scrollTo({ top: 0, behavior: "smooth" });
  }));
  const f = $("#voice-list").querySelector("form.editing");
  if (!f) return;
  const id = f.dataset.id;
  f.querySelector("[data-cancel]").onclick = () => { editing = null; renderVoices(); };
  f.querySelector("[data-delete]").onclick = () => deleteVoice(id);
  f.onsubmit = async (e) => {
    e.preventDefault();
    try { await postJSON(`/api/voices/${id}`, Object.fromEntries(new FormData(f))); editing = null; await reloadVoices(); UI.toast("შენახულია", "ok"); }
    catch (err) { UI.toast(err.message, "error"); }
  };
}

async function deleteVoice(id) {
  if (!(await UI.confirm("ხმა ბიბლიოთეკიდან ამოვიღო? ფაილები გადავა მონაცემების საქაღალდის trash/-ში (საბოლოოდ არ იშლება)."))) return;
  try { await api(`/api/voices/${id}`, { method: "DELETE" }); editing = null; await reloadVoices(); }
  catch (e) { UI.toast(e.message, "error"); }
}

const postJSON = (url, data) => api(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });
async function reloadVoices() { voices = await api("/api/voices"); renderVoices(); renderLibrarySummary(); }
window.reloadVoices = reloadVoices;

// ---------- შემოწმება: ხმა ↔ ნარევი, ერთი და იგივე ფრაზები + MMS (tools/voice_check.py) ----------
let checkId = null, checkTimer = null;
const VERDICT = { good: ["კარგია — ნარევს არ ჩამოუვარდება", "good"], fair: ["საშუალო — მოუსმინეთ ყურადღებით", "maybe"], bad: ["ცუდი — ნარევზე ბევრად უარესია", "bad"] };
const cerCls = (c) => c == null ? "" : c <= 0.08 ? "good" : c <= 0.15 ? "maybe" : "bad";

async function openCheck(id) {
  checkId = id;
  $("#check-card").hidden = false;
  UI.setSub("voices", "library");
  await renderCheck();
  $("#check-card").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function renderCheck() {
  clearTimeout(checkTimer);
  const id = checkId, v = voices.find((x) => x.id === id);
  if (!id || !v) { $("#check-card").hidden = true; return; }
  const c = await api(`/api/voices/${id}/check`);
  const r = c.report;
  const au = (f) => `<audio controls preload="none" src="/api/voices/${id}/check/${f}"></audio>`;
  const cer = (x) => x?.cer != null ? `<span class="verdict ${cerCls(x.cer)}" title="MMS-მა გაიგონა: ${esc(x.heard)}">${Math.round(x.cer * 100)}%</span>` : "";
  const fresh = r && !c.busy && r.stage === "done";
  $("#check-card").innerHTML = `
    <div class="card-head"><h2>შემოწმება: ${esc(v.name)}</h2>
      ${r?.verdict ? `<span class="verdict ${VERDICT[r.verdict][1]}">${VERDICT[r.verdict][0]}</span>` : ""}</div>
    <p class="muted small">ხმა და მისი ნარევი${r?.design ? ` („${esc(r.design)}“)` : ""} ერთსა და იმავე ფრაზებს კითხულობენ. პროცენტი — ავტომატური მსმენელის (MMS)
      შეცდომა: რაც ნაკლებია, მით სუფთად ისმის (შიშინი, კანკალი, დაკარგული ბგერები). „ღ“-ს MMS ვერ აფასებს — ყურით მოუსმინეთ.
      მაუსი პროცენტზე — რა გაიგონა.</p>
    ${r ? `<div class="check-sum">${r.summary ? `საშუალო: ხმა <b>${Math.round(r.summary.voice_cer * 100)}%</b>${r.summary.mix_cer != null ? ` · ნარევი <b>${Math.round(r.summary.mix_cer * 100)}%</b>` : ""}` : ""}
        ${r.started ? `<span class="muted small"> · ${esc(r.started.replace("T", " "))}</span>` : ""}</div>
      <table class="check-table"><thead><tr><th>ფრაზა</th><th>ხმა</th>${r.design ? "<th>ნარევი</th>" : ""}</tr></thead><tbody>
        ${r.rows.map((row) => `<tr><td>${esc(row.text)}</td><td>${au(row.voice.file)} ${cer(row.voice)}</td>${r.design ? `<td>${row.mix ? au(row.mix.file) + " " + cer(row.mix) : ""}</td>` : ""}</tr>`).join("")}
      </tbody></table>` : `<div class="empty">${UI.icon("shield", 22)}<div>ჯერ არ შემოწმებულა.</div></div>`}
    ${c.busy ? `<pre class="tr-log">${esc(c.log.split("\n").slice(-6).join("\n"))}\n… მიმდინარეობს</pre>` : ""}
    <div class="actions">
      <button class="primary small" id="chk-run" ${c.busy ? "disabled" : ""}>${r ? "თავიდან შემოწმება" : "შემოწმების დაწყება"}</button>
      ${fresh && v.status !== "ready" ? `<button class="primary small" id="chk-ready">✓ მზადაა — ბიბლიოთეკაში</button>` : ""}
      ${fresh && v.status !== "hidden" ? `<button class="ghost small" id="chk-hide">დამალვა</button>` : ""}
      <button class="ghost small" id="chk-close">დახურვა</button>
      <span class="muted small">სტატუსი: ${STATUS[v.status] || v.status}</span>
    </div>`;
  $("#chk-run").onclick = async () => { try { await api(`/api/voices/${id}/check`, { method: "POST" }); renderCheck(); } catch (e) { UI.toast(e.message, "error"); } };
  $("#chk-close").onclick = () => { checkId = null; $("#check-card").hidden = true; };
  const setStatus = async (status) => { await postJSON(`/api/voices/${id}`, { status }); await reloadVoices(); renderCheck(); };
  $("#chk-ready")?.addEventListener("click", () => setStatus("ready"));
  $("#chk-hide")?.addEventListener("click", () => setStatus("hidden"));
  if (c.busy) checkTimer = setTimeout(() => { if (checkId === id) renderCheck(); }, 4000);
}

function renderEmotions() {
  $("#emotions").innerHTML = Object.entries(emotions.presets).map(([k, p]) =>
    `<button class="chip" data-emo="${k}" aria-pressed="${k === current}">${esc(p.label)}</button>`).join("");
  $("#emotions").querySelectorAll(".chip").forEach((c) => c.addEventListener("click", () => applyPreset(c.dataset.emo)));

  $("#sliders").innerHTML = SLIDERS.map((s) => {
    const [lo, hi] = emotions.limits[s.key];
    return `<label class="slider"><span>${s.label}</span>
      <input type="range" id="sl-${s.key}" min="${lo}" max="${hi}" step="${s.step}">
      <output id="out-${s.key}"></output></label>`;
  }).join("");
  SLIDERS.forEach((s) => $(`#sl-${s.key}`).addEventListener("input", (e) => {
    $(`#out-${s.key}`).textContent = (+e.target.value).toFixed(2);
    current = null;
    $("#emotions").querySelectorAll(".chip").forEach((c) => c.setAttribute("aria-pressed", "false"));
  }));
  $("#voice-select").addEventListener("change", () => applyPreset(current || "neutral"));
  applyPreset(current);
}

// ხმის საკუთარი ემოციები (ნორმიდან); ხმა თუ არ არის — ზოგადი პრესეტი
function presetFor(key) {
  return curVoice()?.emotions?.[key] || emotions.presets[key];
}

function applyPreset(key) {
  current = key;
  const p = presetFor(key);
  SLIDERS.forEach((s) => { $(`#sl-${s.key}`).value = p[s.key]; $(`#out-${s.key}`).textContent = (+p[s.key]).toFixed(2); });
  $("#emotions").querySelectorAll(".chip").forEach((c) => c.setAttribute("aria-pressed", c.dataset.emo === key));
}

function sliderParams() {
  return Object.fromEntries(SLIDERS.map((s) => [s.key, +$(`#sl-${s.key}`).value]));
}

async function synth(params) {
  return api("/api/synthesize", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ voice: $("#voice-select").value, text: $("#text").value, ...params }),
  });
}

$("#save-default").addEventListener("click", async () => {
  const v = curVoice();
  if (!v) return;
  if (!(await UI.confirm(`ეს მნიშვნელობები გახდეს „${v.name}“-ის ნორმა? (ყველა ემოცია მასზე გადაითვლება)`, { okText: "შენახვა" }))) return;
  try {
    await postJSON(`/api/voices/${v.id}`, { defaults: sliderParams() });
    await reloadVoices(); applyPreset("neutral");
    $("#speak-info").textContent = `✓ „${v.name}“-ის ნორმა შენახულია`;
  } catch (e) { $("#speak-info").textContent = "შეცდომა: " + e.message; }
});
$("#reset-default").addEventListener("click", async () => {
  const v = curVoice();
  if (!v || !(await UI.confirm(`„${v.name}“-ის ნორმა დავაბრუნო წვრთნისდროინდელზე?`, { okText: "დაბრუნება" }))) return;
  try { await api(`/api/voices/${v.id}/reset`, { method: "POST" }); await reloadVoices(); applyPreset("neutral"); }
  catch (e) { $("#speak-info").textContent = "შეცდომა: " + e.message; }
});
$("#hidden-toggle").addEventListener("click", () => { showHidden = !showHidden; renderVoices(); });

// ---------- როგორ წაიკითხავს (ტექსტის გასწორება: app/textnorm.py) ----------
let normTimer = null;
async function updateNorm() {
  const text = $("#text").value;
  const r = await postJSON("/api/textnorm", { text }).catch(() => null);
  if (!r) return;
  const same = r.text.replace(/-ღ/g, "ღ") === text.trim();
  $("#norm-preview").hidden = same;
  $("#norm-text").textContent = r.text;
}
$("#text").addEventListener("input", () => { clearTimeout(normTimer); normTimer = setTimeout(updateNorm, 400); });

// ---------- ლექსიკონი ----------
async function renderLexicon() {
  const d = await api("/api/lexicon");
  const q = $("#lex-filter").value.trim().toLowerCase();
  const rows = d.entries.filter((e) => !q || e.word.toLowerCase().includes(q) || e.say.includes(q));
  $("#lex-list").innerHTML = rows.map((e) => `
    <div class="lex-row"><span class="mono">${esc(e.word)}</span><span>→ ${esc(e.say)}</span>
      <span class="muted small">${e.local ? `<span class="tag">დამატებული</span>` : ""}</span>
      <button class="ghost small" data-say="${esc(e.say)}" title="მოსმენა" aria-label="მოსმენა">${UI.icon("play", 12)}</button>
      <button class="ghost small" data-del="${esc(e.word)}" title="${e.local ? "წაშლა" : "გაუქმება"}" aria-label="წაშლა">${UI.icon("x", 13)}</button></div>`).join("")
    || `<div class="empty">ვერ მოიძებნა.</div>`;
  $("#lex-list").querySelectorAll("[data-say]").forEach((b) => b.addEventListener("click", () => sayWord(b.dataset.say)));
  $("#lex-list").querySelectorAll("[data-del]").forEach((b) => b.addEventListener("click", async () => {
    await api(`/api/lexicon/${encodeURIComponent(b.dataset.del)}`, { method: "DELETE" }); renderLexicon(); updateNorm();
  }));
}
const lexAudio = new Audio();
async function sayWord(text) {
  try {
    const r = await api("/api/synthesize", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ voice: $("#voice-select").value, text }) });
    lexAudio.src = r.url; lexAudio.play();
  } catch (e) { UI.toast(e.message, "error"); }
}
$("#lex-box").addEventListener("toggle", (e) => { if (e.target.open) renderLexicon(); });
// „ლექსიკონი“ ქვე-ჩანართია — გახსნისას ყოველთვის ღიაა
document.addEventListener("subtab", (e) => {
  if (e.detail.page !== "voices" || e.detail.key !== "lexicon") return;
  if ($("#lex-box").open) renderLexicon(); else $("#lex-box").open = true;
});
$("#lex-filter").addEventListener("input", renderLexicon);
$("#lex-try").addEventListener("click", () => { const v = $("#lex-form").say.value.trim(); if (v) sayWord(v); });
$("#lex-form").onsubmit = async (e) => {
  e.preventDefault();
  try { await postJSON("/api/lexicon", Object.fromEntries(new FormData(e.target))); e.target.reset(); renderLexicon(); updateNorm(); UI.toast("დაემატა", "ok"); }
  catch (err) { UI.toast(err.message, "error"); }
};

$("#phrases").innerHTML = PHRASES.map((p, i) => `<button class="chip" data-i="${i}">${esc(p.slice(0, 28))}…</button>`).join("");
$("#phrases").querySelectorAll(".chip").forEach((c) => c.addEventListener("click", () => { $("#text").value = PHRASES[c.dataset.i]; updateNorm(); }));

$("#speak").addEventListener("click", async () => {
  const btn = $("#speak");
  btn.disabled = true;
  $("#speak-info").textContent = "გენერირდება…";
  try {
    const t0 = performance.now();
    const r = await synth(sliderParams());
    const ms = Math.round(performance.now() - t0);
    const player = $("#player");
    player.hidden = false;
    player.src = r.url;
    player.play();
    $("#speak-info").textContent = `${r.seconds} წმ აუდიო · ${ms} მწმ`;
  } catch (e) {
    $("#speak-info").textContent = "შეცდომა: " + e.message;
  } finally {
    btn.disabled = false;
  }
});

$("#compare").addEventListener("click", async () => {
  const list = $("#compare-list");
  list.innerHTML = `<p class="muted small">გენერირდება ყველა ემოცია…</p>`;
  try {
    const rows = [];
    for (const [k, p] of Object.entries(emotions.presets)) {
      const label = p.label;
      const r = await synth({ emotion: k });
      rows.push(`<div class="compare-row"><span>${esc(label)}</span><audio controls preload="none" src="${r.url}"></audio></div>`);
    }
    list.innerHTML = rows.join("");
  } catch (e) {
    list.innerHTML = `<p class="muted small">შეცდომა: ${esc(e.message)}</p>`;
  }
});

const sampleAudio = new Audio();
function bindPlays(root) {
  root.querySelectorAll(".play").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll(".play.on").forEach((x) => x.classList.remove("on"));
    b.classList.add("on");
    sampleAudio.src = b.dataset.src;
    sampleAudio.play();
  }));
}

// ---------- საბაზო მოდელი ----------
function whereTraining(r, job) {
  if (r?.device === "gcp") return "Google Cloud · 1× NVIDIA L4 · batch 32";
  if (r?.local?.running) return "ეს კომპიუტერი (CPU) · batch 8";
  return `Kaggle · ${job.devices || 1}× T4 · batch ${job.batch || "—"}`;
}

// Google Cloud (tools/gcp_train.py): spot მანქანა, შეწყვეტის შემდეგ თავისით ირთვება
function gcpLine(r) {
  const g = r.gcp;
  if (!g) return "";
  const pct = g.batch != null ? Math.round((Math.min(g.batch, g.batches_per_epoch) / g.batches_per_epoch) * 100) : null;
  return `<p class="small"><span class="badge training">Google Cloud L4</span>
    <span class="muted"> · მიმდინარე ეპოქა ${pct != null ? pct + "%" : "—"} · ~${Math.round((g.batches_per_epoch * g.sec_per_batch) / 60)} წთ ეპოქაზე
    · მუშაობს ${hm(g.hours)}${g.restarts ? ` · spot-ის შეწყვეტის შემდეგ ჩაირთო ${g.restarts}-ჯერ` : ""}
    ${g.fetched_at ? ` · ხმის შემქმნელი განახლდა ${esc(g.fetched_at.slice(5, 16).replace("T", " "))}` : ""}</span>
    ${g.error ? ` · <span class="err">${esc(g.error)}</span>` : ""}</p>
    <p class="muted small">დასრულებისას მოდელი ავტომატურად ჩამოიტვირთება და მანქანა წაიშლება. კრედიტი: Google Cloud Console → Billing.</p>`;
}

// ლოკალური (CPU) წვრთნა Kaggle-ის სესიებს შორის — tools/local_train.py
function localLine(l) {
  if (!l) return "";
  if (l.running) {
    const pct = l.steps_per_epoch ? Math.round((l.step_in_epoch / l.steps_per_epoch) * 100) : 0;
    return `<p class="small"><span class="badge training">ლოკალური წვრთნა (CPU)</span>
      <span class="muted"> · ეპოქა ${l.epoch != null ? l.epoch + 1 : "—"}, ${pct}% · ~${l.epoch_hours ?? "?"} სთ ეპოქაზე
      · დაიწყო ${esc((l.started || "").slice(5, 16).replace("T", " "))} · checkpoint ყოველ 30 წთ-ში</span>
      ${l.error ? ` · <span class="err">${esc(l.error)}</span>` : ""}</p>`;
  }
  return l.stopped || l.error ? `<p class="small muted">ლოკალური წვრთნა: ${esc(l.note || "გაჩერებულია")}
    ${l.stopped ? ` (${esc(l.stopped.slice(5, 16).replace("T", " "))})` : ""}${l.error ? ` · <span class="err">${esc(l.error)}</span>` : ""}</p>` : "";
}

function renderBase(b) {
  const r = b.run;
  const ds = b.dataset || {};
  const job = b.job || {};
  const spk = b.preview_speakers || {};
  const maxN = Math.max(1, ...(ds.f0_hist || []).map((h) => h.n));
  const sessions = (r?.sessions || []).slice().reverse();
  const spkLabel = (id) => {
    const f0 = spk[id]?.f0_median;
    return f0 ? `${f0 < 165 ? "დაბალი" : "მაღალი"} ხმა · ${Math.round(f0)} Hz` : id;
  };
  $("#base").innerHTML = `
    <div class="card" data-pane="progress">
      <h2>ქართული საბაზო მოდელი</h2>
      <p class="muted">${esc(b.note || "")}. ქართული გამოთქმა ისწავლა ბევრ ხმაზე — მისგან იქმნება ხმის შემქმნელის ნარევები და
        ცალკე ხმები (მაგ. „ქალი 1“). ლიცენზიისგან თავისუფალია.</p>
      <div class="kv">
        <div><span>საწყისი წერტილი</span><b>LJSpeech (ინგლისური, public domain)</b></div>
        <div><span>დინამიკი</span><b>${ds.speakers ?? "—"}</b></div>
        <div><span>მონაცემი</span><b>${ds.hours ?? "—"} სთ · ${(ds.clips || 0).toLocaleString()} კლიპი</b></div>
        <div><span>სად იწვრთნება</span><b>${whereTraining(r, job)}</b></div>
        <div><span>ეპოქა</span><b>${r?.epoch != null ? r.epoch + 1 : 0} / ${job.max_epochs || "—"}</b></div>
        ${r?.device === "gcp" && r.eta_hours != null && r.status === "training" ? `<div><span>დასრულება</span><b>~${hm(r.eta_hours)} · ${esc(etaClock(r.eta_hours))}</b></div>` : ""}
        <div><span>Kaggle-ზე სულ</span><b>${r?.kaggle_hours ?? 0} სთ</b></div>
        ${r?.status === "done" && r.updated ? `<div><span>დასრულდა</span><b>${esc(r.updated.slice(0, 16).replace("T", " "))}</b></div>` : ""}
      </div>
      ${r ? `
        <div class="bar big"><span style="width:${progressPct(r).toFixed(1)}%"></span></div>
        <p class="small"><span class="badge ${r.status === "failed" ? "bad" : r.status === "done" ? "ready" : "training"}">${RUN_STATUS[r.status] || r.status}</span>
          ${r.kaggle_state && r.status === "training" && r.device !== "gcp" ? `<span class="muted"> · ${esc(r.kaggle_state.replace(/.*status /, "").replace(/"/g, ""))}</span>` : ""}
          ${r.session_started && r.device !== "gcp" ? `<span class="muted"> · მიმდინარე სესია დაიწყო ${esc(r.session_started.slice(5, 16).replace("T", " "))}</span>` : ""}
          ${r.error ? ` · <span class="err">${esc(r.error)}</span>` : ""}</p>
        ${r.status === "done" ? "" : gcpLine(r)}
        ${r.device === "gcp" || r.status === "done" ? "" : localLine(r.local)}`
        : `<div class="empty">წვრთნა ჯერ არ დაწყებულა — dataset Kaggle-ზე იტვირთება.</div>`}
    </div>

    <div class="card" data-pane="samples">
      <h2>მოსასმენი ნიმუშები</h2>
      <p class="muted small help">ერთი და იგივე ფრაზები ოთხი ხმით, ყოველ თაობაზე — ძველი მარცხნივ, ახალი მარჯვნივ.</p>
      ${UI.hint(`წვრთნის გაგრძელებისას ახალი სვეტი ემატება. „ყველა“ ერთ ხმას ყველა თაობით ზედიზედ უკრავს; სვეტის თავში ✕ — თაობის წაშლა.`)}
      <div id="prev-matrix"></div>
    </div>

    <div class="grid2" data-pane="stats">
      <div class="card">
        <h2>ხმის სიმაღლის განაწილება</h2>
        <p class="muted small">${ds.low ?? 0} დაბალი (კაცის ტიპის) · ${ds.high ?? 0} მაღალი (ქალის ტიპის) ხმა · დინამიკზე მედიანა ${ds.minutes_median ?? "—"} წთ</p>
        <div class="hist">${(ds.f0_hist || []).map((h) => `
          <div class="hcol" title="${h.from}–${h.to} Hz: ${h.n} დინამიკი"><span style="height:${(h.n / maxN * 100).toFixed(0)}%"></span><i>${h.from}</i></div>`).join("")}</div>
        <p class="muted small">Hz — ხმის ძირითადი სიხშირე. ~85–155 Hz — კაცის, ~165–255 Hz — ქალის ხმა.</p>
      </div>
      <div class="card">
        <h2>სესიები</h2>
        ${sessions.length ? `<div class="table-wrap"><table><thead><tr><th>დასრულდა</th><th class="num">საათი</th><th class="num">ეპოქა</th><th>შედეგი</th></tr></thead>
          <tbody>${sessions.filter((x) => x.hours).map((x) => `<tr><td>${esc((x.ended || "").slice(5, 16).replace("T", " "))}</td><td class="num">${x.hours ?? "—"}</td>
            <td class="num">${x.epoch != null ? x.epoch + 1 : "—"}</td><td>${esc(RUN_STATUS[x.status] || x.status || "")}</td></tr>`).join("")}</tbody></table></div>`
          : `<div class="empty">დასრულებული სესია ჯერ არ არის.</div>`}
        <p class="muted small">სესიები Kaggle-ზე; ბოლო ეტაპი (ეპოქა 27–72) — Google Cloud L4.</p>
      </div>
    </div>`;
  bindPlays($("#base"));
  renderPreviewMatrix(r?.previews || [], spkLabel);
}

// ნიმუშების შედარება: სვეტები — თაობები (ძველი → ახალი), რიგები — ხმები, ზემოთ — ფრაზის არჩევა
let prevPhrase = 1;
const phraseNo = (f) => +(f.file.match(/_(\d+)\.wav$/) || f.file.match(/(\d+)\.wav$/) || [0, 0])[1];
function renderPreviewMatrix(gens, spkLabel) {
  const box = $("#prev-matrix");
  if (!box) return;
  gens = gens.filter((g) => g.files?.length);
  if (!gens.length) { box.innerHTML = `<div class="empty">ნიმუშები ჯერ არ არის.</div>`; return; }
  const phrases = {}, speakers = [];
  gens.forEach((g) => g.files.forEach((f) => {
    phrases[phraseNo(f)] ||= f.text;
    if (f.speaker && !speakers.includes(f.speaker)) speakers.push(f.speaker);
  }));
  if (!phrases[prevPhrase]) prevPhrase = +Object.keys(phrases)[0];
  const cell = (g, sp) => g.files.find((f) => f.speaker === sp && phraseNo(f) === prevPhrase);
  const src = (f) => `/api/preview/base_ka/${f.file}`;
  box.innerHTML = `
    <div class="chips">${Object.entries(phrases).map(([n, t]) =>
      `<button class="chip ${+n === prevPhrase ? "on" : ""}" data-phr="${n}" title="${esc(t)}">${n}. ${esc(t.slice(0, 24))}…</button>`).join("")}</div>
    <p class="phrase-text">„${esc(phrases[prevPhrase])}“</p>
    <div class="cmp-wrap"><table class="cmp">
      <thead><tr><th>ხმა</th>${gens.map((g) => `<th><b>${esc(g.label || "")}</b>
        <span>ეპოქა ${g.epoch != null ? g.epoch + 1 : "—"} · ${esc((g.time || "").slice(5, 10))}</span>
        <button class="ghost small gen-del" data-gen="${esc(g.files[0].file.split("/")[0])}" data-label="${esc(g.label || "")}" title="ამ თაობის წაშლა" aria-label="ამ თაობის წაშლა">${UI.icon("x", 13)}</button></th>`).join("")}</tr></thead>
      <tbody>${speakers.map((sp) => `<tr>
        <td><span class="small">${esc(spkLabel(sp))}</span>
          <button class="ghost small seq" data-seq="${esc(sp)}" title="ყველა თაობა ზედიზედ">${UI.icon("play", 12)} ყველა</button></td>
        ${gens.map((g) => { const f = cell(g, sp);
          return `<td>${f ? `<button class="play" data-src="${src(f)}" title="${esc(g.label || "")}" aria-label="მოსმენა">${UI.icon("play", 12)}</button>` : "—"}</td>`; }).join("")}
      </tr>`).join("")}</tbody>
    </table></div>`;
  box.querySelectorAll("[data-gen]").forEach((b) => b.addEventListener("click", async () => {
    if (!(await UI.confirm(`წავშალო ნიმუშების თაობა „${b.dataset.label}“?`))) return;
    try {
      await api(`/api/preview/base_ka/${b.dataset.gen}`, { method: "DELETE" });
      renderPreviewMatrix(gens.filter((g) => g.files[0].file.split("/")[0] !== b.dataset.gen), spkLabel);
    } catch (e) { UI.toast(e.message, "error"); }
  }));
  box.querySelectorAll("[data-phr]").forEach((b) => b.addEventListener("click", () => {
    prevPhrase = +b.dataset.phr;
    renderPreviewMatrix(gens, spkLabel);
  }));
  bindPlays(box);
  box.querySelectorAll("[data-seq]").forEach((b) => b.addEventListener("click", () => {
    const btns = [...b.closest("tr").querySelectorAll(".play")];
    let i = 0;
    const next = () => {
      document.querySelectorAll(".play.on").forEach((x) => x.classList.remove("on"));
      if (i >= btns.length) { sampleAudio.onended = null; return; }
      const cur = btns[i++];
      cur.classList.add("on");
      sampleAudio.src = cur.dataset.src;
      sampleAudio.onended = () => setTimeout(next, 400);
      sampleAudio.play();
    };
    next();
  }));
}

// ---------- ხმის შემქმნელი (designer.js) ----------
function renderDesigner() {
  if (!$("#tab-designer").hidden) window.Designer?.render();
}

// ---------- მონაცემები ----------
function renderData(d) {
  $("#data").innerHTML = `
    <div class="card">
      <h2>წყაროები</h2>
      <div class="src-list">${d.sources.map((s) => `
        <div class="src">
          <b>${esc(s.label)}</b>
          <div class="kv small">
            <div><span>დინამიკი</span><b>${s.speakers.toLocaleString()}</b></div>
            <div><span>აუდიო</span><b>${s.hours} სთ (დადასტ. ${s.validated_hours})</b></div>
            <div><span>სუფთა (≥5 წთ)</span><b>${s.good} დინამიკი · ${s.good_hours_cap3} სთ</b></div>
            <div><span>რამდენიმე ადამიანი ერთ ანგარიშზე</span><b>${s.multi_person}</b></div>
          </div>
        </div>`).join("")}</div>
      <p class="muted small">ყველა წყარო CC0 ლიცენზიისაა. ხარისხი: SNR ≥ 25 dB, ერთ ანგარიშზე ერთი ადამიანი, უარყოფითი ხმები &lt; 20%.</p>
    </div>
    <div class="card">
      <h2>Dataset-ები</h2>
      <div class="table-wrap"><table>
        <thead><tr><th>სახელი</th><th>სად</th><th class="num">დინამიკი</th><th class="num">კლიპი</th><th class="num">საათი</th><th>წყარო</th></tr></thead>
        <tbody>${d.datasets.map((x) => `<tr><td class="mono">${esc(x.name)}</td><td>${esc(x.location)}</td>
          <td class="num">${x.speakers ?? (x.speaker ? 1 : "—")}</td><td class="num">${(x.clips || 0).toLocaleString()}</td>
          <td class="num">${x.hours != null ? Number(x.hours).toFixed(1) : "—"}</td>
          <td class="muted small">${esc(x.source || x.speaker || "")}</td></tr>`).join("")}</tbody>
      </table></div>
    </div>`;
}

// ---------- განახლება ----------
async function refresh(fromTab = false) {
  try {
    const [ov, base, tj] = await Promise.all([api("/api/overview"), api("/api/base"), api("/api/train/jobs").catch(() => null)]);
    renderStats(ov, base, tj);
    renderJobs(tj);
    renderLibrarySummary();
    if (!$("#tab-base").hidden || !$("#base").innerHTML) renderBase(base);
    renderDesigner();
    if (fromTab && !$("#tab-settings").hidden) window.Settings?.render();  // ფორმა 60 წამში არ უნდა გასუფთავდეს
    if (!$("#tab-train").hidden) window.Train?.render();
  } catch (e) { console.warn(e); }
}
setInterval(() => { if (!document.hidden) refresh(); }, 60000);
document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", () => refresh(true)));

// ---------- init ----------
(async () => {
  [voices, emotions] = await Promise.all([api("/api/voices"), api("/api/emotions")]);
  renderVoices();
  renderEmotions();
  updateNorm();
  loadRoadmap();
  await refresh(true);
  renderData(await api("/api/data"));
  window.Settings?.maybeFirstRun();
})();
