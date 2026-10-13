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
];
const STATUS = { ready: "მზადაა", training: "იწვრთნება", selected: "არჩეულია", planned: "დაგეგმილი" };

let emotions = { presets: {}, limits: {} };
let voices = [];
let current = "neutral";

// ---------- tabs ----------
document.querySelectorAll(".tabs button").forEach((b) =>
  b.addEventListener("click", () => {
    document.querySelectorAll(".tabs button").forEach((x) => x.setAttribute("aria-selected", x === b));
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

async function loadEnv(refresh = false) {
  $("#env").innerHTML = `<p class="muted small">მოწმდება…</p>`;
  const env = await api(`/api/env${refresh ? "?refresh=1" : ""}`);
  const groups = {};
  env.checks.forEach((c) => (groups[c.group] ||= []).push(c));
  $("#env").innerHTML = Object.entries(groups).map(([g, list]) => `
    <div class="env-group"><h3>${esc(g)}</h3>
      ${list.map((c) => `
        <div class="env-row">
          <span class="mark ${c.ok ? "ok" : "no"}">${c.ok ? "✓" : "✗"}</span>
          <span>${esc(c.name)}</span>
          <span class="detail">${esc(c.detail)}</span>
        </div>`).join("")}
    </div>`).join("") + `<p class="muted small">დისკზე თავისუფალია ${env.disk_free_gb} GB</p>`;
}
$("#env-refresh").addEventListener("click", () => loadEnv(true));

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

function renderStats(ov, base) {
  const ready = voices.filter((v) => v.status === "ready").length;
  const run = base.run || {};
  const ds = base.dataset || {};
  const items = [
    [`${ready}`, "მზა ხმა"],
    [run.epoch != null ? `${epochShown(run)} / ${run.max_epochs}` : "—", "საბაზო მოდელი: ეპოქა"],
    [ds.speakers ? `${ds.speakers} · ${ds.hours} სთ` : "—", "საბაზო მონაცემები"],
    run.device === "gcp" && run.status === "training"
      ? [`~${hm(run.eta_hours)}`, "დარჩა (Google Cloud L4)"]
      : [`${ov.kaggle_week_hours} / ${ov.kaggle_week_limit} სთ`, "Kaggle ბოლო 7 დღეში"],
  ];
  $("#stats").innerHTML = items.map(([b, s]) => `<div class="stat"><b>${esc(b)}</b><span>${esc(s)}</span></div>`).join("");
}

function progressPct(r) {
  const n = r.steps_per_epoch || 1;
  const e = r.epoch != null ? r.epoch + (r.device === "kaggle" ? 1 : 0) : 0;
  return Math.min(100, ((e * n + (r.step_in_epoch || 0)) / ((r.max_epochs || 1) * n)) * 100);
}

function renderJobs(ov) {
  const vname = Object.fromEntries(voices.map((v) => [v.id, v.name]));
  vname.base_ka = "ქართული საბაზო მოდელი";
  $("#jobs").innerHTML = ov.jobs.length ? ov.jobs.map((r) => `
    <div class="job">
      <div class="run-head">
        <div><b>${esc(vname[r.voice] || r.voice)}</b> <span class="muted small">${esc(DEVICE[r.device] || r.device || "")}${r.devices > 1 ? ` · ${r.devices}× GPU` : ""}</span></div>
        <span class="badge ${r.status === "failed" ? "bad" : r.status === "paused" ? "planned" : "training"}">${RUN_STATUS[r.status] || r.status}</span>
      </div>
      <div class="bar"><span style="width:${progressPct(r).toFixed(1)}%"></span></div>
      <div class="muted small">ეპოქა ${r.device === "gcp" ? epochShown(r) : r.epoch != null ? r.epoch + (r.device === "kaggle" ? 1 : 0) : 0} / ${r.max_epochs}
        ${r.clips ? ` · ${r.clips.toLocaleString()} კლიპი` : ""}${r.num_speakers > 1 ? ` · ${r.num_speakers} დინამიკი` : ""}
        ${r.device !== "kaggle" && r.eta_hours != null && r.status === "training" ? ` · დარჩა ~${r.eta_hours} სთ` : ""}
        ${r.device === "kaggle" && r.kaggle_hours ? ` · Kaggle-ზე ${r.kaggle_hours} სთ` : ""}
        ${r.kaggle_state && r.status === "training" ? ` · ${esc(r.kaggle_state.replace(/.*status /, "").replace(/"/g, ""))}` : ""}
        ${r.error ? ` · <span class="err">${esc(r.error)}</span>` : ""}</div>
    </div>`).join("") : `<div class="empty">ამჟამად არაფერი იწვრთნება.</div>`;
}

// ---------- voices ----------
function voiceCard(v) {
  return `
    <div class="voice ${v.status}">
      <div class="row"><span class="name">${esc(v.name)}</span><span class="badge ${v.status}">${STATUS[v.status] || v.status}</span></div>
      <div class="id">${esc(v.id)}</div>
      <div class="meta">${esc(v.gender)} · ${esc(v.age)}${v.character ? " · " + esc(v.character) : ""}</div>
      <div class="meta">${esc(v.source)}${v.size_mb ? ` · ${v.quality || "medium"}, ${v.sample_rate} Hz, ${v.size_mb} MB` : ""}</div>
      ${v.license === "personal" ? `<div class="lic" title="ნათიას ლიცენზია: მხოლოდ ფიზიკური პირის პირადი გამოყენება">🔒 მხოლოდ პირადი გამოყენებისთვის</div>` : ""}
    </div>`;
}

function renderVoices() {
  const active = voices.filter((v) => v.status !== "planned");
  const planned = voices.filter((v) => v.status === "planned");
  $("#voice-list").innerHTML = active.map(voiceCard).join("");
  $("#planned-list").innerHTML = planned.map(voiceCard).join("");
  $("#planned-count").textContent = planned.length;
  const ready = voices.filter((v) => v.status === "ready");
  $("#voice-select").innerHTML = ready.map((v) => `<option value="${esc(v.id)}">${esc(v.name)}${v.license === "personal" ? " 🔒" : ""}</option>`).join("");
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
  applyPreset(current);
}

function applyPreset(key) {
  current = key;
  const p = emotions.presets[key];
  SLIDERS.forEach((s) => { $(`#sl-${s.key}`).value = p[s.key]; $(`#out-${s.key}`).textContent = p[s.key].toFixed(2); });
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

$("#phrases").innerHTML = PHRASES.map((p, i) => `<button class="chip" data-i="${i}">${esc(p.slice(0, 28))}…</button>`).join("");
$("#phrases").querySelectorAll(".chip").forEach((c) => c.addEventListener("click", () => ($("#text").value = PHRASES[c.dataset.i])));

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
      const { label, ...params } = p;
      const r = await synth(params);
      rows.push(`<div class="compare-row"><span>${esc(label)}</span><audio controls preload="none" src="${r.url}"></audio></div>`);
    }
    list.innerHTML = rows.join("");
  } catch (e) {
    list.innerHTML = `<p class="muted small">შეცდომა: ${esc(e.message)}</p>`;
  }
});

// ---------- sources ----------
const sampleAudio = new Audio();
let sourceFilter = "";

function snrClass(v) { return v == null ? "" : v >= 30 ? "good" : v >= 20 ? "mid" : "bad"; }

function renderSources(sources) {
  if (!sources.length) {
    $("#sources").innerHTML = `<div class="card"><div class="empty">წყაროები ჯერ არ არის დამუშავებული.</div></div>`;
    return;
  }
  const slotName = Object.fromEntries(voices.map((v) => [v.id, v.name]));
  $("#sources").innerHTML = sources.map((src) => {
    const sm = src.summary;
    const slots = [...new Set(src.speakers.map((s) => s.slot).filter(Boolean))];
    const list = src.speakers.filter((s) => !sourceFilter || s.slot === sourceFilter);
    return `
    <div class="card">
      <h2>${esc(sm.source)}</h2>
      <p class="muted">${sm.speakers} დინამიკი · ${sm.hours} საათი ·
        1+ საათი აქვს ${sm.speakers_over["1"]}-ს, 2+ საათი — ${sm.speakers_over["2"]}-ს.
        ქვემოთ კანდიდატებია (${src.speakers.length}), თითოს 8 შემთხვევითი კლიპით.</p>
      <div class="chips">
        <button class="chip" data-slot="" aria-pressed="${!sourceFilter}">ყველა</button>
        ${slots.map((s) => `<button class="chip" data-slot="${s}" aria-pressed="${sourceFilter === s}">${esc(slotName[s] || s)}</button>`).join("")}
      </div>
      <div class="legend muted small">F0 — ხმის სიმაღლე (კაცი ~85–155 Hz, ქალი ~165–255 Hz) ·
        SNR — ფონური ხმაურის სისუფთავე (30+ კარგი, &lt;20 ხმაურიანი) · ⚠ — კლიპებს შორის ტონი ძალიან იცვლება, შეიძლება ანგარიშზე რამდენიმე ადამიანი იწერდეს</div>
      <div class="table-wrap"><table class="spk">
        <thead><tr><th>დინამიკი</th><th class="num">საათი</th><th class="num">აუდიო</th><th>სქესი · ასაკი</th>
          <th class="num">F0</th><th class="num">SNR</th><th>შეფასება</th><th>სლოტი</th><th>ნიმუშები</th></tr></thead>
        <tbody>${list.map((s) => `
          <tr class="${s.verdict === "bad" ? "rejected" : ""}">
            <td class="mono">${esc(s.id)}</td>
            <td class="num" title="დადასტურებული: ${s.validated_hours} სთ">${s.hours.toFixed(1)}</td>
            <td class="num" title="კლიპები, რომლებიც აუდიო-არქივშია">${s.audio_clips ?? "—"}</td>
            <td>${s.gender.startsWith("male") ? "კაცი" : s.gender.startsWith("female") ? "ქალი" : "—"} · ${esc(s.age_ka || "—")}</td>
            <td class="num">${s.f0_median ? Math.round(s.f0_median) : "—"}${s.f0_spread > 3 ? ` <span title="F0-ის გაფანტულობა ${s.f0_spread} ნახევარტონი">⚠</span>` : ""}</td>
            <td class="num"><span class="snr ${snrClass(s.snr_db)}">${s.snr_db ?? "—"}</span></td>
            <td>${s.verdict === "good" ? `<span class="verdict good" title="${esc(s.verdict_note)}">✓ კარგი</span>`
              : s.verdict === "bad" ? `<span class="verdict bad" title="${esc(s.verdict_note)}">✗ არ ვარგა</span>`
              : s.verdict === "maybe" ? `<span class="verdict maybe" title="${esc(s.verdict_note)}">? საეჭვო</span>` : ""}</td>
            <td>${s.verdict === "bad" ? "" : `${s.verdict === "good" ? "<b>" : '<span class="muted">'}${esc(slotName[s.slot] || "")}${s.verdict === "good" ? "</b>" : "</span>"}`}</td>
            <td class="samples">${s.samples.map((x, i) =>
              `<button class="play" data-src="/api/sample/${src.name}/${x.file}" title="${esc(x.text)}">${i + 1}</button>`).join("")}</td>
          </tr>`).join("")}</tbody>
      </table></div>
      <p class="muted small" id="now-playing"></p>
    </div>`;
  }).join("");

  $("#sources").querySelectorAll(".chip[data-slot]").forEach((c) =>
    c.addEventListener("click", () => { sourceFilter = c.dataset.slot; renderSources(sources); }));
  $("#sources").querySelectorAll(".play").forEach((b) =>
    b.addEventListener("click", () => {
      $("#sources").querySelectorAll(".play.on").forEach((x) => x.classList.remove("on"));
      b.classList.add("on");
      sampleAudio.src = b.dataset.src;
      sampleAudio.play();
      $("#now-playing").textContent = "▶ " + b.title;
    }));
}
sampleAudio.addEventListener("ended", () => document.querySelectorAll(".play.on").forEach((x) => x.classList.remove("on")));

// ---------- წვრთნები (მიმოხილვა → „ყველა წვრთნა“) ----------
function renderRuns(data) {
  const runs = data.runs || [];
  const vname = Object.fromEntries(voices.map((v) => [v.id, v.name]));
  vname.base_ka = "ქართული საბაზო მოდელი";
  const q = data.queue || {};
  const queueHtml = Object.keys(q).length ? `<div class="queue muted small">${Object.entries(q).map(([dev, list]) =>
    `<div><b>${dev.toUpperCase()}</b> რიგი: ${list.map((s) => {
      const r = runs.find((x) => x.voice === s);
      const cls = r ? (r.status === "done" ? "q-done" : r.status === "failed" ? "q-fail" : "q-now") : "";
      return `<span class="${cls}">${esc(vname[s] || s)}</span>`;
    }).join(" → ")}</div>`).join("")}</div>` : "";
  if (!runs.length) {
    $("#runs").innerHTML = queueHtml + `<div class="empty">წვრთნა ჯერ არ დაწყებულა.</div>`;
    return;
  }
  $("#runs").innerHTML = queueHtml + runs.map((r) => {
    const n = r.steps_per_epoch || 1;
    const pct = Math.min(100, (((r.epoch || 0) * n + (r.step_in_epoch || 0)) / ((r.max_epochs || 1) * n)) * 100);
    const previews = [...(r.previews || [])].reverse().sort((a, b) => (b.reference ? 1 : 0) - (a.reference ? 1 : 0));
    return `
    <div class="run">
      <div class="run-head">
        <div><b>${esc(vname[r.voice] || r.voice)}</b> <span class="mono muted">${esc(r.voice)}</span></div>
        <div><span class="badge ${r.status === "done" ? "ready" : r.status === "failed" || r.status === "paused" ? "planned" : "training"}">${RUN_STATUS[r.status] || r.status}</span>
          <span class="badge planned">${esc((r.device || "").toUpperCase())} · batch ${r.batch}</span></div>
      </div>
      <div class="bar"><span style="width:${pct.toFixed(1)}%"></span></div>
      <div class="muted small">epoch ${r.epoch ?? 0} / ${r.max_epochs} · ${r.clips} კლიპი ·
        ${r.sec_per_step ? r.sec_per_step + " წმ/ნაბიჯი" : ""}
        ${r.status === "training" && r.eta_hours != null ? ` · დარჩა ~${r.eta_hours} სთ` : ""}
        ${r.error ? ` · <span class="err">${esc(r.error)}</span>` : ""}</div>
      ${previews.length ? `<div class="previews">${previews.map((p) => `
        <div class="prev-row"><span class="small">${p.reference ? `<b>${esc(p.label)}</b>` : `${p.final ? "<b>საბოლოო</b>" : esc(p.label || "#" + p.n)} · epoch ${p.epoch ?? "—"} · ${esc((p.time || "").slice(5, 16).replace("T", " "))}`}</span>
          <span class="samples">${p.files.slice(0, 8).map((f, i) =>
            `<button class="play" data-src="/api/preview/${esc(r.voice)}/${f.file}" title="${esc(f.text)}">${i + 1}</button>`).join("")}</span></div>`).join("")}</div>`
        : `<p class="muted small">პირველი მოსასმენი ნიმუში ~45 წუთში გამოჩნდება.</p>`}
    </div>`;
  }).join("");
  $("#runs").querySelectorAll(".play").forEach((b) => b.addEventListener("click", () => {
    document.querySelectorAll(".play.on").forEach((x) => x.classList.remove("on"));
    b.classList.add("on");
    sampleAudio.onended = null;  // ხელით დაჭერა წყვეტს „▶ ყველა“-ს რიგს
    sampleAudio.src = b.dataset.src;
    sampleAudio.play();
  }));
}


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
    <div class="card">
      <h2>ქართული საბაზო მოდელი</h2>
      <p class="muted">${esc(b.note || "")}. მისი ამოცანაა ქართული გამოთქმის სწავლა ბევრ ხმაზე — მისგან შეიქმნება საბაზისო სინთეზური ხმა,
        შემდეგ ქალისა და კაცის ხმები. ლიცენზიისგან თავისუფალია.</p>
      <div class="kv">
        <div><span>საწყისი წერტილი</span><b>LJSpeech (ინგლისური, public domain)</b></div>
        <div><span>დინამიკი</span><b>${ds.speakers ?? "—"}</b></div>
        <div><span>მონაცემი</span><b>${ds.hours ?? "—"} სთ · ${(ds.clips || 0).toLocaleString()} კლიპი</b></div>
        <div><span>სად იწვრთნება</span><b>${whereTraining(r, job)}</b></div>
        <div><span>ეპოქა</span><b>${r?.epoch != null ? r.epoch + 1 : 0} / ${job.max_epochs || "—"}</b></div>
        ${r?.device === "gcp" && r.eta_hours != null ? `<div><span>დასრულება</span><b>~${hm(r.eta_hours)} · ${esc(etaClock(r.eta_hours))}</b></div>` : ""}
        <div><span>Kaggle-ზე სულ</span><b>${r?.kaggle_hours ?? 0} სთ · ბოლო 7 დღეში ${b.kaggle_week_hours}/${b.kaggle_week_limit}</b></div>
      </div>
      ${r ? `
        <div class="bar big"><span style="width:${progressPct(r).toFixed(1)}%"></span></div>
        <p class="small"><span class="badge ${r.status === "failed" ? "bad" : r.status === "done" ? "ready" : "training"}">${RUN_STATUS[r.status] || r.status}</span>
          ${r.kaggle_state && r.status === "training" && r.device !== "gcp" ? `<span class="muted"> · ${esc(r.kaggle_state.replace(/.*status /, "").replace(/"/g, ""))}</span>` : ""}
          ${r.session_started && r.device !== "gcp" ? `<span class="muted"> · მიმდინარე სესია დაიწყო ${esc(r.session_started.slice(5, 16).replace("T", " "))}</span>` : ""}
          ${r.error ? ` · <span class="err">${esc(r.error)}</span>` : ""}</p>
        ${gcpLine(r)}
        ${r.device === "gcp" ? "" : localLine(r.local)}`
        : `<div class="empty">წვრთნა ჯერ არ დაწყებულა — dataset Kaggle-ზე იტვირთება.</div>`}
    </div>

    <div class="card">
      <h2>მოსასმენი ნიმუშები</h2>
      <p class="muted small">ერთი და იგივე ფრაზები ოთხი ხმით (თქვენ მიერ მოწონებული დინამიკები) — ყოველ თაობაზე.
        აირჩიეთ ფრაზა და შეადარეთ სვეტები: ძველი მარცხნივ, ახალი მარჯვნივ. „▶ ყველა“ ერთ ხმას ყველა თაობით ზედიზედ უკრავს.
        ეპოქიდან ეპოქაზე ინგლისური აქცენტი უნდა ქრებოდეს და „ღ“, „ყ“ უნდა იწმინდებოდეს.</p>
      <div id="prev-matrix"></div>
    </div>

    <div class="grid2">
      <div class="card">
        <h2>ხმის სიმაღლის განაწილება</h2>
        <p class="muted small">${ds.low ?? 0} დაბალი (კაცის ტიპის) · ${ds.high ?? 0} მაღალი (ქალის ტიპის) ხმა · დინამიკზე მედიანა ${ds.minutes_median ?? "—"} წთ</p>
        <div class="hist">${(ds.f0_hist || []).map((h) => `
          <div class="hcol" title="${h.from}–${h.to} Hz: ${h.n} დინამიკი"><span style="height:${(h.n / maxN * 100).toFixed(0)}%"></span><i>${h.from}</i></div>`).join("")}</div>
        <p class="muted small">Hz — ხმის ძირითადი სიხშირე. საბაზისო სინთეზური ხმა ~150–170 Hz-ზე იქნება — არც კაცის, არც ქალის.</p>
      </div>
      <div class="card">
        <h2>სესიები</h2>
        ${sessions.length ? `<table><thead><tr><th>დასრულდა</th><th class="num">საათი</th><th class="num">ეპოქა</th><th>შედეგი</th></tr></thead>
          <tbody>${sessions.map((x) => `<tr><td>${esc((x.ended || "").slice(5, 16).replace("T", " "))}</td><td class="num">${x.hours ?? "—"}</td>
            <td class="num">${x.epoch != null ? x.epoch + 1 : "—"}</td><td>${esc(RUN_STATUS[x.status] || x.status || "")}</td></tr>`).join("")}</tbody></table>`
          : `<div class="empty">დასრულებული სესია ჯერ არ არის.</div>`}
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
        <span>ეპოქა ${g.epoch != null ? g.epoch + 1 : "—"} · ${esc((g.time || "").slice(5, 10))}</span></th>`).join("")}</tr></thead>
      <tbody>${speakers.map((sp) => `<tr>
        <td><span class="small">${esc(spkLabel(sp))}</span>
          <button class="ghost small seq" data-seq="${esc(sp)}" title="ყველა თაობა ზედიზედ">▶ ყველა</button></td>
        ${gens.map((g) => { const f = cell(g, sp);
          return `<td>${f ? `<button class="play" data-src="${src(f)}" title="${esc(g.label || "")}">▶</button>` : "—"}</td>`; }).join("")}
      </tr>`).join("")}</tbody>
    </table></div>`;
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
async function refresh() {
  try {
    const [ov, base] = await Promise.all([api("/api/overview"), api("/api/base")]);
    renderStats(ov, base);
    renderJobs(ov);
    if (!$("#tab-base").hidden || !$("#base").innerHTML) renderBase(base);
    renderDesigner();
    if ($("#runs").closest("details").open) renderRuns(await api("/api/training"));
  } catch (e) { console.warn(e); }
}
setInterval(() => { if (!document.hidden) refresh(); }, 60000);
document.querySelectorAll(".tabs button").forEach((b) => b.addEventListener("click", refresh));

// ---------- init ----------
(async () => {
  [voices, emotions] = await Promise.all([api("/api/voices"), api("/api/emotions")]);
  renderVoices();
  renderEmotions();
  loadRoadmap();
  await refresh();
  const [runs, sources, data] = await Promise.all([api("/api/training"), api("/api/sources"), api("/api/data")]);
  renderRuns(runs);
  renderSources(sources);
  renderData(data);
  $("#env-box").addEventListener("toggle", (e) => { if (e.target.open && !$("#env").innerHTML) loadEnv(); }, { once: false });
})();
