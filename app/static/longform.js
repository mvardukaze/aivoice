// „გრძელი ტექსტი → ფაილი“: აბზაცები (ხმა/ემოცია თითოზე) → MP3/WAV + SRT (backend: app/longform.py)
(() => {
  const $ = (s) => document.querySelector(s);
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const api = async (url, opts) => {
    const r = await fetch(url, opts);
    const body = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(body.detail?.[0]?.msg || body.detail || r.statusText);
    return body;
  };
  const DRAFT = "longform-draft";
  let overrides = {}; // აბზაცის ნომერი → {voice, emotion}
  let timer = null;

  const voiceList = () => [...$("#voice-select").options].map((o) => ({ id: o.value, name: o.textContent.replace(/ \(სატესტო\)$/, "") }));
  const emoList = async () => (await api("/api/emotions")).presets;
  let EMO = null;

  // აბზაცები + [ხმა, ემოცია] ნიშნები
  function parse(text) {
    const voices = voiceList();
    const paras = text.replace(/\r/g, "").split(/\n\s*\n|\n/).map((p) => p.trim()).filter(Boolean);
    let curVoice = $("#voice-select").value, curEmo = "neutral";
    return paras.map((p, i) => {
      const m = p.match(/^\[([^\]]+)\]\s*/);
      if (m) {
        for (const tag of m[1].split(/[,;]/).map((x) => x.trim().toLowerCase())) {
          const v = voices.find((x) => x.name.toLowerCase() === tag || x.id === tag);
          const e = Object.entries(EMO).find(([k, x]) => x.label.toLowerCase() === tag || k === tag);
          if (v) curVoice = v.id; else if (e) curEmo = e[0];
        }
        p = p.slice(m[0].length);
      }
      const o = overrides[i] || {};
      return { text: p, voice: o.voice || curVoice, emotion: o.emotion || curEmo };
    }).filter((b) => b.text);
  }

  function renderBlocks() {
    const blocks = parse($("#long-text").value);
    const chars = blocks.reduce((n, b) => n + b.text.length, 0);
    $("#long-count").textContent = blocks.length ? `${blocks.length} აბზაცი · ${chars.toLocaleString()} სიმბოლო · ~${Math.max(1, Math.round(chars / 900))} წთ აუდიო` : "";
    if ($("#long-blocks").hidden) return blocks;
    const voices = voiceList();
    $("#long-blocks").innerHTML = blocks.map((b, i) => `
      <div class="long-block" data-i="${i}">
        <div class="long-ctl">
          <span class="muted small">${i + 1}</span>
          <select data-k="voice">${voices.map((v) => `<option value="${esc(v.id)}" ${v.id === b.voice ? "selected" : ""}>${esc(v.name)}</option>`).join("")}</select>
          <select data-k="emotion">${Object.entries(EMO).map(([k, e]) => `<option value="${k}" ${k === b.emotion ? "selected" : ""}>${esc(e.label)}</option>`).join("")}</select>
          <button class="ghost small" data-play title="ამ აბზაცის მოსმენა" aria-label="მოსმენა">${UI.icon("play", 12)}</button>
        </div>
        <div class="long-text small">${esc(b.text.length > 400 ? b.text.slice(0, 400) + "…" : b.text)}</div>
      </div>`).join("");
    $("#long-blocks").querySelectorAll(".long-block").forEach((el) => {
      const i = +el.dataset.i;
      el.querySelectorAll("select").forEach((s) => s.addEventListener("change", () => {
        overrides[i] = { ...(overrides[i] || {}), [s.dataset.k]: s.value };
      }));
      el.querySelector("[data-play]").addEventListener("click", async (e) => {
        const b = parse($("#long-text").value)[i];
        e.target.disabled = true;
        try {
          const r = await api("/api/synthesize", { method: "POST", headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ voice: b.voice, emotion: b.emotion, text: b.text.slice(0, 2000) }) });
          const a = new Audio(r.url); a.play();
        } catch (err) { UI.toast(err.message, "error"); } finally { e.target.disabled = false; }
      });
    });
    return blocks;
  }

  function showState(s) {
    const box = $("#long-status");
    const pct = s.total ? Math.round((s.done / s.total) * 100) : 0;
    const LBL = { queued: "რიგშია…", running: `ხმოვანდება: ${s.done}/${s.total} წინადადება · ${s.seconds} წმ`, encoding: "MP3-ად გარდაქმნა…",
      done: `✓ მზადაა — ${Math.floor(s.seconds / 60)}:${String(Math.round(s.seconds % 60)).padStart(2, "0")}`, failed: "შეცდომა: " + (s.error || "") };
    const files = s.files || [];
    box.innerHTML = `
      ${["queued", "running", "encoding"].includes(s.status) ? `<div class="bar"><span style="width:${pct}%"></span></div>` : ""}
      <div class="${s.status === "failed" ? "err" : ""}">${esc(LBL[s.status] || s.status)}</div>
      ${s.status === "done" ? `
        <audio controls src="/api/longform/${s.id}/${files[0]}" style="width:100%;margin-top:6px"></audio>
        <div class="actions">${files.map((f) => `<a class="ghost small btn" href="/api/longform/${s.id}/${f}" download>${UI.icon("download", 14)} ${f.endsWith(".srt") ? "სუბტიტრები (SRT)" : f.toUpperCase().replace("AUDIO.", "")}</a>`).join("")}</div>` : ""}`;
    const busy = ["queued", "running", "encoding"].includes(s.status);
    $("#long-go").disabled = busy;
    $("#long-cancel").hidden = !busy;
  }

  async function poll(id) {
    clearTimeout(timer);
    const s = await api(`/api/longform/${id}`).catch(() => null);
    if (!s) return;
    showState(s);
    if (["queued", "running", "encoding"].includes(s.status)) timer = setTimeout(() => poll(id), 1500);
  }

  async function init() {
    EMO = await emoList();
    try { $("#long-text").value = localStorage.getItem(DRAFT) || ""; } catch {}
    renderBlocks();
    $("#long-text").addEventListener("input", () => {
      overrides = {};
      try { localStorage.setItem(DRAFT, $("#long-text").value); } catch {}
      clearTimeout(init.t); init.t = setTimeout(renderBlocks, 300);
    });
    $("#long-file").addEventListener("change", async (e) => {
      const f = e.target.files[0];
      if (!f) return;
      let text = await f.text();
      if (f.name.endsWith(".srt")) text = text.replace(/^\d+\s*$|^[\d:,]+ --> [\d:,]+\s*$/gm, "").replace(/\n{2,}/g, "\n");
      $("#long-text").value = text.trim();
      $("#long-text").dispatchEvent(new Event("input"));
      e.target.value = "";
    });
    $("#long-split").addEventListener("click", () => {
      $("#long-blocks").hidden = !$("#long-blocks").hidden;
      $("#long-split").textContent = $("#long-blocks").hidden ? "აბზაცების ნახვა ▾" : "აბზაცების დამალვა ▴";
      renderBlocks();
    });
    $("#voice-select").addEventListener("change", renderBlocks);
    $("#long-go").addEventListener("click", async () => {
      const blocks = parse($("#long-text").value);
      if (!blocks.length) { $("#long-text").focus(); return; }
      try {
        const r = await api("/api/longform", { method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ blocks, pause: +$("#long-pause").value, sentence_pause: +$("#long-spause").value, mp3: $("#long-mp3").checked }) });
        try { localStorage.setItem("longform-job", r.id); } catch {}
        poll(r.id);
      } catch (err) { showState({ status: "failed", error: err.message }); }
    });
    $("#long-cancel").addEventListener("click", async () => {
      const id = (() => { try { return localStorage.getItem("longform-job"); } catch { return null; } })();
      if (id) await api(`/api/longform/${id}/cancel`, { method: "POST" }).catch(() => {});
    });
    try { const id = localStorage.getItem("longform-job"); if (id) poll(id); } catch {}
  }
  // ხმების სია app.js-ში იტვირთება — ცოტა მოვიცადოთ
  const wait = setInterval(() => { if ($("#voice-select").options.length) { clearInterval(wait); init(); } }, 300);
})();
