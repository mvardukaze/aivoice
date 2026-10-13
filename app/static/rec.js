// ხმის ჩამწერი — ჩანართი „ჩაწერა“ (backend: app/recorder.py)
(() => {
  const $ = (s) => document.querySelector(s);
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const api = async (url, opts) => {
    const r = await fetch(url, opts);
    const body = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(body.detail || r.statusText);
    return body;
  };

  let spk = null, state = null, idx = 0;
  let ctx = null, node = null, stream = null, chunks = [], recording = false, recStart = 0, timer = null;
  const player = new Audio();

  // ---------- დინამიკები ----------
  async function showSpeakers() {
    spk = null;
    stopMic();
    const list = await api("/api/rec/speakers");
    $("#rec").innerHTML = `
      <div class="card">
        <h2>ხმის ჩაწერა</h2>
        <p class="muted">ერთი ადამიანი კითხულობს 1200 წინადადებას (~77 წთ მეტყველება, ჩაწერით ~4–6 სთ).
          ჯობს 3–4 დღეზე გადანაწილება — დღეში 1–1.5 სთ, რომ ხმა არ დაიღალოს. შეწყვეტა და გაგრძელება ნებისმიერ დროს შეიძლება.</p>
        ${list.length ? `<div class="spk-list">${list.map((s) => `
          <button class="spk-card" data-id="${esc(s.id)}">
            <b>${esc(s.name)}</b>
            <span class="muted small">${s.recorded} / ${s.total} · ${s.minutes} წთ · კარგი ${s.good}${s.flagged ? ` · <span class="warn-t">შესამოწმებელი ${s.flagged}</span>` : ""}</span>
            <span class="bar"><span style="width:${(s.recorded / s.total * 100).toFixed(1)}%"></span></span>
          </button>`).join("")}</div>` : ""}
      </div>
      <div class="card">
        <h2>ახალი დინამიკი</h2>
        <form id="new-spk" class="new-spk">
          <label class="lbl" for="spk-name">სახელი</label>
          <input id="spk-name" required minlength="2" maxlength="60" placeholder="მაგ. გიორგი">
          <div class="row2">
            <div><label class="lbl" for="spk-gender">სქესი</label>
              <select id="spk-gender"><option value="">—</option><option>კაცი</option><option>ქალი</option></select></div>
            <div><label class="lbl" for="spk-age">ასაკი</label>
              <input id="spk-age" placeholder="მაგ. 45"></div>
          </div>
          <label class="consent"><input type="checkbox" id="spk-consent" required>
            <span>ვეთანხმები, რომ ჩემი ხმის ჩანაწერები გამოყენებულ იქნას ხმოვანი მოდელის შესაქმნელად და ეს მოდელი შეიძლება საჯაროდ გავრცელდეს.</span></label>
          <button class="primary" type="submit">დაწყება</button>
        </form>
      </div>`;
    $("#rec").querySelectorAll(".spk-card").forEach((b) => b.addEventListener("click", () => open(b.dataset.id)));
    $("#new-spk").addEventListener("submit", async (e) => {
      e.preventDefault();
      try {
        const r = await api("/api/rec/speakers", {
          method: "POST", headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ name: $("#spk-name").value, gender: $("#spk-gender").value, age: $("#spk-age").value, consent: $("#spk-consent").checked }),
        });
        open(r.id);
      } catch (err) { alert(err.message); }
    });
  }

  // ---------- ჩამწერი ეკრანი ----------
  async function open(id) {
    spk = id;
    state = await api(`/api/rec/${id}/state`);
    idx = Math.max(0, state.items.findIndex((it) => !it.recorded));
    if (idx < 0) idx = 0;
    $("#rec").innerHTML = `
      <div class="card rec-card">
        <div class="rec-head">
          <button class="ghost small" id="rec-back">← დინამიკები</button>
          <b>${esc(state.speaker.name)}</b>
          <span class="muted small" id="rec-stats"></span>
        </div>
        <div class="bar"><span id="rec-progress"></span></div>
        <div class="rec-nav">
          <button class="ghost small" id="rec-prev" title="წინა (←)">←</button>
          <span class="muted small" id="rec-pos"></span>
          <button class="ghost small" id="rec-next" title="შემდეგი (→)">→</button>
          <button class="ghost small" id="rec-jump">პირველი ჩაუწერელი</button>
          <button class="ghost small" id="rec-export" title="კარგი ჩანაწერები → datasets/rec_…, წვრთნისთვის">dataset-ის შექმნა</button>
        </div>
        <p class="prompt" id="rec-text"></p>
        <div class="meter"><span id="rec-level"></span><i class="meter-ok"></i></div>
        <div class="rec-controls">
          <button class="rec-btn" id="rec-toggle" title="Space">● ჩაწერა</button>
          <button class="ghost" id="rec-play" title="P">▶ მოსმენა</button>
          <button class="ghost" id="rec-del" title="Delete">წაშლა</button>
          <span class="rec-time" id="rec-time"></span>
        </div>
        <div id="rec-qc" class="rec-qc"></div>
        <details class="tips"><summary>რჩევები და მალსახმობები</summary>
          <ul>
            <li><b>Space</b> — ჩაწერის დაწყება/დასრულება; დასრულებისას ავტომატურად შემდეგზე გადადის. <b>←/→</b> — წინა/შემდეგი, <b>P</b> — მოსმენა.</li>
            <li>ჩაწერის დაწყების შემდეგ ~0.5 წამი დაიცადეთ, მერე დაიწყეთ კითხვა; ბოლოს ~0.5 წამი დაიცადეთ, მერე გააჩერეთ.</li>
            <li>იკითხეთ მშვიდად, ბუნებრივად, ერთნაირი ტემპითა და ხმამაღლობით — თეატრალურობა საჭირო არ არის.</li>
            <li>მიკროფონი ყოველთვის ერთსა და იმავე მანძილზე (~15–20 სმ). დონის ზოლი მწვანე ზონაში უნდა იყოს, წითელში არა.</li>
            <li>ჩუმი ოთახი, ექოს გარეშე; ტელეფონი, ვენტილატორი, კონდიციონერი — გამორთეთ.</li>
            <li>შეცდომაზე უბრალოდ თავიდან ჩაწერეთ (← და Space) — ძველი ჩანაწერი ჩანაცვლდება.</li>
          </ul>
          <label class="lbl" for="rec-mic">მიკროფონი</label><select id="rec-mic"></select>
        </details>
      </div>`;
    $("#rec-back").onclick = showSpeakers;
    $("#rec-prev").onclick = () => go(idx - 1);
    $("#rec-next").onclick = () => go(idx + 1);
    $("#rec-jump").onclick = () => go(Math.max(0, state.items.findIndex((it) => !it.recorded)));
    $("#rec-export").onclick = async () => {
      try {
        const r = await api(`/api/rec/${spk}/export`, { method: "POST" });
        alert(`dataset მზადაა: ${r.clips} კლიპი, ${r.hours} სთ (datasets/rec_${spk})`);
      } catch (e) { alert(e.message); }
    };
    $("#rec-toggle").onclick = toggle;
    $("#rec-play").onclick = play;
    $("#rec-del").onclick = del;
    $("#rec-mic").onchange = () => startMic($("#rec-mic").value);
    render();
    await startMic();
  }

  function render() {
    const it = state.items[idx];
    const st = state.stats;
    $("#rec-stats").textContent = `${st.recorded} / ${st.total} · ${st.minutes} წთ · კარგი ${st.good}` +
      (st.flagged ? ` · შესამოწმებელი ${st.flagged}` : "") + (st.checking ? ` · whisper-ით მოწმდება ${st.checking}` : "");
    $("#rec-progress").style.width = `${(st.recorded / st.total) * 100}%`;
    $("#rec-pos").textContent = `${idx + 1} / ${state.items.length} · ${it.id}`;
    $("#rec-text").textContent = it.text;
    $("#rec-play").disabled = !it.recorded;
    $("#rec-del").disabled = !it.recorded;
    $("#rec-qc").innerHTML = it.recorded ? qcHtml(it) : `<span class="muted small">ჯერ არ არის ჩაწერილი</span>`;
  }

  function qcHtml(it) {
    const parts = [];
    if (!it.warnings?.length) parts.push(`<span class="verdict good">✓ ხარისხი კარგია</span>`);
    it.warnings?.forEach((w) => parts.push(`<span class="verdict bad">${esc(w)}</span>`));
    if (it.cer != null) {
      parts.push(it.cer <= 0.3
        ? `<span class="verdict good" title="${esc(it.whisper)}">✓ ტექსტი ემთხვევა (${Math.round((1 - it.cer) * 100)}%)</span>`
        : `<span class="verdict maybe" title="${esc(it.whisper)}">? whisper-მა სხვა გაიგო — მოუსმინეთ</span>`);
    } else parts.push(`<span class="muted small">whisper-ით მოწმდება…</span>`);
    return parts.join(" ");
  }

  function go(i) {
    if (recording) return;
    idx = Math.min(state.items.length - 1, Math.max(0, i));
    render();
  }

  // ---------- მიკროფონი ----------
  async function startMic(deviceId) {
    stopMic();
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        // ხმის „გაუმჯობესება“ გამორთულია — წვრთნას სუფთა, დაუმუშავებელი სიგნალი სჭირდება
        audio: { deviceId: deviceId ? { exact: deviceId } : undefined, echoCancellation: false,
                 noiseSuppression: false, autoGainControl: false, channelCount: 1 },
      });
    } catch (e) {
      $("#rec-qc").innerHTML = `<span class="verdict bad">მიკროფონზე წვდომა ვერ მოხერხდა: ${esc(e.message)}</span>`;
      return;
    }
    ctx = new AudioContext();
    await ctx.audioWorklet.addModule("rec-worklet.js?v=1");
    const src = ctx.createMediaStreamSource(stream);
    node = new AudioWorkletNode(ctx, "capture");
    let peakHold = 0;
    node.port.onmessage = (e) => {
      if (e.data.data) chunks.push(e.data.data);
      peakHold = Math.max(e.data.peak, peakHold * 0.93);
      const lvl = $("#rec-level");
      if (lvl) {
        const db = 20 * Math.log10(peakHold + 1e-6);
        lvl.style.width = `${Math.max(0, Math.min(100, (db + 60) / 60 * 100))}%`;
        lvl.className = peakHold > 0.95 ? "clip" : peakHold > 0.1 ? "ok" : "";
      }
    };
    src.connect(node);
    const devs = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "audioinput");
    const cur = stream.getAudioTracks()[0].getSettings().deviceId;
    $("#rec-mic").innerHTML = devs.map((d) => `<option value="${esc(d.deviceId)}" ${d.deviceId === cur ? "selected" : ""}>${esc(d.label || "მიკროფონი")}</option>`).join("");
  }

  function stopMic() {
    stream?.getTracks().forEach((t) => t.stop());
    ctx?.close();
    stream = ctx = node = null;
  }

  function toggle() { recording ? stop() : start(); }

  function start() {
    if (!node) return;
    player.pause();
    chunks = [];
    recording = true;
    recStart = performance.now();
    node.port.postMessage("start");
    $("#rec-toggle").textContent = "■ გაჩერება";
    $("#rec-toggle").classList.add("on");
    timer = setInterval(() => { $("#rec-time").textContent = ((performance.now() - recStart) / 1000).toFixed(1) + " წმ"; }, 100);
  }

  async function stop() {
    recording = false;
    node.port.postMessage("stop");
    clearInterval(timer);
    $("#rec-toggle").textContent = "● ჩაწერა";
    $("#rec-toggle").classList.remove("on");
    await new Promise((r) => setTimeout(r, 80)); // ბოლო ბლოკები
    const it = state.items[idx];
    const blob = wav(chunks, ctx.sampleRate);
    $("#rec-qc").innerHTML = `<span class="muted small">იგზავნება…</span>`;
    try {
      const r = await api(`/api/rec/${spk}/${it.id}`, { method: "POST", headers: { "Content-Type": "audio/wav" }, body: blob });
      Object.assign(it, { recorded: true, warnings: r.warnings, cer: null, dur: r.speech_sec });
      state.stats = r.stats;
      if (!r.warnings.length) go(idx + 1); // კარგია → შემდეგი
      else render();
    } catch (e) {
      $("#rec-qc").innerHTML = `<span class="verdict bad">შეცდომა: ${esc(e.message)}</span>`;
    }
  }

  function wav(bufs, sr) {
    const n = bufs.reduce((a, b) => a + b.length, 0);
    const dv = new DataView(new ArrayBuffer(44 + n * 2));
    const w = (o, s) => [...s].forEach((c, i) => dv.setUint8(o + i, c.charCodeAt(0)));
    w(0, "RIFF"); dv.setUint32(4, 36 + n * 2, true); w(8, "WAVEfmt ");
    dv.setUint32(16, 16, true); dv.setUint16(20, 1, true); dv.setUint16(22, 1, true);
    dv.setUint32(24, sr, true); dv.setUint32(28, sr * 2, true); dv.setUint16(32, 2, true); dv.setUint16(34, 16, true);
    w(36, "data"); dv.setUint32(40, n * 2, true);
    let o = 44;
    for (const b of bufs) for (let i = 0; i < b.length; i++, o += 2) dv.setInt16(o, Math.max(-1, Math.min(1, b[i])) * 0x7fff, true);
    return new Blob([dv], { type: "audio/wav" });
  }

  function play() {
    const it = state.items[idx];
    if (!it.recorded) return;
    player.src = `/api/rec/${spk}/audio/${it.id}?t=${Date.now()}`;
    player.play();
  }

  async function del() {
    const it = state.items[idx];
    if (!it.recorded || !confirm("წავშალო ეს ჩანაწერი?")) return;
    const r = await api(`/api/rec/${spk}/${it.id}`, { method: "DELETE" });
    Object.assign(it, { recorded: false, warnings: [], cer: null });
    state.stats = r.stats;
    render();
  }

  // whisper-ის შედეგები ფონურად მოდის — ყოველ 10 წამში განახლება
  setInterval(async () => {
    if (!spk || recording || $("#tab-record").hidden) return;
    try {
      const s = await api(`/api/rec/${spk}/state`);
      s.items.forEach((x, i) => Object.assign(state.items[i], x));
      state.stats = s.stats;
      render();
    } catch {}
  }, 10000);

  document.addEventListener("keydown", (e) => {
    if (!spk || $("#tab-record").hidden || ["INPUT", "TEXTAREA", "SELECT"].includes(e.target.tagName)) return;
    if (e.code === "Space") { e.preventDefault(); e.target.blur?.(); toggle(); }
    else if (e.code === "ArrowRight") go(idx + 1);
    else if (e.code === "ArrowLeft") go(idx - 1);
    else if (e.code === "KeyP") play();
  });

  // Space-ის keyup ფოკუსირებულ ღილაკზე „click“-ს იწვევს — ორმაგი ჩართვა/გამორთვა რომ არ მოხდეს
  document.addEventListener("keyup", (e) => { if (spk && e.code === "Space" && !$("#tab-record").hidden) e.preventDefault(); });

  document.querySelector('.tabs button[data-tab="record"]').addEventListener("click", () => { if (!spk) showSpeakers(); });
  if (!$("#tab-record").hidden) showSpeakers();
})();
