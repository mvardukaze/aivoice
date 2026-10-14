// „მონაცემები → ხარისხი“: Common Voice-ის ჩანაწერების შემოწმება (backend: app/quality_api.py, tools/clip_quality.py)
(() => {
  const $ = (s) => document.querySelector(s);
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const api = async (url, opts) => {
    const r = await fetch(url, opts);
    const body = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(body.detail?.[0]?.msg || body.detail || r.statusText);
    return body;
  };
  const post = (url, data = {}) => api(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });
  const pct = (x) => (x == null ? "—" : Math.round(x * 100) + "%");
  const DEC = { auto: "ავტო", include: "შევიდეს", exclude: "ამოვრიცხოთ" };
  let st = { show: "all", pitch: "all", sort: "bad", dir: 1, limit: 60, open: null };  // dir: 1 — ცუდიდან, -1 — კარგიდან
  let data = null, timer = null;
  const audio = new Audio();

  async function render() {
    const box = $("#quality");
    if (!box) return;
    data = await api("/api/quality").catch((e) => ({ error: e.message }));
    if (data.error) { box.innerHTML = `<div class="card"><div class="empty">${esc(data.error)}</div></div>`; return; }
    const p = data.progress || {}, t = data.totals;
    const running = data.busy || (p.stage && p.stage !== "done" && p.updated && Date.now() - new Date(p.updated) < 10 * 60e3);
    const done = p.total ? Math.round(((p.done || t.clips) / p.total) * 100) : 0;
    box.innerHTML = `
      <div class="card">
        <div class="card-head"><h2>ჩანაწერების ხარისხი — Common Voice 27</h2>
          <div class="actions" style="margin:0">${running
            ? `<button class="ghost small" id="q-stop">გაჩერება</button>`
            : `<button class="primary small" id="q-run">${t.clips ? "გაგრძელება" : "შემოწმების დაწყება"}</button>`}</div></div>
        <p class="muted small">თითო ჩანაწერი მოწმდება: ემთხვევა თუ არა ტექსტს (MMS), შიშინი, ხმაური, ჩუმი ხმა, გადატვირთვა, სიჩქარე.
          ცუდი ჩანაწერები ამოვარდება, დინამიკი — თუ მისი ჩანაწერების დიდი ნაწილი ცუდია. ბოლო სიტყვა თქვენია.</p>
        ${p.total ? `<div class="small">${running ? `<b>მიმდინარეობს:</b> ` : ""}${(p.done || t.clips).toLocaleString()} / ${p.total.toLocaleString()} ჩანაწერი
          ${p.rate ? ` · ${p.rate}/წმ` : ""}${running && p.eta_h != null ? ` · დარჩა ~${p.eta_h} სთ` : ""}${p.stage === "load" ? " · მზადდება (არქივის კითხვა)…" : ""}${p.stage === "cooling" ? ` · <span class="warn-text">ვიდეობარათი ცივდება (${p.temp}°C) — 70°C-ზე გააგრძელებს</span>` : p.temp ? ` · ვიდეობარათი ${p.temp}°C` : ""}</div>
          <div class="bar"><span style="width:${done}%"></span></div>` : ""}
        ${t.clips ? `<div class="kv">
          <div><span>შემოწმებული</span><b>${t.clips.toLocaleString()} · ${t.hours} სთ</b></div>
          <div><span>კარგი ჩანაწერი</span><b>${pct(t.good / t.clips)} · ${t.good_hours} სთ</b></div>
          <div><span>შევა dataset-ში</span><b>${t.include} დინამიკი · ${t.include_hours} სთ</b></div>
          <div><span>მათგან დაბალი ხმა</span><b>${t.include_low}</b></div>
        </div>
        <div class="chips small">${Object.entries(t.why).sort((a, b) => b[1] - a[1]).map(([k, n]) =>
          `<span class="chip" title="${esc(data.rules[k] || k)}">${esc(data.rules[k] || k)}: ${n.toLocaleString()}</span>`).join("")}</div>` : ""}
      </div>
      ${t.clips ? `<div id="sel-card" class="card"></div>` + earCard() + speakersCard() : ""}`;
    bind(box);
    renderSel();
    clearTimeout(timer);
    if (running) timer = setTimeout(() => { if (!$("#tab-data").hidden && !st.open) { const y = window.scrollY; render().then(() => window.scrollTo(0, y)); } }, 30000);
  }

  function filtered() {
    let r = data.speakers.slice();
    if (st.show === "new") r = r.filter((x) => !x.current);
    if (st.show === "current") r = r.filter((x) => x.current);
    if (st.show === "exclude") r = r.filter((x) => x.final === "exclude");
    if (st.show === "manual") r = r.filter((x) => x.decision !== "auto");
    if (st.show === "inset") r = r.filter((x) => Object.keys(x.in_sets || {}).length);
    if (st.show === "notset") r = r.filter((x) => !Object.keys(x.in_sets || {}).length);
    if (st.pitch === "low") r = r.filter((x) => (x.f0 || 999) < 165);
    if (st.pitch === "high") r = r.filter((x) => (x.f0 || 0) >= 165);
    const key = { bad: (x) => -x.bad_share, cer: (x) => -(x.cer || 0), hiss: (x) => -(x.hiss || -99), snr: (x) => x.snr || 0,
      min: (x) => -x.good_min, f0: (x) => x.f0 || 0 }[st.sort];
    return r.sort((a, b) => (key(a) - key(b)) * st.dir);
  }

  // ---------- შერჩეული (საუკეთესო ~40 სთ): შემთხვევითი მოსმენა; X — ჩანაწერის ამოღება, D — მთელი დინამიკის ----------
  const SELN = "base_ka_v2";
  const sel = { clips: [], i: 0, playing: false, stats: null, rej: 0 };
  async function renderSel() {
    const box = document.getElementById("sel-card");
    if (!box) return;
    if (!sel.stats) {
      const d = await api(`/api/quality/selection/${SELN}`).catch(() => null);
      if (!d) { box.hidden = true; return; }
      Object.assign(sel, { clips: d.clips, stats: d.stats, rej: d.rejected, rejSpk: d.rejected_speakers.length });
    }
    const s = sel.stats, c = sel.clips[sel.i];
    box.innerHTML = `
      <div class="card-head"><h2>შერჩეული — საუკეთესო ${s.hours} სთ</h2><span class="muted small">${sel.i + 1} / ${sel.clips.length} შემთხვევითი</span></div>
      <div class="kv"><div><span>ჩანაწერი</span><b>${s.clips.toLocaleString()}</b></div><div><span>დინამიკი</span><b>${s.speakers} (ახალი ${s.new_speakers})</b></div>
        <div><span>დაბალი ხმა</span><b>${s.low_speakers} · ${s.low_hours} სთ</b></div><div><span>MMS საშუალოდ</span><b>${(s.mean_cer * 100).toFixed(1)}%</b></div>
        <div><span>თქვენ ამოიღეთ</span><b>${sel.rej} ჩანაწერი · ${sel.rejSpk || 0} დინამიკი</b></div></div>
      <p class="muted small">შემთხვევითი ჩანაწერები შერჩეულიდან, ზედიზედ. <kbd>X</kbd> — ეს ჩანაწერი ცუდია · <kbd>D</kbd> — მთელი დინამიკი ცუდია ·
        <kbd>→</kbd> შემდეგი · <kbd>Space</kbd> პაუზა. (ეს კლავიშები მუშაობს, როცა „შერჩეული“ უკრავს.)</p>
      <div class="ear-row">
        <button class="primary small" id="sel-play">${sel.playing ? "❚❚ პაუზა" : "▶ შემთხვევითი მოსმენა"}</button>
        <button class="ghost small" id="sel-next">▶▶</button>
        <div class="ear-now">${c ? `<b class="mono">${c.spk}</b> <span class="muted small">MMS ${Math.round(c.cer * 100)}% · SNR ${c.snr} · ${c.dur} წმ</span>
          <div class="small">${esc(c.text)}</div>` : "—"}</div>
        <button class="ghost small" id="sel-x">✕ ჩანაწერი ცუდია</button>
        <button class="ghost small danger" id="sel-d">დინამიკი ცუდია</button>
      </div>`;
    $("#sel-play").onclick = () => (sel.playing ? selPause() : selPlay());
    $("#sel-next").onclick = () => selPlay(sel.i + 1);
    $("#sel-x").onclick = () => selReject("clip");
    $("#sel-d").onclick = () => selReject("spk");
  }
  function selPlay(i = sel.i) {
    if (ear.playing) earPause();
    sel.i = Math.max(0, Math.min(i, sel.clips.length - 1));
    const c = sel.clips[sel.i];
    if (!c) return;
    sel.playing = true;
    audio.src = `/api/quality/clip/${encodeURIComponent(c.id)}`;
    audio.onended = () => { if (sel.playing) setTimeout(() => sel.playing && selPlay(sel.i + 1), 500); };
    audio.play().catch(() => {});
    renderSel();
  }
  function selPause() { sel.playing = false; audio.pause(); renderSel(); }
  async function selReject(kind) {
    const c = sel.clips[sel.i];
    if (!c) return;
    const r = await post(`/api/quality/selection/${SELN}/reject`, kind === "clip" ? { id: c.id } : { spk: c.spk }).catch((e) => { window.UI?.toast(e.message, "error"); });
    if (!r) return;
    sel.rej = r.rejected; sel.rejSpk = r.rejected_speakers;
    if (kind === "spk") { sel.clips = sel.clips.filter((x, j) => j <= sel.i || x.spk !== c.spk); window.UI?.toast(`${c.spk} — ამოღებულია`, "ok", 2500); }
    selPlay(sel.i + 1);
  }

  // ---------- ყურით გადარჩევა: თითო დინამიკის ტიპიური ჩანაწერი ზედიზედ; I — შევიდეს, X — ამოვრიცხოთ ----------
  const ear = { i: 0, playing: false, queue: [] };
  function earCard() {
    ear.queue = filtered().filter((r) => r.sample);
    ear.i = Math.min(ear.i, Math.max(0, ear.queue.length - 1));
    const r = ear.queue[ear.i];
    return `<div class="card ear-card">
      <div class="card-head"><h2>ყურით გადარჩევა</h2><span class="muted small">${ear.queue.length ? `${ear.i + 1} / ${ear.queue.length}` : "სია ცარიელია"} · ფილტრი და რიგი — ქვემოთა ცხრილისა</span></div>
      <p class="muted small">თითო დინამიკის ერთი ტიპიური (კარგი) ჩანაწერი, ზედიზედ. კლავიშები: <kbd>I</kbd> შევიდეს · <kbd>X</kbd> ამოვრიცხოთ · <kbd>A</kbd> ავტო ·
        <kbd>→</kbd> შემდეგი · <kbd>←</kbd> წინა · <kbd>Space</kbd> პაუზა. გადაწყვეტილების შემდეგ თავისით გადადის შემდეგზე.</p>
      <div class="ear-row">
        <button class="ghost small" id="ear-prev" aria-label="წინა">◀</button>
        <button class="primary small" id="ear-play">${ear.playing ? "❚❚ პაუზა" : "▶ დაწყება"}</button>
        <button class="ghost small" id="ear-next" aria-label="შემდეგი">▶▶</button>
        <div class="ear-now" id="ear-now">${r ? earNow(r) : "—"}</div>
        <div class="ear-dec">${r ? decSeg(r) : ""}</div>
      </div>
      <div class="bar"><span style="width:${ear.queue.length ? ((ear.i + 1) / ear.queue.length) * 100 : 0}%"></span></div>
    </div>`;
  }
  const earNow = (r) => `<b class="mono">${r.spk}</b> ${r.current ? "" : `<span class="badge testing">ახალი</span>`}
    <span class="muted small">${r.f0 ? Math.round(r.f0) + " Hz · " : ""}ცუდი ${pct(r.bad_share)} · კარგი ${r.good_min} წთ · ავტო: ${r.suggest === "include" ? "შევიდეს" : "ამოვრიცხოთ"}</span>`;
  function earShow() {
    const r = ear.queue[ear.i];
    const card = document.querySelector(".ear-card");
    if (!card || !r) return;
    card.querySelector("#ear-now").innerHTML = earNow(r);
    card.querySelector(".ear-dec").innerHTML = decSeg(r);
    bindSeg(r.spk);
    card.querySelector(".card-head .muted").textContent = `${ear.i + 1} / ${ear.queue.length} · ფილტრი და რიგი — ქვემოთა ცხრილისა`;
    card.querySelector(".bar span").style.width = `${((ear.i + 1) / ear.queue.length) * 100}%`;
    card.querySelector("#ear-play").textContent = ear.playing ? "❚❚ პაუზა" : "▶ დაწყება";
    document.querySelectorAll("tr[data-spk].ear-on").forEach((x) => x.classList.remove("ear-on"));
    document.querySelector(`tr[data-spk="${r.spk}"]`)?.classList.add("ear-on");
  }
  function earPlay(i = ear.i) {
    if (sel.playing) { sel.playing = false; renderSel(); }
    ear.i = Math.max(0, Math.min(i, ear.queue.length - 1));
    const r = ear.queue[ear.i];
    if (!r) return;
    ear.playing = true;
    audio.src = `/api/quality/clip/${encodeURIComponent(r.sample)}`;
    audio.onended = () => { if (ear.playing) setTimeout(() => ear.playing && earPlay(ear.i + 1), 600); };
    audio.play().catch(() => {});
    earShow();
  }
  function earPause() { ear.playing = false; audio.pause(); earShow(); }
  async function earDecide(decision) {
    const r = ear.queue[ear.i];
    if (!r) return;
    await setDecision(r.spk, decision);
    if (ear.playing || decision !== "auto") earPlay(ear.i + 1); else earShow();
  }
  function bindEar() {
    $("#ear-play")?.addEventListener("click", () => (ear.playing ? earPause() : earPlay()));
    $("#ear-next")?.addEventListener("click", () => earPlay(ear.i + 1));
    $("#ear-prev")?.addEventListener("click", () => earPlay(ear.i - 1));
    const r = ear.queue[ear.i];
    if (r) bindSeg(r.spk);
  }
  document.addEventListener("keydown", (e) => {
    if ($("#tab-data")?.hidden || !document.querySelector(".ear-card") || document.querySelector("dialog[open]")
      || /INPUT|TEXTAREA|SELECT/.test(document.activeElement?.tagName)) return;
    const k = e.key.toLowerCase();
    if (sel.playing) {  // „შერჩეულის“ მოსმენა
      if (e.code === "Space") { e.preventDefault(); selPause(); }
      else if (e.key === "ArrowRight") selPlay(sel.i + 1);
      else if (e.key === "ArrowLeft") selPlay(sel.i - 1);
      else if (k === "x" || k === "ხ") selReject("clip");
      else if (k === "d" || k === "დ") selReject("spk");
      return;
    }
    if (e.code === "Space") { e.preventDefault(); ear.playing ? earPause() : earPlay(); }
    else if (e.key === "ArrowRight") earPlay(ear.i + 1);
    else if (e.key === "ArrowLeft") earPlay(ear.i - 1);
    else if (k === "i" || k === "შ") earDecide("include");
    else if (k === "x" || k === "ხ") earDecide("exclude");
    else if (k === "a" || k === "ა") earDecide("auto");
  });

  // სვეტის სათაური — დაჭერით დალაგება; იგივეზე მეორედ — მიმართულება
  const th = (k, t, title = "") => `<th class="num q-sortable ${st.sort === k ? "on" : ""}" data-sort="${k}" title="${esc(title || "დალაგება")}">${t}${st.sort === k ? (st.dir === 1 ? " ↓" : " ↑") : ""}</th>`;

  function speakersCard() {
    const rows = filtered();
    const opt = (v, cur, t) => `<option value="${v}" ${v === cur ? "selected" : ""}>${t}</option>`;
    return `<div class="card">
      <div class="card-head"><h2>დინამიკები</h2><span class="muted small">${rows.length} ნაჩვენები</span></div>
      <div class="aud-opts">
        <label>ჩვენება <select id="q-show">${opt("all", st.show, "ყველა")}${opt("exclude", st.show, "ამოსარიცხი")}${opt("new", st.show, "ახალი კანდიდატები")}${opt("current", st.show, "ახლანდელი 296")}${opt("manual", st.show, "ჩემი გადაწყვეტილებები")}${opt("inset", st.show, "უკვე გაშვებული (v2)")}${opt("notset", st.show, "ჯერ გაუშვებელი")}</select></label>
        <label>ხმა <select id="q-pitch">${opt("all", st.pitch, "ყველა")}${opt("low", st.pitch, "დაბალი (კაცის ტიპის)")}${opt("high", st.pitch, "მაღალი")}</select></label>
        <label>დალაგება <select id="q-sort">${opt("bad", st.sort, "ცუდი ჩანაწერების წილით")}${opt("cer", st.sort, "MMS შეცდომით")}${opt("hiss", st.sort, "შიშინით")}${opt("snr", st.sort, "ხმაურით")}${opt("min", st.sort, "კარგი წუთებით")}${opt("f0", st.sort, "სიმაღლით")}</select></label>
        <button class="ghost small" id="q-dir" title="მიმართულების შეცვლა">${st.dir === 1 ? "↓ ცუდიდან" : "↑ კარგიდან"}</button>
      </div>
      <div class="table-wrap"><table class="q-table">
        <thead><tr><th>დინამიკი</th>${th("f0", "Hz")}${th("min", "კარგი წთ")}${th("bad", "ცუდი")}${th("cer", "MMS")}
          ${th("snr", "SNR", "მეტყველება − ფონი, dB (მეტი = უკეთესი)")}${th("hiss", "შიშინი", "შიშინი, dB (ნაკლები = უკეთესი)")}
          <th>რატომ</th><th>გადაწყვეტილება</th><th></th></tr></thead>
        <tbody>${rows.slice(0, st.limit).map(row).join("")}</tbody></table></div>
      ${rows.length > st.limit ? `<div class="actions"><button class="ghost small" id="q-more">კიდევ 60 (${rows.length - st.limit})</button></div>` : ""}
    </div>`;
  }

  function row(r) {
    const top = Object.entries(r.why).sort((a, b) => b[1] - a[1]).slice(0, 2).map(([k, n]) => `${esc(data.rules[k] || k)} ${n}`).join(", ");
    return `<tr class="${r.final === "exclude" ? "q-ex" : ""}" data-spk="${r.spk}">
      <td><span class="mono small">${r.spk}</span> ${r.current ? "" : `<span class="badge testing">ახალი</span>`}
        ${Object.entries(r.in_sets || {}).map(([n, m]) => `<span class="badge ready" title="${esc(n)}-ში გაშვებულია ${m} წუთი">${esc(n.replace("base_ka_", ""))} · ${m} წთ</span>`).join(" ")}</td>
      <td class="num">${r.f0 ? Math.round(r.f0) : "—"}</td>
      <td class="num">${r.good_min} <span class="muted small">/ ${r.min}</span></td>
      <td class="num ${r.bad_share > 0.4 ? "err" : ""}">${pct(r.bad_share)}</td>
      <td class="num" title="${r.mms_n ? `MMS-ით შემოწმებული: ${r.mms_n}` : "MMS ჯერ არ შემოწმებულა"}">${r.cer == null ? `<span class="muted">${r.mms_n ? "…" : "—"}</span>` : pct(r.cer)}</td><td class="num">${r.snr ?? "—"}</td><td class="num">${r.hiss ?? "—"}</td>
      <td class="small muted">${top}</td>
      <td>${decSeg(r)}</td>
      <td><button class="ghost small ${st.open === r.spk ? "on" : ""}" data-open>მოსმენა</button></td>
    </tr>`;
  }

  const decSeg = (r) => `<div class="seg small" data-seg="${r.spk}">${["auto", "include", "exclude"].map((d) =>
    `<button data-dec="${d}" class="${r.decision === d ? "on" : ""}" title="${d === "auto" ? "ავტომატურად: " + (r.suggest === "include" ? "შევიდეს" : "ამოვრიცხოთ") : ""}">${DEC[d]}${d === "auto" ? (r.suggest === "include" ? " ✓" : " ✗") : ""}</button>`).join("")}</div>`;

  // გადაწყვეტილება — ადგილზე (გვერდი თავიდან არ იხატება → სქროლი არ ირევა)
  async function setDecision(spk, decision) {
    try { await post("/api/quality/decision", { spk, decision }); }
    catch (e) { window.UI?.toast(e.message, "error"); return; }
    const r = data.speakers.find((x) => x.spk === spk);
    if (!r) return;
    r.decision = decision;
    r.final = decision === "auto" ? r.suggest : decision;
    document.querySelectorAll(`[data-seg="${spk}"]`).forEach((seg) => {
      seg.outerHTML = decSeg(r);
    });
    document.querySelectorAll(`tr[data-spk="${spk}"]`).forEach((tr) => tr.classList.toggle("q-ex", r.final === "exclude"));
    bindSeg(spk);
  }
  function bindSeg(spk) {
    document.querySelectorAll(`[data-seg="${spk}"] [data-dec]`).forEach((b) =>
      b.addEventListener("click", () => setDecision(spk, b.dataset.dec)));
  }

  // გვერდითი პანელი: დინამიკის ჩანაწერები
  function drawer() {
    let d = document.getElementById("q-drawer");
    if (!d) {
      d = document.createElement("aside");
      d.id = "q-drawer";
      d.className = "q-drawer";
      d.setAttribute("aria-label", "დინამიკის ჩანაწერები");
      document.body.appendChild(d);
      document.addEventListener("keydown", (e) => { if (e.key === "Escape" && d.classList.contains("open")) closeDrawer(); });
    }
    return d;
  }
  function closeDrawer() {
    drawer().classList.remove("open");
    audio.pause();
    st.open = null;
    document.querySelectorAll("tr[data-spk] [data-open].on").forEach((b) => b.classList.remove("on"));
  }

  async function loadClips(spk) {
    const d0 = drawer();
    const r = data.speakers.find((x) => x.spk === spk) || {};
    d0.innerHTML = `<div class="q-drawer-head"><div><b class="mono">${spk}</b> ${r.current ? "" : `<span class="badge testing">ახალი</span>`}
        <div class="muted small">${r.f0 ? Math.round(r.f0) + " Hz · " : ""}კარგი ${r.good_min ?? "—"} / ${r.min ?? "—"} წთ · ცუდი ${pct(r.bad_share)}</div></div>
        <button class="ghost small" data-close aria-label="დახურვა">✕</button></div>
      <div class="q-drawer-dec">${decSeg(r)}</div>
      <div id="q-clips-${spk}" class="q-drawer-body small muted">იტვირთება…</div>`;
    d0.classList.add("open");
    d0.querySelector("[data-close]").onclick = closeDrawer;
    bindSeg(spk);
    const el = document.getElementById(`q-clips-${spk}`);
    const d = await api(`/api/quality/speaker/${spk}`);
    const show = d.clips.slice(0, 12).concat(d.clips.length > 12 ? d.clips.slice(-3) : []);
    el.innerHTML = `<div class="muted small">ყველაზე ცუდი 12 და საუკეთესო 3 (სულ ${d.clips.length})</div>` + show.map((c) => `
      <div class="q-clip">
        <button class="play" data-src="/api/quality/clip/${encodeURIComponent(c.id)}" aria-label="მოსმენა">${window.UI ? UI.icon("play", 12) : "▶"}</button>
        <div><div>${esc(c.text)}</div>
          ${c.heard ? `<div class="muted">MMS-მა გაიგონა: ${esc(c.heard)}</div>` : ""}
          <div class="small">${c.bad?.length ? `<span class="err">${c.bad.map((k) => esc(data.rules[k] || k)).join(", ")}</span> · ` : `<span class="ok">კარგი</span> · `}
            ${c.cer != null ? `MMS ${pct(Math.min(1, c.cer))}${c.text.length < 10 ? " (მოკლე — არასანდო)" : ""} · ` : ""}SNR ${c.snr ?? "—"} · შიშინი ${c.hiss ?? "—"} · ${c.dur ?? "—"} წმ${c.kind === "other" ? " · შეუმოწმებელი CV-ში" : ""}</div></div>
      </div>`).join("");
    el.querySelectorAll("[data-src]").forEach((b) => b.addEventListener("click", () => {
      el.querySelectorAll(".play.on").forEach((x) => x.classList.remove("on"));
      b.classList.add("on"); audio.src = b.dataset.src; audio.play();
    }));
  }

  function bind(box) {
    $("#q-run")?.addEventListener("click", async () => {
      try { await post("/api/quality/run"); window.UI?.toast("შემოწმება დაიწყო — რამდენიმე საათი გაგრძელდება", "ok"); setTimeout(render, 2000); }
      catch (e) { window.UI ? UI.toast(e.message, "error") : alert(e.message); }
    });
    $("#q-stop")?.addEventListener("click", async () => {
      await post("/api/quality/stop").catch(() => {});
      window.UI?.toast("ჩერდება — შემდეგ ჯერზე იქიდან გააგრძელებს", "info"); setTimeout(render, 2000);
    });
    const re = () => { st.limit = 60; ear.i = 0; render(); };
    $("#q-show")?.addEventListener("change", (e) => { st.show = e.target.value; re(); });
    $("#q-pitch")?.addEventListener("change", (e) => { st.pitch = e.target.value; re(); });
    $("#q-sort")?.addEventListener("change", (e) => { st.sort = e.target.value; re(); });
    $("#q-dir")?.addEventListener("click", () => { st.dir = -st.dir; re(); });
    bindEar();
    box.querySelectorAll("th[data-sort]").forEach((h) => h.addEventListener("click", () => {
      if (st.sort === h.dataset.sort) st.dir = -st.dir; else { st.sort = h.dataset.sort; st.dir = 1; }
      re();
    }));
    $("#q-more")?.addEventListener("click", () => { st.limit += 60; render(); });
    box.querySelectorAll("tr[data-spk]").forEach((tr) => {
      const spk = tr.dataset.spk;
      tr.querySelectorAll("[data-dec]").forEach((b) => b.addEventListener("click", () => setDecision(spk, b.dataset.dec)));
      tr.querySelector("[data-open]").addEventListener("click", (e) => {
        document.querySelectorAll("tr[data-spk] [data-open].on").forEach((x) => x.classList.remove("on"));
        if (st.open === spk) { closeDrawer(); return; }
        st.open = spk; e.currentTarget.classList.add("on"); loadClips(spk);
      });
    });
  }

  // ჩანართის გახსნისას
  document.querySelector('.tabs button[data-tab="data"]')?.addEventListener("click", () => setTimeout(render, 50));
  document.addEventListener("subtab", (e) => { if (e.detail.page === "data" && e.detail.key === "quality") render(); });
  if (!$("#tab-data")?.hidden) render();
})();
