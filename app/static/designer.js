// ხმის შემქმნელი — ჩანართი „ხმის შემქმნელი“ (backend: app/designer.py)
window.Designer = (() => {
  const $ = (s) => document.querySelector(s);
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const api = async (url, opts) => {
    const r = await fetch(url, opts);
    const body = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(body.detail || r.statusText);
    return body;
  };
  const PHRASES = [
    "გამარჯობა! ეს ახალი, სინთეზური ხმაა, რომელიც არავის ეკუთვნის.",
    "ოდესღაც, ძალიან დიდი ხნის წინ, ერთ პატარა სოფელში ცხოვრობდა მოხუცი მეთევზე.",
    "შენ აქ რას აკეთებ? ხომ გითხარი, არ მოსულიყავი!",
    "ზღვის ღრმა ტალღებმა ნავი ნაპირისკენ გააქანა.",
  ];

  let speakers = [], byName = {}, mix = {}, meta = null, inited = false, last = null;
  let audio = null;
  // ყველა ხმის მოსმენა (audition): რიგი, მიმდინარე ინდექსი, შეფასებები
  let scan = {};  // tools/gh_scan.py: MMS-ის სიზუსტე და „ღ“
  const ghN = (x) => x.gh_count ?? x.gh_hit;  // „ღ“-ების რაოდენობა MMS-ის ჩანაწერში (6-დან)
  const scanScore = (n) => { const x = scan[n]; return x ? ghN(x) / 6 * 2 - x.cer : -9; };
  const scanLabel = (n) => { const x = scan[n]; return x ? `ღ ${ghN(x)}/6 · შეცდომა ${Math.round(x.cer * 100)}%` : ""; };
  let ratings = {}, aud = null, au = { queue: [], i: 0, playing: false, phrase: 1, order: "f0", skipBad: true, cache: {} };

  const f0color = (f) => {
    if (!f) return "var(--muted)";
    const t = Math.max(0, Math.min(1, (f - 90) / 150)); // 90 Hz → ლურჯი, 240 Hz → ნარინჯისფერი
    return `hsl(${220 - t * 190} 70% 52%)`;
  };
  const label = (n) => {
    const s = byName[n] || {};
    return `${s.f0 ? Math.round(s.f0) + " Hz" : "—"} · ${s.minutes ? Math.round(s.minutes) + " წთ" : ""}`;
  };

  async function init() {
    inited = true;
    const st = await api("/api/designer/status");
    if (!st.ready) {
      $("#designer").innerHTML = `<div class="card"><h2>ხმის შემქმნელი</h2><div class="empty">საბაზო მოდელი ჯერ არ არის მზად შემქმნელისთვის.</div></div>`;
      return;
    }
    const d = await api("/api/designer/speakers");
    speakers = d.speakers; meta = d.meta;
    byName = Object.fromEntries(speakers.map((s) => [s.name, s]));
    $("#designer").innerHTML = `
      <div class="card">
        <div class="card-head"><h2>ხმის შემქმნელი</h2>
          <span class="muted small">საბაზო მოდელი · ეპოქა ${meta.epoch != null ? meta.epoch + 1 : "—"} (შუალედური — ხარისხი ყოველ სესიაზე უმჯობესდება)</span></div>
        <p class="muted">აირჩიეთ რამდენიმე ხმა რუკაზე ან მზა ვარიანტით, ძრავებით დაალაგეთ წილები და მოუსმინეთ.
          ნარევი ახალი ხმაა — არცერთ რეალურ დინამიკს არ ეკუთვნის.</p>
        <div class="des-grid">
          <div>
            <div class="lbl">ხმების რუკა <span class="muted small">— მარცხნივ დაბალი ხმები, მარჯვნივ მაღალი; ზემოთ-ქვემოთ — ჟღერადობის სხვა თვისება</span></div>
            <div class="des-map-wrap">
              <svg id="des-map" viewBox="0 0 100 100" preserveAspectRatio="xMidYMid meet" role="img" aria-label="დინამიკების რუკა"></svg>
              <div class="des-zoom"><button class="ghost small" id="des-zin" aria-label="გადიდება">+</button><button class="ghost small" id="des-zout" aria-label="დაპატარავება">−</button><button class="ghost small" id="des-zreset" aria-label="მთლიანი რუკა">⤢</button></div>
              <div class="des-axis"><span>← დაბალი</span><span>სიმაღლე</span><span>მაღალი →</span></div>
            </div>
            <p class="muted small" id="des-hover">დააჭირეთ წერტილს — დაემატება ან მოიხსნება ნარევიდან. ბორბალი — გადიდება, გადათრევა — გადაადგილება.</p>
          </div>
          <div>
            <div class="lbl">მზა ნარევი</div>
            <div class="chips">
              <button class="chip" data-sug="neutral">ნეიტრალური</button>
              <button class="chip" data-sug="female">ქალი</button>
              <button class="chip" data-sug="male">კაცი</button>
              <button class="chip" data-sug="random">შემთხვევითი</button>
              <button class="chip" id="des-clear">გასუფთავება</button>
            </div>
            <div class="lbl">ნარევი</div>
            <div id="des-mix"></div>
            <div class="sliders">
              <label class="slider"><span>ტემპი</span><input type="range" id="des-len" min="0.8" max="1.25" step="0.01" value="1" title="მეტი = უფრო სწრაფი"><output id="des-len-o">1.00</output></label>
              <label class="slider"><span>ინტონაცია</span><input type="range" id="des-noise" min="0.3" max="0.9" step="0.01" value="0.667"><output id="des-noise-o">0.67</output></label>
            </div>
          </div>
        </div>
        <label class="lbl" for="des-text">ტექსტი</label>
        <textarea id="des-text" rows="2">${esc(PHRASES[0])}</textarea>
        <div class="chips">${PHRASES.map((p, i) => `<button class="chip" data-ph="${i}">${esc(p.slice(0, 26))}…</button>`).join("")}</div>
        <div class="actions">
          <button class="primary" id="des-play">▶ მოსმენა</button>
          <span class="muted small" id="des-info"></span>
        </div>
        <audio id="des-audio" controls preload="none"></audio>
        <div class="des-save">
          <input id="des-name" maxlength="60" placeholder="ხმის სახელი, მაგ. „ნეიტრალური 1“">
          <button class="ghost" id="des-save-btn">შენახვა</button>
        </div>
      </div>
      <div class="card" id="rank-card"></div>
      <div class="card" id="gh-card"></div>
      <div class="card" id="aud-card">
        <div class="card-head"><h2>ყველა ხმის მოსმენა</h2><span class="muted small" id="aud-count"></span></div>
        <p class="muted small">რუკის ყველა ხმა ერთსა და იმავე ფრაზას ზედიზედ კითხულობს, თავისით გადადის შემდეგზე.
          მონიშნეთ, ვინ ამბობს კარგად — 👍 ან 👎. კლავიშები: <kbd>Space</kbd> პაუზა, <kbd>→</kbd> შემდეგი, <kbd>←</kbd> წინა,
          <kbd>L</kbd> მომწონს, <kbd>D</kbd> არ მომწონს. ტემპი და ინტონაცია ზემოთა ძრავებიდან მოდის.</p>
        <div class="aud-opts">
          <label>ფრაზა <select id="aud-phrase">${PHRASES.map((p, i) => `<option value="${i}" ${i === 1 ? "selected" : ""}>${i + 1}. ${esc(p.slice(0, 40))}…</option>`).join("")}</select></label>
          <label>რიგი <select id="aud-order"><option value="f0">სიმაღლით (დაბალი → მაღალი)</option><option value="map">რუკის მიხედვით</option><option value="minutes">ჩანაწერის მოცულობით</option><option value="random">შემთხვევით</option><option value="auto">ავტომატური რეიტინგით (საუკეთესო პირველი)</option></select></label>
          <label class="check"><input type="checkbox" id="aud-skip" checked> 👎-ები გამოტოვე</label>
        </div>
        <div class="aud-player">
          <button class="ghost" id="aud-start" title="სიის დასაწყისიდან">⏮ თავიდან</button>
          <button class="ghost" id="aud-prev" title="წინა (←)">◀</button>
          <button class="primary" id="aud-play" title="დაკვრა / პაუზა (Space)">▶ დაწყება</button>
          <button class="ghost" id="aud-next" title="შემდეგი (→)">▶▶</button>
          <div class="aud-now" id="aud-now">—</div>
          <button class="ghost aud-rate" id="aud-like" title="მომწონს (L)">👍</button>
          <button class="ghost aud-rate" id="aud-dislike" title="არ მომწონს (D)">👎</button>
        </div>
        <div class="bar"><span id="aud-bar" style="width:0%"></span></div>
        <div class="lbl">მოწონებული ხმები <button class="ghost small" id="aud-mix-all">ყველა ნარევში →</button></div>
        <div id="aud-liked"></div>
      </div>
      <div class="card">
        <h2>შენახული ხმები</h2>
        <div id="des-list"></div>
      </div>`;
    audio = $("#des-audio");
    aud = new Audio();
    ratings = await api("/api/designer/ratings").catch(() => ({}));
    scan = await api("/api/designer/scan").catch(() => ({}));
    bindAudition();
    renderGh();
    renderRank();
    bindZoom();
    drawMap();
    renderMix();
    loadDesigns();
    $("#designer").querySelectorAll("[data-sug]").forEach((b) => b.addEventListener("click", async () => {
      mix = (await api(`/api/designer/suggest?mode=${b.dataset.sug}`)).weights;
      Object.keys(mix).forEach((k) => (mix[k] = Math.round(mix[k] * 100)));
      renderMix(); drawMap(); play();
    }));
    $("#des-clear").onclick = () => { mix = {}; renderMix(); drawMap(); };
    $("#designer").querySelectorAll("[data-ph]").forEach((b) => b.addEventListener("click", () => { $("#des-text").value = PHRASES[b.dataset.ph]; play(); }));
    for (const id of ["len", "noise"]) $(`#des-${id}`).addEventListener("input", (e) => ($(`#des-${id}-o`).textContent = (+e.target.value).toFixed(2)));
    for (const id of ["len", "noise"]) $(`#des-${id}`).addEventListener("change", play);
    $("#des-play").onclick = play;
    $("#des-save-btn").onclick = save;
  }

  function drawMap() {
    const svg = $("#des-map");
    if (!svg) return;
    const k = Math.min(1, Math.max(0.45, view.w / 100)); // გადიდებისას წერტილები პროპორციულად არ იზრდება
    const cur = au.queue[au.i];
    const dot = (s) => {
      const on = mix[s.name] != null, now = au.playing && s.name === cur, rt = ratings[s.name]?.rating;
      const stroke = now ? `stroke="var(--accent)" stroke-width="${(1 * k).toFixed(2)}"`
        : on ? `stroke="var(--text)" stroke-width="${(0.7 * k).toFixed(2)}"`
        : rt === 1 ? `stroke="#2e9e5b" stroke-width="${(0.6 * k).toFixed(2)}"` : rt === -1 ? 'fill-opacity="0.25"' : 'fill-opacity="0.8"';
      return `<circle cx="${(4 + s.x * 92).toFixed(2)}" cy="${(4 + (1 - s.y) * 92).toFixed(2)}" r="${((now ? 3 : on ? 2.3 : 1.75) * k).toFixed(2)}"
        fill="${f0color(s.f0)}" ${stroke} data-n="${esc(s.name)}"><title>${esc(s.name)} · ${esc(label(s.name))}</title></circle>`;
    };
    // არჩეულები ბოლოს — ზემოდან ჩანს
    const top = (s) => mix[s.name] != null || (au.playing && s.name === cur);
    svg.innerHTML = speakers.filter((s) => !top(s)).map(dot).join("") + speakers.filter(top).map(dot).join("");
    svg.setAttribute("viewBox", `${view.x} ${view.y} ${view.w} ${view.w}`);
    svg.querySelectorAll("circle").forEach((c) => {
      c.addEventListener("click", () => {
        if (dragged) return;
        const n = c.dataset.n;
        if (mix[n] != null) delete mix[n]; else mix[n] = 50;
        renderMix(); drawMap();
      });
      c.addEventListener("mouseenter", () => ($("#des-hover").textContent = `${c.dataset.n} · ${label(c.dataset.n)}`));
    });
  }

  // გადიდება / გადაადგილება
  let view = { x: 0, y: 0, w: 100 }, dragged = false;
  function setView(w, cx, cy) {
    w = Math.max(20, Math.min(100, w));
    view = { w, x: Math.max(0, Math.min(100 - w, cx - w / 2)), y: Math.max(0, Math.min(100 - w, cy - w / 2)) };
    drawMap();
  }
  function bindZoom() {
    const svg = $("#des-map");
    const toMap = (e) => { const r = svg.getBoundingClientRect(); return [view.x + (e.clientX - r.left) / r.width * view.w, view.y + (e.clientY - r.top) / r.height * view.w]; };
    svg.addEventListener("wheel", (e) => {
      e.preventDefault();
      const [mx, my] = toMap(e), f = e.deltaY > 0 ? 1.25 : 0.8, w = Math.max(20, Math.min(100, view.w * f));
      // კურსორის ქვეშ მყოფი წერტილი ადგილზე რჩება
      setView(w, mx - (mx - view.x) * w / view.w + w / 2, my - (my - view.y) * w / view.w + w / 2);
    }, { passive: false });
    let start = null;
    svg.addEventListener("pointerdown", (e) => { start = { cx: e.clientX, cy: e.clientY, v: { ...view } }; dragged = false; });
    window.addEventListener("pointermove", (e) => {
      if (!start || view.w >= 100) return;
      const r = svg.getBoundingClientRect(), dx = (e.clientX - start.cx) / r.width * view.w, dy = (e.clientY - start.cy) / r.height * view.w;
      if (Math.abs(e.clientX - start.cx) + Math.abs(e.clientY - start.cy) > 4) dragged = true;
      if (dragged) setView(view.w, start.v.x - dx + view.w / 2, start.v.y - dy + view.w / 2);
    });
    window.addEventListener("pointerup", () => { start = null; setTimeout(() => (dragged = false), 0); });
    $("#des-zin").onclick = () => setView(view.w * 0.7, view.x + view.w / 2, view.y + view.w / 2);
    $("#des-zout").onclick = () => setView(view.w / 0.7, view.x + view.w / 2, view.y + view.w / 2);
    $("#des-zreset").onclick = () => setView(100, 50, 50);
  }

  function renderMix() {
    const names = Object.keys(mix);
    const tot = names.reduce((a, n) => a + (+mix[n] || 0), 0) || 1;
    $("#des-mix").innerHTML = names.length ? names.map((n) => `
      <div class="mix-row">
        <span class="dotc" style="background:${f0color(byName[n]?.f0)}"></span>
        <span class="small mix-name" title="${esc(n)}">${esc(label(n))}</span>
        <input type="range" min="0" max="100" value="${mix[n]}" data-n="${esc(n)}">
        <output class="small">${Math.round((+mix[n] || 0) / tot * 100)}%</output>
        <button class="x" data-rm="${esc(n)}" aria-label="მოხსნა">×</button>
      </div>`).join("") : `<div class="empty small">ნარევი ცარიელია — აირჩიეთ მზა ვარიანტი ან დააჭირეთ რუკას.</div>`;
    $("#des-mix").querySelectorAll("input[type=range]").forEach((r) => r.addEventListener("change", () => { mix[r.dataset.n] = +r.value; renderMix(); play(); }));
    $("#des-mix").querySelectorAll("[data-rm]").forEach((b) => b.addEventListener("click", () => { delete mix[b.dataset.rm]; renderMix(); drawMap(); }));
  }

  const lenScale = () => +(1 / +$("#des-len").value).toFixed(3);

  function body(extra = {}) {
    return JSON.stringify({ weights: mix, text: $("#des-text").value, length_scale: lenScale(), noise_scale: +$("#des-noise").value, ...extra });
  }

  function uniqueness(n) {
    const s = n?.[0]?.sim ?? 0;
    return s >= 0.9 ? `<span class="verdict bad">ძალიან ჰგავს ${esc(n[0].name)}-ს (${s})</span>`
      : s >= 0.8 ? `<span class="verdict maybe">ჰგავს ${esc(n[0].name)}-ს (${s})</span>`
      : `<span class="verdict good">უნიკალური — ყველაზე ახლოს ${s}</span>`;
  }

  async function play() {
    if (!Object.keys(mix).length) return;
    $("#des-info").innerHTML = "გენერირდება…";
    $("#des-play").disabled = true;
    try {
      const r = await api("/api/designer/synth", { method: "POST", headers: { "Content-Type": "application/json" }, body: body() });
      last = r;
      audio.src = r.url; audio.play().catch(() => {});
      $("#des-info").innerHTML = `${r.f0_estimate ? `სიმაღლე ≈ ${r.f0_estimate} Hz · ` : ""}${uniqueness(r.nearest)}`;
    } catch (e) { $("#des-info").textContent = "შეცდომა: " + e.message; }
    finally { $("#des-play").disabled = false; }
  }

  async function save() {
    const name = $("#des-name").value.trim();
    if (!name) { $("#des-name").focus(); return; }
    if (!Object.keys(mix).length) return;
    try {
      await api("/api/designer/designs", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, weights: mix, length_scale: lenScale(), noise_scale: +$("#des-noise").value }) });
      $("#des-name").value = "";
      loadDesigns();
    } catch (e) { alert(e.message); }
  }

  async function loadDesigns() {
    const list = await api("/api/designer/designs");
    $("#des-list").innerHTML = list.length ? list.map((d) => `
      <div class="des-item">
        <div><b>${esc(d.name)}</b> <span class="muted small">${d.f0_estimate ? `≈ ${d.f0_estimate} Hz · ` : ""}${Object.keys(d.weights).length} ხმის ნარევი ·
          შექმნილია ${esc((d.created || "").slice(0, 16).replace("T", " "))} · ეპოქა ${d.model_epoch != null ? d.model_epoch + 1 : "—"}</span></div>
        <div class="des-actions">
          <button class="ghost small" data-load="${esc(d.id)}">ჩატვირთვა</button>
          <button class="ghost small" data-del="${esc(d.id)}">წაშლა</button>
        </div>
      </div>`).join("") : `<div class="empty">ჯერ არცერთი ხმა არ არის შენახული.</div>`;
    $("#des-list").querySelectorAll("[data-load]").forEach((b) => b.addEventListener("click", () => {
      const d = list.find((x) => x.id === b.dataset.load);
      mix = Object.fromEntries(Object.entries(d.weights).map(([k, v]) => [k, Math.round(v * 100)]));
      $("#des-len").value = (1 / d.length_scale).toFixed(2); $("#des-noise").value = d.noise_scale;
      $("#des-len-o").textContent = (1 / d.length_scale).toFixed(2); $("#des-noise-o").textContent = (+d.noise_scale).toFixed(2);
      renderMix(); drawMap(); play();
      window.scrollTo({ top: 0, behavior: "smooth" });
    }));
    $("#des-list").querySelectorAll("[data-del]").forEach((b) => b.addEventListener("click", async () => {
      if (!confirm("წავშალო ეს ხმა?")) return;
      await api(`/api/designer/designs/${b.dataset.del}`, { method: "DELETE" });
      loadDesigns();
    }));
  }

  // ---------- ყველა ხმის მოსმენა ----------
  function buildQueue() {
    let q = speakers.slice();
    if (au.order === "f0") q.sort((a, b) => (a.f0 || 0) - (b.f0 || 0));
    else if (au.order === "map") q.sort((a, b) => a.x - b.x || a.y - b.y);
    else if (au.order === "minutes") q.sort((a, b) => (b.minutes || 0) - (a.minutes || 0));
    else if (au.order === "auto") q.sort((a, b) => scanScore(b.name) - scanScore(a.name));
    else q.sort(() => Math.random() - 0.5);
    au.queue = q.map((s) => s.name);
  }
  const audBody = (n) => JSON.stringify({ weights: { [n]: 1 }, text: PHRASES[au.phrase], length_scale: lenScale(), noise_scale: +$("#des-noise").value });
  function synthOne(n) {
    const key = `${n}|${au.phrase}|${lenScale()}|${$("#des-noise").value}`;
    return (au.cache[key] ||= api("/api/designer/synth", { method: "POST", headers: { "Content-Type": "application/json" }, body: audBody(n) })
      .catch((e) => { delete au.cache[key]; throw e; }));
  }
  const skippable = (n) => au.skipBad && ratings[n]?.rating === -1;
  function nextIndex(from, dir) {
    let i = from;
    do { i += dir; } while (i >= 0 && i < au.queue.length && skippable(au.queue[i]));
    return i;
  }
  async function audPlay(i) {
    if (i < 0 || i >= au.queue.length) { audStop(); return; }
    au.i = i; au.playing = true;
    const n = au.queue[i];
    renderNow("გენერირდება…");
    drawMap();
    try {
      const r = await synthOne(n);
      if (!au.playing || au.queue[au.i] !== n) return;  // სანამ მზადდებოდა, სხვაზე გადავიდნენ
      aud.src = r.url;
      await aud.play().catch(() => {});
      renderNow();
      const j = nextIndex(i, 1);
      if (j < au.queue.length) synthOne(au.queue[j]).catch(() => {});  // შემდეგის წინასწარ მომზადება
    } catch (e) { renderNow("შეცდომა: " + e.message); }
  }
  function audRestart() {  // სიის დასაწყისი: თუ უკრავდა — პირველი (გამოუტოვებელი) ხმიდან გააგრძელებს
    const was = au.playing;
    aud.pause(); au.playing = false; au.i = 0;
    $("#aud-play").textContent = "▶ დაწყება";
    if (was) audPlay(skippable(au.queue[0]) ? nextIndex(0, 1) : 0); else { renderNow(); drawMap(); }
  }
  function audStop() {
    au.playing = false; aud.pause();
    $("#aud-play").textContent = "▶ გაგრძელება";
    renderNow(); drawMap();
  }
  function renderNow(msg) {
    const n = au.queue[au.i];
    const s = byName[n] || {};
    const rt = ratings[n]?.rating;
    $("#aud-now").innerHTML = n ? `<b>${au.i + 1} / ${au.queue.length}</b> · ${esc(label(n))} <span class="muted small">${esc(n)}</span>
      ${scan[n] ? `<span class="small"> · ${esc(scanLabel(n))}</span>` : ""}
      ${rt === 1 ? "<span class=\"verdict good\">👍</span>" : rt === -1 ? "<span class=\"verdict bad\">👎</span>" : ""}
      ${msg ? `<span class="muted small"> · ${esc(msg)}</span>` : ""}` : "—";
    $("#aud-bar").style.width = `${au.queue.length ? ((au.i + 1) / au.queue.length) * 100 : 0}%`;
    if (au.playing) $("#aud-play").textContent = "⏸ პაუზა";
    const liked = Object.keys(ratings).filter((k) => ratings[k].rating === 1).length;
    const bad = Object.keys(ratings).filter((k) => ratings[k].rating === -1).length;
    $("#aud-count").textContent = `👍 ${liked} · 👎 ${bad} · შეუფასებელი ${speakers.length - liked - bad}`;
  }
  async function rateCur(v) {
    const n = au.queue[au.i];
    if (!n) return;
    const nv = ratings[n]?.rating === v ? 0 : v;  // იგივეზე მეორე დაჭერა — მოხსნა
    if (nv) ratings[n] = { rating: nv, phrase: au.phrase }; else delete ratings[n];
    renderNow(); renderLiked(); drawMap();
    api("/api/designer/ratings", { method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: n, rating: nv, phrase: au.phrase }) }).catch((e) => alert("შეფასება ვერ შეინახა: " + e.message));
    if (nv === -1 && au.playing) audPlay(nextIndex(au.i, 1));  // 👎 — მაშინვე შემდეგზე
  }
  function renderLiked() {
    const liked = speakers.filter((s) => ratings[s.name]?.rating === 1).sort((a, b) => (a.f0 || 0) - (b.f0 || 0));
    $("#aud-liked").innerHTML = liked.length ? `<div class="aud-liked">${liked.map((s) => `
      <span class="aud-chip"><span class="dotc" style="background:${f0color(s.f0)}"></span>${esc(label(s.name))}
        <button class="x" data-aplay="${esc(s.name)}" title="მოსმენა">▶</button>
        <button class="x" data-amix="${esc(s.name)}" title="ნარევში დამატება">+</button></span>`).join("")}</div>`
      : `<div class="empty small">ჯერ არაფერია მონიშნული.</div>`;
    $("#aud-liked").querySelectorAll("[data-aplay]").forEach((b) => b.addEventListener("click", () => {
      const i = au.queue.indexOf(b.dataset.aplay); if (i >= 0) audPlay(i);
    }));
    $("#aud-liked").querySelectorAll("[data-amix]").forEach((b) => b.addEventListener("click", () => {
      mix[b.dataset.amix] = 50; renderMix(); drawMap();
    }));
  }
  function bindAudition() {
    buildQueue(); renderNow(); renderLiked();
    aud.addEventListener("ended", () => { if (au.playing) setTimeout(() => au.playing && audPlay(nextIndex(au.i, 1)), 500); });
    $("#aud-play").onclick = () => (au.playing ? audStop() : audPlay(au.queue[au.i] && !skippable(au.queue[au.i]) ? au.i : nextIndex(au.i, 1)));
    $("#aud-next").onclick = () => audPlay(nextIndex(au.i, 1));
    $("#aud-start").onclick = audRestart;
    $("#aud-prev").onclick = () => audPlay(Math.max(0, nextIndex(au.i, -1)));
    $("#aud-like").onclick = () => rateCur(1);
    $("#aud-dislike").onclick = () => rateCur(-1);
    $("#aud-phrase").onchange = (e) => { au.phrase = +e.target.value; if (au.playing) audPlay(au.i); };
    $("#aud-order").onchange = async (e) => {  // ახალი რიგი — ყოველთვის თავიდან
      if (e.target.value === "auto") scan = await api("/api/designer/scan").catch(() => scan);
      au.order = e.target.value; buildQueue(); audRestart(); };
    $("#aud-skip").onchange = (e) => (au.skipBad = e.target.checked);
    $("#aud-mix-all").onclick = () => {
      speakers.filter((s) => ratings[s.name]?.rating === 1).forEach((s) => (mix[s.name] = 50));
      renderMix(); drawMap(); play(); window.scrollTo({ top: 0, behavior: "smooth" });
    };
    document.addEventListener("keydown", (e) => {
      if ($("#tab-designer")?.hidden || /INPUT|TEXTAREA|SELECT/.test(document.activeElement?.tagName)) return;
      if (e.code === "Space") { e.preventDefault(); $("#aud-play").click(); }
      else if (e.key === "ArrowRight") $("#aud-next").click();
      else if (e.key === "ArrowLeft") $("#aud-prev").click();
      else if (e.key.toLowerCase() === "l") rateCur(1);
      else if (e.key.toLowerCase() === "d") rateCur(-1);
    });
  }

  // ---------- რეიტინგი (tools/mms_rank.py): ყველა ხმა და ნარევი, 5 წინადადება, საუკეთესოდან უარესისკენ ----------
  let rk = { data: null, lo: 0, hi: 400, kind: "all", show: 40, sent: 0 };
  async function renderRank() {
    const box = $("#rank-card");
    rk.data = await api("/api/designer/rank").catch(() => null);
    if (!rk.data || !rk.data.rows.length) { box.hidden = true; return; }
    box.hidden = false;
    const S = rk.data.sentences;
    const rows = rk.data.rows.filter((r) => (rk.kind === "all" || r.kind === rk.kind)
      && (r.kind === "mix" || ((r.f0 || 0) >= rk.lo && (r.f0 || 0) <= rk.hi)));
    const pct = (x) => Math.round(x * 100);
    const cls = (x) => (x <= 0.12 ? "ok" : x <= 0.25 ? "" : "bad");
    box.innerHTML = `
      <div class="card-head"><h2>რეიტინგი — საუკეთესოდან უარესისკენ</h2><span class="muted small">${rk.data.total} შეფასებული (ხმები + ნარევები)</span></div>
      <p class="muted small">თითოეულმა 5 წინადადება წაიკითხა; ავტომატურმა მსმენელმა (MMS) ჩაიწერა, რაც გაიგონა. რიცხვი — შეცდომა (%): რაც ნაკლებია, მით სუფთად და გასაგებად ლაპარაკობს.
        „ღ“-ს ის კარგად ვერ აფასებს — ეს ბოლოს ყურით. ▶ უკრავს არჩეულ წინადადებას; სვეტის რიცხვზე მიტანისას ჩანს, რა „გაიგონა“.</p>
      <div class="aud-opts">
        <label>ჩვენება <select id="rk-kind"><option value="all">ყველა</option><option value="speaker">მხოლოდ ხმები</option><option value="mix">მხოლოდ ნარევები</option></select></label>
        <label>სიმაღლე <input type="number" id="rk-lo" value="${rk.lo}" style="width:70px"> – <input type="number" id="rk-hi" value="${rk.hi}" style="width:70px"> Hz</label>
        <label>▶ წინადადება <select id="rk-sent">${S.map((t, i) => `<option value="${i}" ${i === rk.sent ? "selected" : ""}>${i + 1}. ${esc(t.slice(0, 30))}…</option>`).join("")}</select></label>
      </div>
      <div class="cmp-wrap"><table class="cmp rank-table">
        <thead><tr><th>#</th><th></th><th>ხმა / ნარევი</th><th>საშუალო</th>${S.map((t, i) => `<th title="${esc(t)}">${i + 1}</th>`).join("")}<th></th></tr></thead>
        <tbody>${rows.slice(0, rk.show).map((r, i) => `<tr class="${r.kind === "mix" ? "mixrow" : ""}">
          <td>${i + 1}</td>
          <td><button class="play" data-rk="${esc(r.kind)}|${esc(r.name)}">▶</button></td>
          <td class="small">${r.kind === "mix" ? `<b>${esc(r.name)}</b>` : `${r.f0 ? Math.round(r.f0) + " Hz" : ""} <span class="muted">${esc(r.name)}</span>
            ${ratings[r.name]?.rating === 1 ? "👍" : ratings[r.name]?.rating === -1 ? "👎" : ""}`}</td>
          <td class="${cls(r.cer)}"><b>${pct(r.cer)}%</b></td>
          ${r.per.map((x, j) => `<td class="small ${cls(x)}" title="${esc(r.texts[j] || "")}">${pct(x)}</td>`).join("")}
          <td>${r.kind === "speaker" ? `<button class="x" data-rkmix="${esc(r.name)}" title="ნარევში დამატება">+</button>` : ""}</td>
        </tr>`).join("")}</tbody></table></div>
      ${rows.length > rk.show ? `<div class="actions"><button class="ghost small" id="rk-more">კიდევ 40 (${rows.length - rk.show} დარჩა)</button></div>` : ""}`;
    $("#rk-kind").value = rk.kind;
    $("#rk-kind").onchange = (e) => { rk.kind = e.target.value; renderRank(); };
    $("#rk-lo").onchange = (e) => { rk.lo = +e.target.value || 0; renderRank(); };
    $("#rk-hi").onchange = (e) => { rk.hi = +e.target.value || 400; renderRank(); };
    $("#rk-sent").onchange = (e) => { rk.sent = +e.target.value; };
    const more = $("#rk-more"); if (more) more.onclick = () => { rk.show += 40; renderRank(); };
    box.querySelectorAll("[data-rk]").forEach((b) => b.addEventListener("click", async () => {
      const [kind, name] = b.dataset.rk.split("|");
      const weights = kind === "mix" ? (await api("/api/designer/designs")).find((d) => d.name === name)?.weights : { [name]: 1 };
      if (!weights) return;
      box.querySelectorAll(".play.on").forEach((x) => x.classList.remove("on")); b.classList.add("on");
      const r = await api("/api/designer/synth", { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ weights, text: S[rk.sent], length_scale: lenScale(), noise_scale: +$("#des-noise").value }) });
      aud.src = r.url; aud.play();
    }));
    box.querySelectorAll("[data-rkmix]").forEach((b) => b.addEventListener("click", () => { mix[b.dataset.rkmix] = 50; renderMix(); drawMap(); }));
  }

  // ---------- „ღ“-ს ბრმა ტესტი: ჯერ ყური, მერე ავტომატური ამომცნობების პასუხი ----------
  let gh = { answers: {} };
  const GH_LEVELS = [[3, "საუკეთესო"], [2, "კარგი"], [1, "საშუალო"], [0, "ცუდი"]];
  const lvName = (v) => (GH_LEVELS.find(([l]) => l === v) || [0, "—"])[1];
  async function renderGh() {
    const box = $("#gh-card");
    const d = await api("/api/designer/ghtest").catch(() => null);
    if (!d || !d.clips.length) { box.hidden = true; return; }
    box.hidden = false;
    if (!gh.loaded) { gh.answers = { ...(d.answers || {}) }; gh.loaded = true; }  // სერვერიდან მხოლოდ პირველად
    const done = Object.keys(gh.answers).length;
    const head = `<div class="card-head"><h2>„ღ“-ს ბრმა ტესტი</h2><span class="muted small">${done} / ${d.clips.length} შეფასებული</span></div>
      <p class="muted small">20 ხმა ამბობს: <b>„${esc(d.text || "")}“</b> — აქ 4 „ღ“-ია (დღეს, ზღვაზე, ტალღები, აღარ). ნომრები შემთხვევითია — ვინ არის, არ ჩანს.
        მოუსმინეთ და შეაფასეთ „ღ“-ები: საუკეთესო, კარგი, საშუალო თუ ცუდი. ბოლოს „შედარება“ გაჩვენებთ, რამდენად ემთხვევა თქვენს ყურს ავტომატური მსმენელი (MMS).</p>`;
    if (d.compare) { box.innerHTML = head + ghCompare(d.compare); bindGhReset(); return; }
    box.innerHTML = head + `<div class="gh-grid">${d.clips.map((c) => {
        const v = gh.answers[c.id];
        return `<div class="gh-item ${v >= 2 ? "yes" : v != null ? "no" : ""}">
          <b>#${c.id}</b>
          <button class="play" data-src="${c.url}">▶</button>
          ${GH_LEVELS.map(([lv, t]) => `<button class="ghost small ${v === lv ? "on" : ""}" data-gh="${c.id}" data-v="${lv}">${t}</button>`).join("")}
        </div>`; }).join("")}</div>
      <div class="actions"><button class="primary" id="gh-submit" ${done < d.clips.length ? "disabled" : ""}>შედარება</button>
        <span class="muted small">${d.ready ? "" : "ავტომატური შეფასება ჯერ მიმდინარეობს — შედარება მის დასრულებას დაელოდება."}</span></div>`;
    box.querySelectorAll(".play").forEach((b) => b.addEventListener("click", () => { aud.src = b.dataset.src; aud.play(); }));
    box.querySelectorAll("[data-gh]").forEach((b) => b.addEventListener("click", () => {
      gh.answers[b.dataset.gh] = +b.dataset.v;
      ghSave(false);
      renderGh();
    }));
    const sub = $("#gh-submit");
    if (sub) sub.onclick = async () => {
      try {
        await ghSave(true);
        renderGh();
      } catch (e) { alert(e.message); }
    };
  }
  const ghSave = (final) => api("/api/designer/ghtest/answers", { method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ answers: gh.answers, final }) }).catch((e) => alert("ვერ შეინახა: " + e.message));
  function ghCompare(c) {
    const mark = (v) => (v === true ? "✓" : v === false ? "✗" : "—");
    return `<p><b>თანხვედრა თქვენს ყურთან:</b> ორ ჯგუფად — <b>${c.agree.mms}%</b>
        <span class="muted small">(თქვენი „საუკეთესო/კარგი“ ↔ MMS-ის 3–4 „ღ“)</span>
        · რიგის მსგავსება — <b>${c.rho ?? "—"}</b> <span class="muted small">(1 = სრული თანხვედრა, 0 = კავშირი არ არის)</span></p>
      <div class="cmp-wrap"><table class="cmp gh-table">
        <thead><tr><th>#</th><th>ხმა</th><th>თქვენ</th><th>MMS</th><th>რა დაწერა MMS-მა</th><th>MMS-ის ჯგუფი</th></tr></thead>
        <tbody>${c.rows.map((r) => `<tr>
          <td><button class="play" data-src="/api/designer/ghtest/audio/${String(r.id).padStart(2, "0")}.wav">#${r.id}</button></td>
          <td class="small">${r.f0 ? Math.round(r.f0) + " Hz" : ""} <span class="muted">${esc(r.speaker)}</span></td>
          <td>${r.user != null ? esc(lvName(r.user)) : "—"}</td>
          <td class="${r.user == null ? "" : (r.user >= 2) === r.mms_gh ? "ok" : "bad"}">ღ ${r.mms_count ?? "—"}/4</td>
          <td class="small">${esc(r.mms || "")}</td>
          <td class="small muted">${esc(r.group || "")}</td>
        </tr>`).join("")}</tbody></table></div>
      <div class="actions"><button class="ghost small" id="gh-reset">თავიდან შეფასება</button></div>`;
  }
  function bindGhReset() {
    $("#gh-card").querySelectorAll(".play").forEach((b) => b.addEventListener("click", () => { aud.src = b.dataset.src; aud.play(); }));
    $("#gh-reset").onclick = async () => { gh.answers = {}; await ghSave(false); renderGh(); };
  }

  return { render: () => { if (!inited) init().catch((e) => ($("#designer").innerHTML = `<div class="card"><div class="empty">შეცდომა: ${esc(e.message)}</div></div>`)); } };
})();
