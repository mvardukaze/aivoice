// „ხმის შემქმნელი → აღწერით“: მეორე ძრავი OmniVoice — ხმის შექმნა აღწერით (backend: app/omni.py, tools/omni_server.py)
window.Omni = (() => {
  const $ = (s) => document.querySelector(s);
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const api = async (url, opts) => {
    const r = await fetch(url, opts);
    const body = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(Array.isArray(body.detail) ? body.detail[0]?.msg : body.detail || r.statusText);
    return body;
  };
  const post = (url, data) => api(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });
  const PHRASE = "გამარჯობა! მე ახალი ქართული ხმა ვარ. დღეს ამინდი მშვენიერია, არა?";
  // instruct-ის მნიშვნელობა → „შენახვის“ ფორმის ნაგულისხმევი ასაკი
  const AGE_HINT = { child: "8–12", teenager: "13–18", "young adult": "20–30", "middle-aged": "35–55", elderly: "65+" };

  let opts = null, st = null, variants = [], running = false, stop = false, saving = null, timer = null, savedId = null;

  const box = () => $("#omni-card");

  function statusLine(s) {
    if (!s.running) return `<span class="verdict maybe">სერვერი გამორთულია</span>
      <span class="muted small">პირველ მოთხოვნაზე თავად ჩაირთვება (~15 წმ + მოდელის ჩატვირთვა ~10 წმ)</span>
      <button class="ghost small" id="om-start">ჩართვა</button>`;
    const v = s.server || {};
    return `<span class="verdict good">ეშვება</span>
      <span class="muted small">მოდელი: ${v.loaded ? `<b>ჩატვირთულია</b> · ${esc(v.device)}` : `არ არის ჩატვირთული (ჩაიტვირთება საჭიროებისას; გამოიტვირთება ${v.idle_unload_min} წთ უსაქმობის შემდეგ)`}
      · ${v.busy ? `<b>მუშაობს</b>${v.queue ? ` · რიგში ${v.queue}` : ""}` : "თავისუფალია"}${v.error ? ` · <span class="err">${esc(v.error)}</span>` : ""}</span>`;
  }

  async function refreshStatus() {
    clearTimeout(timer);
    if (!box() || UI.getSub("designer") !== "describe" || $("#tab-designer").hidden) return;
    st = await api("/api/omni/status").catch(() => st);
    const el = $("#om-status");
    if (el && st?.installed) { el.innerHTML = statusLine(st); bindStart(); }
    timer = setTimeout(refreshStatus, running ? 3000 : 8000);
  }

  function bindStart() {
    $("#om-start")?.addEventListener("click", async (e) => {
      e.target.disabled = true; e.target.textContent = "ირთვება…";
      try { await post("/api/omni/start", {}); } catch (err) { UI.toast(err.message, "error"); }
      refreshStatus();
    });
  }

  const sel = (k) => `<label class="lbl" for="om-${k}">${esc(opts[k].label)}</label>
    <select id="om-${k}">${Object.entries(opts[k].items).map(([v, l]) => `<option value="${esc(v)}">${esc(l)}</option>`).join("")}</select>`;

  async function render() {
    if (!box()) return;
    [st, opts] = await Promise.all([api("/api/omni/status"), opts ? opts : api("/api/omni/options")]);
    if (!st.installed) {
      box().innerHTML = `<div class="card-head"><h2>ხმა აღწერით</h2><span class="tag">არ არის დაყენებული</span></div>
        <p class="muted small help">მეორე ძრავი — <b>OmniVoice</b> (k2-fsa; კოდი Apache-2.0, წონები CC-BY-NC — მხოლოდ არაკომერციულად): ახალ ხმას ქმნის აღწერით (სქესი, ასაკი, სიმაღლე),
          წვრთნის გარეშე. შენახული ხმა ბიბლიოთეკაში ჩნდება და „მოსმენაში“ / „გრძელ ტექსტში“ მუშაობს, როგორც Piper-ის ხმები.</p>
        <p class="muted small">წვრთნით შექმნილი (კომერციულად გამოსაყენებელი) ხმისთვის — ჩანართი „ნარევით“.</p>
        <p class="small">დასაყენებლად (~6 GB: torch + მოდელი; სასურველია NVIDIA GPU ან Apple Silicon):</p>
        <pre class="tr-log">${esc(st.install_cmd)}</pre>
        <p class="muted small">შემდეგ გადატვირთეთ ეს გვერდი. CPU-ზეც მუშაობს, ოღონდ ძალიან ნელა.</p>`;
      return;
    }
    if (!st.enabled) {
      box().innerHTML = `<div class="card-head"><h2>ხმა აღწერით</h2><span class="tag">გამორთულია</span></div>
        <p class="muted small">ჩართვა: „პარამეტრები“ → OmniVoice → „OmniVoice ჩართულია“ → კი.</p>`;
      return;
    }
    box().innerHTML = `
      <div class="card-head"><h2>ხმა აღწერით</h2><span><span class="tag engine-omni">OmniVoice</span> <span class="tag nc" title="მოდელის ლიცენზია CC-BY-NC">არაკომერციული</span></span></div>
      <p class="muted small help">აღწერეთ ხმა, მოუსმინეთ რამდენიმე ვარიანტს და საუკეთესო შეინახეთ ბიბლიოთეკაში — წვრთნა არ სჭირდება, ტემბრი შენახვის შემდეგ ფიქსირებულია.
        OmniVoice-ის ხმები <b>მხოლოდ არაკომერციულადაა</b> (CC-BY-NC). კომერციული ხმისთვის — „ნარევით“ + წვრთნა.</p>
      ${UI.hint(`ვარიანტები ერთი აღწერით, სხვადასხვა „seed“-ით იქმნება — ყოველი ახალი ხმაა, არცერთ ადამიანს არ ეკუთვნის.
        ნამდვილი ადამიანის ხმის კლონირება (ჩანაწერის ატვირთვა) გამორთულია. GTX 1050 Ti-ზე ვარიანტი ~5-ჯერ ნელა იქმნება, ვიდრე მისი ხანგრძლივობა,
        შენახული ხმით გახმოვანება — ~10-ჯერ. ემოცია OmniVoice-ის ხმაზე ტემპსა და ხმამაღლობას ცვლის.`)}
      <div class="small om-status" id="om-status">${statusLine(st)}</div>
      <div class="om-grid">
        <div>${sel("gender")}</div><div>${sel("age")}</div><div>${sel("pitch")}</div><div>${sel("style")}</div>
      </div>
      <label class="lbl" for="om-text">სატესტო ფრაზა</label>
      <textarea id="om-text" rows="2" maxlength="400">${esc(PHRASE)}</textarea>
      <div class="actions action-bar">
        <label class="field-inline">ვარიანტები <input type="number" id="om-n" value="4" min="1" max="8"></label>
        <button class="primary" id="om-go">${UI.icon("sparkle", 14)} ვარიანტების შექმნა</button>
        <button class="ghost small" id="om-stop" hidden>გაჩერება</button>
        <span class="muted small" id="om-info"></span>
      </div>
      <div class="bar" id="om-bar" hidden><span style="width:0%"></span></div>
      <div id="om-variants" class="om-variants"></div>`;
    $("#om-age").value = "young adult";
    $("#om-pitch").value = "moderate pitch";
    bindStart();
    $("#om-go").onclick = generate;
    $("#om-stop").onclick = () => { stop = true; $("#om-info").textContent = "ჩერდება მიმდინარე ვარიანტის შემდეგ…"; };
    if (running) { $("#om-go").disabled = true; $("#om-stop").hidden = false; }
    renderVariants();
    refreshStatus();
  }

  async function generate() {
    if (running) return;
    const n = Math.max(1, Math.min(8, +$("#om-n").value || 4));
    const desc = { gender: $("#om-gender").value, age: $("#om-age").value, pitch: $("#om-pitch").value, style: $("#om-style").value };
    const text = $("#om-text").value.trim();
    if (!text) { $("#om-text").focus(); return; }
    running = true; stop = false; variants = []; saving = null; savedId = null;
    $("#om-go").disabled = true; $("#om-stop").hidden = false; $("#om-bar").hidden = false;
    renderVariants();
    const base = Math.floor(Math.random() * 1e6);
    const t0 = performance.now();
    for (let i = 0; i < n && !stop; i++) {
      $("#om-info").textContent = `იქმნება ვარიანტი ${i + 1} / ${n}…${i === 0 && !st?.server?.loaded ? " (პირველად — მოდელიც იტვირთება)" : ""}`;
      $("#om-bar span").style.width = `${(i / n) * 100}%`;
      refreshStatus();
      try {
        const r = await post("/api/omni/design", { text, ...desc, seed: base + i });
        variants.push({ ...r, desc });
        renderVariants();
      } catch (e) { $("#om-info").textContent = "შეცდომა: " + e.message; break; }
    }
    $("#om-bar span").style.width = "100%";
    if (variants.length) $("#om-info").textContent = `✓ ${variants.length} ვარიანტი · ${Math.round((performance.now() - t0) / 1000)} წმ`;
    running = false;
    $("#om-go").disabled = false; $("#om-stop").hidden = true;
    setTimeout(() => { if (!running) $("#om-bar").hidden = true; }, 800);
    refreshStatus();
  }

  function renderVariants() {
    const el = $("#om-variants");
    if (!el) return;
    el.innerHTML = variants.map((v, i) => `
      <div class="om-variant" data-i="${i}">
        <div class="om-vhead"><b>ვარიანტი ${i + 1}</b><span class="muted small">seed ${v.seed} · ${v.seconds} წმ</span></div>
        <audio controls preload="none" src="${esc(v.url)}"></audio>
        ${saving === i ? `
          <form class="om-save">
            <label class="lbl">სახელი <input name="name" required maxlength="60" placeholder="მაგ. კაცი 2"></label>
            <div class="om-grid">
              <label class="lbl">სქესი <input name="gender" maxlength="20" value="${esc(opts.gender.items[v.desc.gender] || "")}"></label>
              <label class="lbl">ასაკი <input name="age" maxlength="20" value="${esc(AGE_HINT[v.desc.age] || "")}"></label>
            </div>
            <label class="lbl">ხასიათი <input name="character" maxlength="200" placeholder="მაგ. მშვიდი, თბილი, მთხრობელი"></label>
            <div class="actions">
              <button class="primary small" type="submit">${UI.icon("save", 14)} შენახვა</button>
              <button class="ghost small" type="button" data-cancel>გაუქმება</button>
            </div>
          </form>` : `<div class="actions tight"><button class="ghost small" data-save>${UI.icon("save", 14)} შენახვა ბიბლიოთეკაში</button></div>`}
      </div>`).join("") + (savedId ? `<div class="om-saved small"><span class="verdict good">✓ შენახულია: ${esc(savedId)}</span>
        <button class="ghost small" data-listen-saved>${UI.icon("play", 12)} „მოსმენაში“ გახსნა</button></div>` : "");
    el.querySelector("[data-listen-saved]")?.addEventListener("click", () => window.listenVoice?.(savedId));
    el.querySelectorAll("[data-save]").forEach((b) => b.addEventListener("click", () => {
      saving = +b.closest("[data-i]").dataset.i; renderVariants();
      el.querySelector("form.om-save input[name=name]")?.focus();
    }));
    el.querySelector("[data-cancel]")?.addEventListener("click", () => { saving = null; renderVariants(); });
    const f = el.querySelector("form.om-save");
    if (f) f.onsubmit = async (e) => {
      e.preventDefault();
      const v = variants[saving];
      const btn = f.querySelector("[type=submit]");
      btn.disabled = true; btn.textContent = "ინახება…";
      try {
        const r = await post("/api/omni/voices", { design_id: v.id, ...Object.fromEntries(new FormData(f)) });
        UI.toast(`ხმა შენახულია (${r.id}) — „ხმები → ბიბლიოთეკა“, სტატუსი: სატესტო`, "ok");
        saving = null; savedId = r.id; renderVariants();
        await window.reloadVoices?.();
      } catch (err) { UI.toast(err.message, "error"); btn.disabled = false; btn.textContent = "შენახვა"; }
    };
  }

  // გენერაციისას თავიდან არ ვხატავთ — ვარიანტები და ღილაკების მდგომარეობა უნდა დარჩეს
  const show = () => { if (!box().innerHTML || !running) render().catch((e) => (box().innerHTML = `<div class="empty">შეცდომა: ${esc(e.message)}</div>`)); else refreshStatus(); };
  document.addEventListener("subtab", (e) => { if (e.detail.page === "designer" && e.detail.key === "describe" && !$("#tab-designer").hidden) show(); });
  document.querySelector('.tabs button[data-tab="designer"]')?.addEventListener("click", () => { if (UI.getSub("designer") === "describe") show(); });
  return { render };
})();
