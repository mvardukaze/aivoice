// „წვრთნა“: ახალი ხმა ნარევიდან → მომზადება → გაშვება (Kaggle / ეს კომპიუტერი) → ლოდინი → „ხმებში“ (backend: app/train_api.py)
window.Train = (() => {
  const $ = (s) => document.querySelector(s);
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const api = async (url, opts) => {
    const r = await fetch(url, opts);
    const body = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(body.detail?.[0]?.msg || body.detail || r.statusText);
    return body;
  };
  const post = (url, data = {}) => api(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });

  const STAGE = {
    new: ["შექმნილია", "planned"], prepare: ["მზადდება", "training"], prepared: ["მზადაა გასაშვებად", "planned"],
    uploading: ["იტვირთება", "training"], training: ["იწვრთნება", "training"], partial: ["სესია დასრულდა", "training"],
    done: ["დასრულდა", "ready"], failed: ["შეცდომა", "bad"], stopped: ["გაჩერებულია", "planned"],
    selecting: ["ირჩევა ჩანაწერები", "training"],
  };
  const TARGET = { kaggle: "Kaggle (უფასო GPU)", local: "ეს კომპიუტერი (CPU, ნელი)", gcp: "Google Cloud", gpu: "ეს კომპიუტერი (GPU)" };
  const ACTIVE = ["training", "uploading", "prepare", "selecting"];
  let openLog = null, timer = null, designs = [], baseOpts = null;

  const slugify = (s) => (s || "").toLowerCase().replace(/[^a-z0-9]+/g, "_").replace(/^_|_$/g, "");

  // ფორმა ერთხელ იხატება (აკრეფილი არ იკარგება); სია — ყოველ განახლებაზე
  async function render() {
    const box = $("#train");
    if (!$("#tr-new")) await renderForm(box);
    await renderJobs();
  }

  async function renderForm(box) {
    designs = await api("/api/designer/designs").catch(() => []);
    box.innerHTML = `
      <div class="card">
        <h2>ახალი ხმა შენახული ნარევიდან</h2>
        <p class="muted small help">შენახული ნარევი ცალკე, მსუბუქ ხმად იქცევა და ~8 საათში თავისით გამოჩნდება „ხმებში“.</p>
        ${UI.hint(`ნარევი იქმნება „ხმის შემქმნელში“ („შენახვა“). ჯერ ამ კომპიუტერზე მზადდება ~3000 ჩანაწერი (~1–2 სთ), მერე იწვრთნება Kaggle-ზე (~6 სთ, უფასოდ).`)}
        <form id="tr-new" class="tr-new">
          <label>ნარევი <select name="design" required>${designs.map((d) => `<option>${esc(d.name)}</option>`).join("")}</select></label>
          <label>ხმის სახელი <input name="name" placeholder="მაგ. ქალი 2" required></label>
          <label>ID <input name="slot" pattern="[a-z][a-z0-9_]{1,30}" placeholder="მაგ. f_mix_02" required></label>
          <label>სქესი <select name="gender"><option value="ქალი">ქალი</option><option value="კაცი">კაცი</option><option value="">ნეიტრალური</option></select></label>
          <label>ეპოქები <input name="epochs" type="number" value="150" min="10" max="2000"></label>
          <button class="primary" type="submit">შექმნა</button>
          <span class="muted small" id="tr-new-msg"></span>
        </form>
      </div>
      <details class="card tr-base" id="tr-base">
        <summary><h2>საბაზო მოდელი — შექმნა / გაუმჯობესება</h2></summary>
        <div id="tr-base-body"><p class="muted small">იტვირთება…</p></div>
      </details>
      <div class="card">
        <h2>ხმები და წვრთნა</h2>
        <div id="tr-jobs"></div>
      </div>`;
    bindForm();
    await loadBase();
  }

  // ---------------- საბაზო მოდელი (backend: tools/base_ctl.py) ----------------
  const R = () => baseOpts.rates;
  // ≈ დრო/ფასი Google Cloud L4-ზე: ბატჩი/ეპოქა ≈ ჩანაწერები / 32, ≈ 0.8 ბატჩი/წმ, ≈ $0.9/სთ
  function estimate(clips, epochs) {
    const per = Math.max(1, Math.round((clips * 0.99) / R().batch));
    const h = (epochs * per) / R().batch_per_s / 3600;
    return { per, h, usd: h * R().usd_per_h };
  }
  const fmtH = (h) => (h >= 48 ? `${Math.round(h)} სთ (≈ ${(h / 24).toFixed(1)} დღე)` : `${h.toFixed(1)} სთ`);

  async function loadBase() {
    try { baseOpts = await api("/api/train/base/options"); }
    catch (err) { $("#tr-base-body").innerHTML = `<p class="err small">${esc(err.message)}</p>`; return; }
    renderBase();
  }

  function renderBase() {
    const o = baseOpts, ms = o.models, sels = o.selections, g = o.gpu, gc = o.gcp;
    const last = ms[ms.length - 1];
    const gpuWhy = !g ? "NVIDIA GPU ვერ ვიპოვე" : !g.ok ? g.reason : !o.training_env ? "წვრთნის გარემო არ არის (./install.sh --training)" : "";
    const gcpWhy = !gc.configured ? "Google Cloud არ არის მორგებული (gcloud)" : "";
    $("#tr-base-body").innerHTML = `
      <p class="muted small help">საბაზო მოდელი ასობით ადამიანის ხმაზე ისწავლება — მისგან კეთდება ხმის შემქმნელი და ყველა ახალი ხმა.
        გაგრძელება = არსებულ მოდელს ემატება ახალი ჩანაწერები და კიდევ რამდენიმე ეპოქა (ძველი ხმები უცვლელად რჩება).</p>
      ${UI.hint(`ნაბიჯები: <b>შექმნა</b> (ჩანაწერების შერჩევა, წამები) → <b>მომზადება</b> (აუდიოს დამუშავება ამ კომპიუტერზე, 1–3 სთ) → <b>გაშვება</b>
        (Google Cloud L4 ან ამ კომპიუტერის ძლიერი GPU, 1–4 დღე). დასრულებისას ახალი მოდელი თავისით ჩნდება „ხმის შემქმნელში“.
        ერთდროულად Google Cloud-ზე მხოლოდ ერთი წვრთნა შეიძლება.`)}
      <form id="tr-base-form" class="tr-base-form">
        <fieldset><legend>1. საიდან</legend>
          <label class="check-inline"><input type="radio" name="mode" value="continue" ${ms.length ? "checked" : "disabled"}> არსებულის გაგრძელება</label>
          <select name="from_model" ${ms.length ? "" : "disabled"}>${ms.map((m) => `<option value="${esc(m.slot)}" ${m === last ? "selected" : ""}>${esc(m.slot)} · ${m.epoch} ეპოქა · ${m.hours} სთ · ${m.speakers} ხმა</option>`).join("")}</select>
          <label class="check-inline"><input type="radio" name="mode" value="scratch" ${ms.length ? "" : "checked"}> ნულიდან (LJSpeech-იდან)</label>
        </fieldset>
        <fieldset><legend>2. ჩანაწერები (Common Voice, ხარისხით შერჩეული)</legend>
          <label class="check-inline"><input type="radio" name="data" value="new" ${o.quality ? "checked" : "disabled"}> ახალი შერჩევა</label>
          <label class="check-inline"><input name="hours" type="number" value="40" min="1" max="400" step="5" style="width:80px"> საათი</label>
          <div class="tr-checks" data-new>
            <label class="check-inline"><input type="checkbox" name="short" checked> მოკლე ფრაზებიც („კი“, „არა, მადლობა“)</label>
            <label class="check-inline"><input type="checkbox" name="punct" checked> მეტი კითხვა და ძახილი (?, !)</label>
            <label class="check-inline"><input type="checkbox" name="clusters" checked> რთული ბგერები: რ ღ ბ ფ პ თანხმოვნებთან</label>
          </div>
          ${o.quality ? "" : `<p class="muted small">ახალი შერჩევისთვის ჯერ საჭიროა ხარისხის შემოწმება („მონაცემები → ხარისხი“).</p>`}
          <label class="check-inline"><input type="radio" name="data" value="existing" ${sels.length ? (o.quality ? "" : "checked") : "disabled"}> არსებული შერჩევა</label>
          <select name="selection" ${sels.length ? "" : "disabled"}>${sels.length ? sels.map((x) => `<option value="${esc(x.name)}">${esc(x.name)} · ${x.hours} სთ · ${x.clips} ჩანაწერი</option>`).join("") : `<option value="">— არ არის —</option>`}</select>
          <p class="muted small" data-cont-note>გაგრძელებისას აირჩევა მხოლოდ ჯერ გამოუყენებელი ჩანაწერები და დაემატება წინა მოდელის ჩანაწერებს.</p>
        </fieldset>
        <fieldset><legend>3. ეპოქები</legend>
          <label class="check-inline"><input name="epochs" type="number" min="1" max="2000" style="width:90px"> <span data-ep-label></span></label>
          <p class="small" data-est></p>
        </fieldset>
        <fieldset><legend>4. სად</legend>
          <label class="check-inline" ${gcpWhy ? `title="${esc(gcpWhy)}"` : ""}><input type="radio" name="on" value="gcp" ${gcpWhy ? "disabled" : "checked"}> Google Cloud (L4, ფასიანი, ≈ $${R().usd_per_h}/სთ)</label>
          ${gcpWhy ? `<span class="muted small">${esc(gcpWhy)}</span>` : ""}
          ${gc.busy ? `<p class="small warn-t">Google Cloud ახლა დაკავებულია: ${esc(gc.slot)} (${esc(gc.phase)}) — გაშვება მის დასრულებამდე ვერ მოხერხდება.</p>` : ""}
          <label class="check-inline"><input type="radio" name="on" value="local" ${gpuWhy ? "disabled" : gcpWhy ? "checked" : ""}> ეს კომპიუტერი (GPU${g ? ": " + esc(g.name) + ", " + g.vram_gb + " GB" : ""})</label>
          ${gpuWhy ? `<span class="muted small">— ${esc(gpuWhy)}</span>` : ""}
        </fieldset>
        <div class="tr-base-foot">
          <label>ID <input name="slot" value="${esc(o.next_slot)}" pattern="base_ka_v[0-9]{1,3}" required style="width:130px"></label>
          <button class="primary" type="submit">შექმნა</button>
          <span class="muted small" id="tr-base-msg"></span>
        </div>
      </form>`;
    bindBase();
  }

  function baseForm() {
    const f = $("#tr-base-form"), mode = f.querySelector("[name=mode]:checked")?.value || "scratch";
    const data = f.querySelector("[name=data]:checked")?.value || "new";
    return { f, mode, data, from: baseOpts.models.find((m) => m.slot === f.from_model.value),
             sel: baseOpts.selections.find((x) => x.name === f.selection.value) };
  }

  function updateBase() {
    const { f, mode, data, from, sel } = baseForm();
    if (!f.epochs.dataset.touched) f.epochs.value = mode === "scratch" ? R().scratch_epochs : R().continue_epochs;
    f.from_model.disabled = mode !== "continue";
    f.hours.disabled = data !== "new";
    f.querySelectorAll("[data-new] input").forEach((i) => (i.disabled = data !== "new"));
    f.selection.disabled = data !== "existing" || !baseOpts.selections.length;
    f.querySelector("[data-cont-note]").hidden = mode !== "continue";
    const ep = +f.epochs.value || 0, cont = mode === "continue" && from;
    f.querySelector("[data-ep-label]").textContent = cont ? `დამატებით (ეპოქა ${from.epoch} → ${from.epoch + ep})` : "სულ";
    const added = data === "existing" ? (sel?.clips || 0) : Math.round((+f.hours.value || 0) * R().clips_per_hour);
    const clips = added + (cont ? from.clips : 0), e = estimate(clips, ep);
    const local = f.querySelector("[name=on]:checked")?.value === "local";
    f.querySelector("[data-est]").innerHTML = `≈ ${clips.toLocaleString("en")} ჩანაწერი · ${e.per} ბატჩი ეპოქაში · ` +
      (local ? `ამ კომპიუტერზე დრო GPU-ზეა დამოკიდებული (Google Cloud L4-ზე იქნებოდა ≈ ${fmtH(e.h)})`
             : `Google Cloud L4: <b>≈ ${fmtH(e.h)}</b>, <b>≈ $${Math.round(e.usd)}</b>`) +
      (data === "new" ? `<br><span class="muted">ზუსტი რაოდენობა შერჩევის შემდეგ გამოჩნდება (თუ შესაფერისი ჩანაწერი ნაკლებია — ნაკლები იქნება).</span>` : "");
  }

  function bindBase() {
    const f = $("#tr-base-form");
    f.epochs.addEventListener("input", () => (f.epochs.dataset.touched = "1"));
    f.addEventListener("change", updateBase);
    f.addEventListener("input", updateBase);
    updateBase();
    f.onsubmit = async (e) => {
      e.preventDefault();
      const { mode, data } = baseForm(), msg = $("#tr-base-msg");
      const d = { slot: f.slot.value, mode, from_model: mode === "continue" ? f.from_model.value : "", epochs: +f.epochs.value,
                  on: f.querySelector("[name=on]:checked")?.value || "gcp", hours: +f.hours.value || 40,
                  selection: data === "existing" ? f.selection.value : "",
                  short: data === "new" && f.short.checked, punct: data === "new" && f.punct.checked, clusters: data === "new" && f.clusters.checked };
      const btn = f.querySelector("button[type=submit]");
      btn.disabled = true;
      try {
        await post("/api/train/base/new", d);
        msg.textContent = "იქმნება — ქვემოთ, ბარათზე დააჭირეთ „მომზადებას“.";
        openLog = d.slot;
        setTimeout(async () => { await renderJobs(); await loadBase(); $("#tr-base-msg").textContent = msg.textContent; }, 1500);
      } catch (err) { msg.textContent = "შეცდომა: " + err.message; }
      btn.disabled = false;
    };
  }

  async function renderJobs() {
    const { jobs } = await api("/api/train/jobs");
    const box = $("#tr-jobs");
    box.innerHTML = jobs.length ? jobs.map(jobCard).join("") : `<div class="empty">${UI.icon("activity", 22)}<div>ჯერ არაფერია.</div></div>`;
    bind(box, jobs);
  }

  function jobCard(j) {
    if (j.base) return baseCard(j);
    const [label, cls] = STAGE[j.stage] || [j.stage || "—", "planned"];
    const pct = j.max_epochs ? Math.min(100, (j.epoch / j.max_epochs) * 100) : 0;
    const base = j.kind === "base";
    const canPrepare = !base && !j.busy && ["new", "failed", "stopped", "prepared", undefined, null].includes(j.stage) && !(j.clips && j.init);
    const canStart = !base && !j.busy && j.clips > 0 && j.init && !["training", "uploading", "prepare"].includes(j.stage);
    const running = ["training", "uploading", "partial"].includes(j.stage);
    return `<div class="tr-job" data-slot="${esc(j.slot)}">
      <div class="run-head">
        <div><b>${esc(j.name)}</b> <span class="muted small mono">${esc(j.slot)}</span>
          ${j.design ? `<span class="muted small"> · ნარევი „${esc(j.design)}“</span>` : ""}${base ? `<span class="muted small"> · საბაზო მოდელი</span>` : ""}</div>
        <span class="badge ${cls}">${esc(label)}${j.busy ? " …" : ""}</span>
      </div>
      <div class="bar"><span style="width:${pct.toFixed(1)}%"></span></div>
      <div class="muted small">ეპოქა ${j.epoch} / ${j.max_epochs ?? "—"}
        ${j.target ? ` · ${esc(TARGET[j.target] || j.target)}` : ""}
        ${j.clips ? ` · ${j.clips} ჩანაწერი` : ""}${j.init === false && !base ? " · საწყისი checkpoint ჯერ არ არის" : ""}
        ${j.local?.running && j.local.steps_per_epoch ? ` · მიმდინარე ეპოქა ${Math.round(j.local.step_in_epoch / j.local.steps_per_epoch * 100)}%` : ""}
        ${j.note ? ` · ${esc(j.note)}` : ""}${j.error ? ` · <span class="err">${esc(j.error)}</span>` : ""}</div>
      <div class="actions">
        ${canPrepare ? `<button class="primary small" data-act="prepare">1. მომზადება (ამ კომპიუტერზე)</button>` : ""}
        ${canStart ? `<select data-on><option value="kaggle">Kaggle (რეკომენდებული)</option><option value="local">ეს კომპიუტერი (CPU)</option></select>
          <input type="number" data-epochs value="${j.max_epochs || 150}" min="10" max="2000" style="width:80px" title="ეპოქები">
          <button class="primary small" data-act="start">2. წვრთნის გაშვება</button>` : ""}
        ${running || j.busy ? `<button class="ghost small" data-act="stop">გაჩერება</button>` : ""}
        ${j.stage === "partial" || (running && j.target === "kaggle") ? `<button class="ghost small" data-act="fetch">შედეგის ჩამოტვირთვა</button>` : ""}
        ${j.ready ? `<button class="ghost small" data-goto="voices">${j.voice_status === "testing" ? "შემოწმება და დამტკიცება „ხმებში“ →" : "მოსმენა „ხმებში“ →"}</button>` : ""}
        <button class="ghost small" data-log>ლოგი</button>
      </div>
      <pre class="tr-log" ${openLog === j.slot ? "" : "hidden"}></pre>
    </div>`;
  }

  // base_ctl-ით შექმნილი საბაზო მოდელი: მომზადება → გაშვება (Google Cloud / ეს კომპიუტერი) → გაჩერება
  function baseCard(j) {
    const b = j.base, [label, cls] = STAGE[j.stage] || [j.stage || "—", "planned"];
    const from = b.from_epoch || 0, ep = Math.max(j.epoch || 0, from);
    const pct = j.max_epochs > from ? Math.min(100, ((ep - from) / (j.max_epochs - from)) * 100) : 0;
    const active = ACTIVE.includes(j.stage);
    const canPrepare = !j.busy && !active && j.stage !== "done" && (!j.clips || j.stage === "failed");
    const canStart = !j.busy && !active && j.stage !== "done" && j.clips > 0;
    const g = baseOpts?.gpu, gpuOk = g?.ok && baseOpts?.training_env;
    const left = j.max_epochs - ep, e = baseOpts ? estimate(j.clips, left) : null;
    return `<div class="tr-job" data-slot="${esc(j.slot)}" data-base>
      <div class="run-head">
        <div><b>${esc(j.slot)}</b> <span class="muted small">· საბაზო მოდელი · ${b.from ? `${esc(b.from)}-ის გაგრძელება (+${b.epochs} ეპოქა)` : "ნულიდან"}</span></div>
        <span class="badge ${cls}">${esc(label)}${j.busy ? " …" : ""}</span>
      </div>
      <div class="bar"><span style="width:${pct.toFixed(1)}%"></span></div>
      <div class="muted small">ეპოქა ${ep} / ${j.max_epochs ?? "—"}
        ${j.target ? ` · ${esc(TARGET[j.target] || j.target)}` : ""}
        ${j.clips ? ` · dataset: ${j.clips} ჩანაწერი` : ` · შერჩევა: ${esc(b.selection || "")}`}
        ${j.local?.running && j.local.steps_per_epoch ? ` · მიმდინარე ეპოქა ${Math.round(j.local.step_in_epoch / j.local.steps_per_epoch * 100)}%` : ""}
        ${j.kaggle_state && active ? ` · ${esc(j.kaggle_state)}` : ""}
        ${j.note ? ` · ${esc(j.note)}` : ""}${j.error ? ` · <span class="err">${esc(j.error)}</span>` : ""}</div>
      <div class="actions">
        ${canPrepare ? `<button class="primary small" data-act="prepare">1. მომზადება (dataset, ამ კომპიუტერზე)</button>` : ""}
        ${canStart ? `<select data-on>
            <option value="gcp" ${b.target !== "local" ? "selected" : ""}>Google Cloud (L4)</option>
            <option value="local" ${gpuOk ? (b.target === "local" ? "selected" : "") : "disabled"}>ეს კომპიუტერი (GPU)${gpuOk ? "" : " — " + (g ? "მცირე GPU" : "GPU არ არის")}</option></select>
          <button class="primary small" data-act="start" data-est="${e ? `≈ ${fmtH(e.h)}, ≈ $${Math.round(e.usd)}` : ""}">2. წვრთნის გაშვება</button>` : ""}
        ${active || j.busy ? `<button class="ghost small" data-act="stop">გაჩერება</button>` : ""}
        ${j.stage === "done" ? `<button class="ghost small" data-goto-designer>„ხმის შემქმნელში“ →</button>` : ""}
        <button class="ghost small" data-log>ლოგი</button>
      </div>
      <pre class="tr-log" ${openLog === j.slot ? "" : "hidden"}></pre>
    </div>`;
  }

  function bindForm() {
    const f = $("#tr-new");
    f.name.addEventListener("input", () => { if (!f.slot.dataset.touched) f.slot.value = slugify(f.name.value) ? "v_" + slugify(f.name.value) : ""; });
    f.slot.addEventListener("input", () => (f.slot.dataset.touched = "1"));
    f.onsubmit = async (e) => {
      e.preventDefault();
      const d = Object.fromEntries(new FormData(f));
      d.epochs = +d.epochs;
      try { await post("/api/train/new", d); $("#tr-new-msg").textContent = "შეიქმნა — ქვემოთ დააჭირეთ „მომზადებას“."; f.reset(); renderJobs(); }
      catch (err) { $("#tr-new-msg").textContent = "შეცდომა: " + err.message; }
    };
  }

  function bind(box, jobs) {
    box.querySelectorAll(".tr-job").forEach((el) => {
      const slot = el.dataset.slot;
      el.querySelectorAll("[data-act]").forEach((b) => b.addEventListener("click", async () => {
        const act = b.dataset.act, isBase = el.hasAttribute("data-base");
        if (act === "stop" && !(await UI.confirm("გავაჩერო?", { okText: "გაჩერება", danger: true }))) return;
        b.disabled = true;
        try {
          if (isBase) {
            const on = el.querySelector("[data-on]")?.value;
            if (act === "start" && on === "gcp" && !(await UI.confirm(`Google Cloud-ზე წვრთნა ფასიანია: ${b.dataset.est || ""}. მანქანა დასრულებისას თავისით წაიშლება. გავუშვა?`, { okText: "გაშვება" }))) { b.disabled = false; return; }
            await post(`/api/train/base/${slot}/${act}`, act === "start" ? { on } : {});
          } else if (act === "start") {
            const on = el.querySelector("[data-on]").value, epochs = +el.querySelector("[data-epochs]").value;
            if (on === "local" && !(await UI.confirm("ამ კომპიუტერის პროცესორზე წვრთნა ძალიან ნელია (დღეები). გავაგრძელო?", { okText: "გაგრძელება" }))) { b.disabled = false; return; }
            await post(`/api/train/${slot}/start`, { on, epochs });
          } else await post(`/api/train/${slot}/${act}`);
          openLog = slot;
          setTimeout(renderJobs, 1500);
        } catch (err) { UI.toast(err.message, "error"); b.disabled = false; }
      }));
      el.querySelector("[data-log]").addEventListener("click", () => { openLog = openLog === slot ? null : slot; renderJobs(); });
      el.querySelector("[data-goto-designer]")?.addEventListener("click", () => document.querySelector('.tabs button[data-tab="designer"]').click());
      el.querySelector("[data-goto]")?.addEventListener("click", () => { document.querySelector('.tabs button[data-tab="voices"]').click(); UI.setSub("voices", "library"); });
      if (openLog === slot) api(`/api/train/${slot}/log`).then((l) => {
        el.querySelector(".tr-log").textContent = (l.ctl || "").split("\n").slice(-40).join("\n") + (l.train ? "\n---- train.log ----\n" + l.train.split("\n").slice(-15).join("\n") : "") || "ლოგი ცარიელია";
      });
    });
    clearInterval(timer);
    if (jobs.some((j) => j.busy || ACTIVE.includes(j.stage)))
      timer = setInterval(() => { if (!$("#tab-train").hidden && !document.hidden) renderJobs(); }, 15000);
  }

  return { render };
})();
