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
  };
  const TARGET = { kaggle: "Kaggle (უფასო GPU)", local: "ეს კომპიუტერი (CPU, ნელი)", gcp: "Google Cloud" };
  let openLog = null, timer = null, designs = [];

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
        <p class="muted small">ნარევი იქმნება „ხმის შემქმნელში“ („შენახვა“). აქედან ის ცალკე, მსუბუქ ხმად იქცევა: ჯერ ამ კომპიუტერზე
          მზადდება ~3000 ჩანაწერი (~1–2 სთ), მერე იწვრთნება Kaggle-ზე (~6 სთ, უფასოდ) და ბოლოს თავისით ჩნდება „ხმებში“.</p>
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
      <div class="card">
        <h2>ხმები და წვრთნა</h2>
        <div id="tr-jobs"></div>
      </div>`;
    bindForm();
  }

  async function renderJobs() {
    const { jobs } = await api("/api/train/jobs");
    const box = $("#tr-jobs");
    box.innerHTML = jobs.length ? jobs.map(jobCard).join("") : `<div class="empty">ჯერ არაფერია.</div>`;
    bind(box, jobs);
  }

  function jobCard(j) {
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
        ${j.ready ? `<button class="ghost small" data-goto="voices">მოსმენა „ხმებში“ →</button>` : ""}
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
        const act = b.dataset.act;
        if (act === "stop" && !confirm("გავაჩერო?")) return;
        b.disabled = true;
        try {
          if (act === "start") {
            const on = el.querySelector("[data-on]").value, epochs = +el.querySelector("[data-epochs]").value;
            if (on === "local" && !confirm("ამ კომპიუტერის პროცესორზე წვრთნა ძალიან ნელია (დღეები). გავაგრძელო?")) { b.disabled = false; return; }
            await post(`/api/train/${slot}/start`, { on, epochs });
          } else await post(`/api/train/${slot}/${act}`);
          openLog = slot;
          setTimeout(renderJobs, 1500);
        } catch (err) { alert(err.message); b.disabled = false; }
      }));
      el.querySelector("[data-log]").addEventListener("click", () => { openLog = openLog === slot ? null : slot; renderJobs(); });
      el.querySelector("[data-goto]")?.addEventListener("click", () => document.querySelector('.tabs button[data-tab="voices"]').click());
      if (openLog === slot) api(`/api/train/${slot}/log`).then((l) => {
        el.querySelector(".tr-log").textContent = (l.ctl || "").split("\n").slice(-40).join("\n") + (l.train ? "\n---- train.log ----\n" + l.train.split("\n").slice(-15).join("\n") : "") || "ლოგი ცარიელია";
      });
    });
    clearInterval(timer);
    if (jobs.some((j) => j.busy || ["training", "uploading", "prepare"].includes(j.stage)))
      timer = setInterval(() => { if (!$("#tab-train").hidden && !document.hidden) renderJobs(); }, 15000);
  }

  return { render };
})();
