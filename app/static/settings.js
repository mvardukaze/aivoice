// „პარამეტრები“ + პირველი დაყენების ოსტატი (backend: app/settings.py, app/config.py)
window.Settings = (() => {
  const $ = (s) => document.querySelector(s);
  const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
  const api = async (url, opts) => {
    const r = await fetch(url, opts);
    const body = await r.json().catch(() => ({}));
    if (!r.ok) throw new Error(body.detail || r.statusText);
    return body;
  };
  const post = (url, data) => api(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(data) });

  const SECTION = { paths: "მისამართები", server: "პანელი", kaggle: "Kaggle", gcp: "Google Cloud", hf: "Hugging Face", training: "წვრთნა" };
  const STATE = { ok: ["✓", "good"], warn: ["!", "maybe"], missing: ["✗", "bad"] };
  let restartNeeded = false;

  async function render() {
    const box = $("#settings");
    const [st, s] = await Promise.all([api("/api/setup/status"), api("/api/settings")]);
    const groups = {};
    st.items.forEach((i) => (groups[i.group] ||= []).push(i));
    box.innerHTML = `
      ${!st.configured ? `<div class="card wizard">
        <h2>მოგესალმებით! პირველი დაყენება</h2>
        <p>პროექტი პირველად ეშვება ამ კომპიუტერზე. სამი ნაბიჯი:</p>
        <ol>
          <li><b>მონაცემების საქაღალდე</b> — ქვემოთ „მისამართებში“ მიუთითეთ, სად შეინახოს დიდი ფაილები (≥ 50 GB), და დააჭირეთ „შენახვა“.</li>
          <li><b>ანგარიშები</b> — Kaggle-ის token (წვრთნისთვის) და Hugging Face-ის token (მოდელების ჩამოსატვირთად).</li>
          <li><b>მოდელები</b> — ქვემოთ „მოდელები“ → „ჩამოტვირთვა“ (ხმის შემქმნელი და მზა ხმები).</li>
        </ol>
        <p class="muted small">ყველა ნაბიჯის მდგომარეობა ჩანს ქვემოთ „შემოწმებაში“ — მწვანე ✓ ნიშნავს, რომ მზადაა.</p>
      </div>` : ""}

      <div class="card" id="upd-card" data-pane="update">
        <h2>განახლება</h2>
        <div class="muted small" id="upd-cur"></div>
        <div id="upd-body" class="small muted">მოწმდება GitHub…</div>
        <pre class="tr-log" id="upd-log" hidden></pre>
      </div>

      <div class="card" data-pane="checks">
        <div class="card-head"><h2>შემოწმება</h2>
          <span class="verdict ${st.ready ? "good" : "bad"}">${st.ready ? "ყველაფერი აუცილებელი მზადაა" : "რაღაც აკლია"}</span></div>
        ${Object.entries(groups).map(([g, items]) => `
          <div class="check-group"><div class="lbl">${esc(g)}</div>
          <div class="checks">${items.map((i) => `
            <div class="check-row"><span class="verdict ${STATE[i.state][1]}">${STATE[i.state][0]}</span>
              <b>${esc(i.name)}</b> <span class="muted small">${esc(i.detail)}</span></div>`).join("")}</div></div>`).join("")}
        <div class="actions"><button class="ghost small" id="set-recheck">თავიდან შემოწმება</button></div>
      </div>

      <div class="card" data-pane="config">
        <div class="card-head"><h2>პარამეტრები</h2><span class="muted small mono">${esc(s.file)}</span></div>
        <p class="muted small">მისამართებში შეგიძლიათ გამოიყენოთ <code>{data_root}</code> და <code>~</code>. ცვლილება ძალაში შედის პანელის გადატვირთვის შემდეგ.</p>
        <form id="set-form">
          ${s.sections.map((sec) => `
            <fieldset><legend>${esc(SECTION[sec.key] || sec.key)}</legend>
              ${sec.fields.map((f) => `
                <label class="set-field">
                  <span class="set-label">${esc(f.label)}</span>
                  <input name="${esc(sec.key)}.${esc(f.key)}" type="${f.type === "int" ? "number" : "text"}" value="${esc(f.value)}" placeholder="${esc(f.default)}">
                  <span class="muted small">${esc(f.help)}${f.resolved ? ` → <span class="mono">${esc(f.resolved)}</span> ${f.exists ? "✓" : "<span class='err'>(არ არსებობს — შენახვისას შეიქმნება)</span>"}` : ""}</span>
                </label>`).join("")}
            </fieldset>`).join("")}
          <div class="actions">
            <button class="primary" type="submit">შენახვა</button>
            <button class="ghost" type="button" id="set-restart" ${restartNeeded ? "" : "hidden"}>პანელის გადატვირთვა</button>
            <span class="muted small" id="set-msg"></span>
          </div>
        </form>
      </div>

      <div class="card" id="models-card" data-pane="models">
        <div class="card-head"><h2>მოდელები (Hugging Face)</h2><span class="muted small mono" id="mdl-repo"></span></div>
        <p class="muted small help">ხმები და ნარევები საერთო პირად რეპოზიტორიაშია (Hugging Face; საჭიროა Read token და ავტორის მიცემული წვდომა). „ჩამოტვირთვა“ მოიტანს ახალს, „ატვირთვა“ გაგზავნის თქვენსას (Write token).</p>
        <details class="hint"><summary>${UI.icon("help", 14)} დაწვრილებით</summary>
          <p class="muted small">„ჩამოტვირთვა“ აქ შეცვლილ პარამეტრებს არ გადაწერს. „ატვირთვა“ აგზავნის მხოლოდ შეცვლილს და საჭიროებს Write token-ს.
          „დამალული“ ხმები არ იტვირთება.</p></details>
        <label class="check small"><input type="checkbox" id="mdl-base"> საბაზო checkpoint-ებიც (~1.8 GB — საჭიროა ახალი ხმის მოსამზადებლად)</label>
        <label class="check small"><input type="checkbox" id="mdl-data"> + ხარისხის მონაცემები (~4.5 GB) — საბაზოს გასაუმჯობესებლად</label>
        <div class="actions">
          <button class="primary small" data-mdl="pull">${UI.icon("download", 14)} ჩამოტვირთვა</button>
          <button class="ghost small" data-mdl="push">${UI.icon("download", 14).replace('class="ico"', 'class="ico flip"')} ატვირთვა</button>
          <button class="ghost small" id="mdl-check">შედარება</button>
        </div>
        <div id="mdl-files" class="small"></div>
        <pre class="tr-log" id="mdl-log" hidden></pre>
      </div>

      <div class="card" data-pane="accounts">
        <h2>ანგარიშები და გასაღებები</h2>
        <p class="muted small">გასაღები მხოლოდ ამ კომპიუტერზე ინახება (დაცულ ფაილში) და პანელი მას უკან არასოდეს აჩვენებს. ჩატში ან სხვაგან ნუ ჩასვამთ.</p>
        ${s.secrets.map((x) => `
          <div class="secret-row">
            <div><span class="verdict ${x.present ? "good" : "maybe"}">${x.present ? "✓ დაყენებულია" : "არ არის"}</span> <b>${esc(x.label)}</b>
              <div class="muted small">${esc(x.help)} <span class="mono">(${esc(x.where)})</span></div></div>
            <div class="secret-in"><input type="password" autocomplete="off" data-secret="${esc(x.name)}" placeholder="${x.present ? "შეცვლა…" : "ჩასვით აქ"}">
              <button class="ghost small" data-secret-save="${esc(x.name)}">შენახვა</button></div>
          </div>`).join("")}
      </div>`;

    $("#set-recheck").onclick = render;
    bindModels();
    renderUpdate();
    $("#set-form").onsubmit = async (e) => {
      e.preventDefault();
      const values = {};
      new FormData(e.target).forEach((v, k) => { const [sec, key] = k.split("."); (values[sec] ||= {})[key] = v; });
      try {
        await post("/api/settings", { values });
        restartNeeded = true;
        await render();
        $("#set-msg").textContent = "შენახულია. ძალაში შესვლისთვის — „პანელის გადატვირთვა“.";
      } catch (err) { $("#set-msg").textContent = "შეცდომა: " + err.message; }
    };
    $("#set-restart").onclick = async () => {
      $("#set-msg").textContent = "იტვირთება თავიდან…";
      await post("/api/settings/restart", {}).catch(() => {});
      setTimeout(() => location.reload(), 4000);
    };
    box.querySelectorAll("[data-secret-save]").forEach((b) => b.addEventListener("click", async () => {
      const inp = box.querySelector(`[data-secret="${b.dataset.secretSave}"]`);
      if (!inp.value.trim()) { inp.focus(); return; }
      try { await post("/api/settings/secret", { name: b.dataset.secretSave, value: inp.value }); inp.value = ""; render(); UI.toast("გასაღები შენახულია", "ok"); }
      catch (err) { UI.toast(err.message, "error"); }
    }));
  }

  // ---------- განახლება ----------
  async function renderUpdate() {
    const u = await api("/api/update").catch((e) => ({ ok: false, error: e.message }));
    const body = $("#upd-body");
    if (!body) return;
    markTab(u);
    if (!u.ok) { body.innerHTML = `<span class="err">${esc(u.error)}</span>`; return; }
    $("#upd-cur").textContent = "ახლა: " + u.current;
    if (!u.behind) {
      body.innerHTML = `<div class="actions tight"><span class="verdict good">${UI.icon("check", 12)} უახლესი ვერსიაა</span> <button class="ghost small" id="upd-recheck">ხელახლა შემოწმება</button></div>`;
      $("#upd-recheck").onclick = renderUpdate;
      return;
    }
    body.innerHTML = `
      <div><span class="verdict maybe">ახალი ვერსია: ${u.behind} ცვლილება</span></div>
      <ul class="upd-list">${u.commits.map((c) => { const [, d, msg] = c.split("|"); return `<li><span class="mono">${esc(d)}</span> ${esc(msg.split("\n")[0])}</li>`; }).join("")}</ul>
      ${u.needs.requirements || u.needs.training ? `<div>ბიბლიოთეკებიც განახლდება (შეიძლება რამდენიმე წუთი დასჭირდეს).</div>` : ""}
      ${u.blocked.length ? `<div class="err">ხელით შეცვლილი ფაილები ხელს უშლის: ${u.blocked.map(esc).join(", ")}</div>` : ""}
      ${u.ahead ? `<div class="err">ამ კომპიუტერზე საკუთარი ცვლილებებია (${u.ahead}) — ავტომატური განახლება შეუძლებელია.</div>` : ""}
      <div class="actions"><button class="primary small" id="upd-go" ${u.blocked.length || u.ahead ? "disabled" : ""}>განახლება და გადატვირთვა</button></div>`;
    $("#upd-go").onclick = async () => {
      if (!(await UI.confirm("განვაახლო? მიმდინარე წვრთნა/მომზადება არ შეწყდება; პანელი თავიდან ჩაიტვირთება.", { okText: "განახლება" }))) return;
      try { await post("/api/update", {}); pollUpdate(); } catch (err) { UI.toast(err.message, "error"); }
    };
  }

  function pollUpdate() {
    const el = $("#upd-log");
    el.hidden = false;
    document.querySelector("#upd-go")?.setAttribute("disabled", "");
    const t = setInterval(async () => {
      const l = await api("/api/update/log").catch(() => null);
      if (!l) return;
      el.textContent = l.log + (l.busy ? "\n… მიმდინარეობს" : "");
      if (l.busy) return;
      clearInterval(t);
      if (l.code === 0) {
        el.textContent += "\n↻ პანელი თავიდან იტვირთება…";
        await post("/api/settings/restart", {}).catch(() => {});
        setTimeout(() => location.reload(), 5000);
      }
    }, 2000);
  }

  function markTab(u) {
    const b = document.querySelector('.tabs button[data-tab="settings"]');
    if (b && u?.ok && u.behind) { b.classList.add("attn"); b.title = `ახალი ვერსია: ${u.behind} ცვლილება`; }
    window.UI?.setStatus("update", u?.ok && u.behind ? { text: "ახალი ვერსია", kind: "accent", title: `ახალი ვერსია: ${u.behind} ცვლილება`, go: ["settings", "update"] } : null);
  }

  // ---------- მოდელები ----------
  const MDL = { same: ["✓", "good", "ერთნაირია"], differs: ["≠", "maybe", "განსხვავდება"], remote: ["↓", "maybe", "მხოლოდ რეპოზიტორიაში"], local: ["↑", "maybe", "მხოლოდ აქ"] };
  let mdlTimer = null;

  function bindModels() {
    document.querySelectorAll("[data-mdl]").forEach((b) => b.addEventListener("click", async () => {
      const base = $("#mdl-base").checked, data = $("#mdl-data").checked;
      const extra = [base && "საბაზო checkpoint-ებიც", data && "ხარისხის მონაცემებიც"].filter(Boolean).join(", ");
      if (b.dataset.mdl === "push" && !(await UI.confirm("ატვირთვა საერთო რეპოზიტორიაში" + (extra ? ` (${extra})` : "") + "?", { okText: "ატვირთვა" }))) return;
      try { await post(`/api/models/${b.dataset.mdl}`, { base, data }); pollModels(); }
      catch (err) { UI.toast(err.message, "error"); }
    }));
    $("#mdl-check").onclick = checkModels;
    api("/api/models/log").then((l) => { if (l.busy || l.log) showLog(l); if (l.busy) pollModels(); });
  }

  function showLog(l) {
    const el = $("#mdl-log");
    if (!el) return;
    el.hidden = false;
    el.textContent = l.log.split("\n").slice(-25).join("\n") + (l.busy ? "\n… მიმდინარეობს" : "");
    document.querySelectorAll("[data-mdl]").forEach((b) => (b.disabled = l.busy));
  }

  function pollModels() {
    clearInterval(mdlTimer);
    const tick = async () => {
      const l = await api("/api/models/log").catch(() => null);
      if (!l) return;
      showLog(l);
      if (!l.busy) { clearInterval(mdlTimer); window.reloadVoices?.(); }
    };
    tick();
    mdlTimer = setInterval(tick, 3000);
  }

  async function checkModels() {
    const box = $("#mdl-files");
    box.innerHTML = `<span class="muted">მოწმდება… (დიდი ფაილების პირველად შედარებას ~1 წთ სჭირდება)</span>`;
    const s = await api("/api/models/status").catch((e) => ({ error: e.message, files: [] }));
    $("#mdl-repo").textContent = s.repo || "";
    if (s.error) { box.innerHTML = `<span class="err">${esc(s.error)}</span>`; return; }
    const groups = {};
    s.files.forEach((f) => (groups[f.path.split("/")[0]] ||= []).push(f));
    const NAME = { designer: "ხმის შემქმნელი", designs: "ნარევები", voices: "ხმები", base_ka: "საბაზო მოდელი", ljspeech: "LJSpeech (საწყისი)", cv27: "ხარისხის მონაცემები (Common Voice)", data: "ყურით შემოწმება" };
    box.innerHTML = Object.entries(groups).map(([g, fs]) => {
      const cnt = {};
      fs.forEach((f) => (cnt[f.state] = (cnt[f.state] || 0) + 1));
      const diff = fs.filter((f) => f.state !== "same");
      return `<div class="check-row"><b>${esc(NAME[g] || g)}</b>
        ${Object.entries(cnt).map(([k, n]) => `<span class="verdict ${MDL[k][1]}" title="${MDL[k][2]}">${MDL[k][0]} ${n}</span>`).join(" ")}
        ${diff.length ? `<span class="muted mono">${diff.slice(0, 6).map((f) => `${MDL[f.state][0]} ${esc(f.path.split("/").slice(1).join("/"))}`).join(", ")}${diff.length > 6 ? " …" : ""}</span>` : ""}</div>`;
    }).join("") + `<div class="muted">✓ ერთნაირია · ↓ მხოლოდ რეპოზიტორიაში · ↑ მხოლოდ აქ · ≠ განსხვავდება</div>`;
  }

  // პირველ გაშვებაზე (კონფიგურაცია ჯერ არ არის) — პირდაპირ „პარამეტრებზე“
  async function maybeFirstRun() {
    const st = await api("/api/setup/status").catch(() => null);
    if (st && !st.configured) document.querySelector('.tabs button[data-tab="settings"]')?.click();
    const b = document.querySelector('.tabs button[data-tab="settings"]');
    if (b && st && !st.ready) b.classList.add("attn");
    if (st && !st.ready) window.UI?.setStatus("setup", { text: "დაყენება", kind: "warn", title: "რაღაც აკლია — პარამეტრები → შემოწმება", go: ["settings", "checks"] });
    api("/api/update").then(markTab).catch(() => {});  // ახალი ვერსია — „პარამეტრებზე“ ნიშანი
  }

  return { render, maybeFirstRun };
})();
