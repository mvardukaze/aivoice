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
          <li><b>მოდელები</b> — ჩამოტვირთეთ საბაზო მოდელი და მზა ხმები (ბრძანება: <code>./aivoice models pull</code>).</li>
        </ol>
        <p class="muted small">ყველა ნაბიჯის მდგომარეობა ჩანს ქვემოთ „შემოწმებაში“ — მწვანე ✓ ნიშნავს, რომ მზადაა.</p>
      </div>` : ""}

      <div class="card">
        <div class="card-head"><h2>შემოწმება</h2>
          <span class="verdict ${st.ready ? "good" : "bad"}">${st.ready ? "ყველაფერი აუცილებელი მზადაა" : "რაღაც აკლია"}</span></div>
        ${Object.entries(groups).map(([g, items]) => `
          <div class="lbl">${esc(g)}</div>
          <div class="checks">${items.map((i) => `
            <div class="check-row"><span class="verdict ${STATE[i.state][1]}">${STATE[i.state][0]}</span>
              <b>${esc(i.name)}</b> <span class="muted small">${esc(i.detail)}</span></div>`).join("")}</div>`).join("")}
        <div class="actions"><button class="ghost small" id="set-recheck">თავიდან შემოწმება</button></div>
      </div>

      <div class="card">
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

      <div class="card">
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
      try { await post("/api/settings/secret", { name: b.dataset.secretSave, value: inp.value }); inp.value = ""; render(); }
      catch (err) { alert(err.message); }
    }));
  }

  // პირველ გაშვებაზე (კონფიგურაცია ჯერ არ არის) — პირდაპირ „პარამეტრებზე“
  async function maybeFirstRun() {
    const st = await api("/api/setup/status").catch(() => null);
    if (st && !st.configured) document.querySelector('.tabs button[data-tab="settings"]')?.click();
    const b = document.querySelector('.tabs button[data-tab="settings"]');
    if (b && st && !st.ready) b.classList.add("attn");
  }

  return { render, maybeFirstRun };
})();
