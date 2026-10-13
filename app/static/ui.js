// საერთო UI: ხატულები, შეტყობინებები (toast), დადასტურების ფანჯარა, ჩამოსაშლელი მენიუ,
// ქვე-ჩანართები, გვერდითი მენიუ, თემა. ეს ფაილი სხვა სკრიპტებზე ადრე იტვირთება.
window.UI = (() => {
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch {} },
  };
  const escT = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

  // ---------- ხატულები (inline SVG, stroke = currentColor) ----------
  const P = {
    dots: '<circle cx="12" cy="5" r="1.6" fill="currentColor" stroke="none"/><circle cx="12" cy="12" r="1.6" fill="currentColor" stroke="none"/><circle cx="12" cy="19" r="1.6" fill="currentColor" stroke="none"/>',
    edit: '<path d="M12 20h9"/><path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4z"/>',
    trash: '<path d="M3 6h18"/><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6"/><path d="M10 11v6M14 11v6"/><path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2"/>',
    eyeoff: '<path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94"/><path d="M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19"/><path d="M14.12 14.12a3 3 0 1 1-4.24-4.24"/><path d="M1 1l22 22"/>',
    shield: '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><path d="M9 12l2 2 4-4"/>',
    play: '<path d="M7 4.5v15l12-7.5z" fill="currentColor" stroke="none"/>',
    check: '<path d="M20 6L9 17l-5-5"/>',
    x: '<path d="M18 6L6 18M6 6l12 12"/>',
    alert: '<circle cx="12" cy="12" r="9"/><path d="M12 8v4.5M12 16h.01"/>',
    info: '<circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/>',
    activity: '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    wave: '<path d="M3 10v4M7 6v12M11 3v18M15 7v10M19 10v4"/>',
    sparkle: '<path d="M12 3l1.9 5.1L19 10l-5.1 1.9L12 17l-1.9-5.1L5 10l5.1-1.9z"/><path d="M19 15l.8 2.2L22 18l-2.2.8L19 21l-.8-2.2L16 18l2.2-.8z"/>',
    layers: '<path d="M12 2L2 7l10 5 10-5-10-5z"/><path d="M2 17l10 5 10-5"/><path d="M2 12l10 5 10-5"/>',
    arrow: '<path d="M5 12h14M13 6l6 6-6 6"/>',
    download: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="M7 10l5 5 5-5"/><path d="M12 15V3"/>',
    save: '<path d="M19 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11l5 5v11a2 2 0 0 1-2 2z"/><path d="M17 21v-8H7v8M7 3v5h8"/>',
    file: '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><path d="M14 2v6h6M8 13h8M8 17h5"/>',
    up: '<path d="M7 10v11H3V10z"/><path d="M7 10l4-8a3 3 0 0 1 3 3v4h5.5a2 2 0 0 1 2 2.3l-1.4 8A2 2 0 0 1 18.1 21H7"/>',
    down: '<path d="M17 14V3h4v11z"/><path d="M17 14l-4 8a3 3 0 0 1-3-3v-4H4.5a2 2 0 0 1-2-2.3l1.4-8A2 2 0 0 1 5.9 3H17"/>',
    first: '<path d="M18 5v14L8 12z" fill="currentColor" stroke="none"/><path d="M6 5v14"/>',
    prev: '<path d="M18 5v14L7 12z" fill="currentColor" stroke="none"/>',
    next: '<path d="M6 5v14l11-7z" fill="currentColor" stroke="none"/>',
    pause: '<path d="M7 5h3v14H7zM14 5h3v14h-3z" fill="currentColor" stroke="none"/>',
    fit: '<path d="M15 3h6v6M9 21H3v-6M21 3l-7 7M3 21l7-7"/>',
    reset: '<path d="M3 12a9 9 0 1 0 3-6.7L3 8"/><path d="M3 3v5h5"/>',
    help: '<circle cx="12" cy="12" r="9"/><path d="M9.5 9a2.5 2.5 0 0 1 4.9.8c0 1.7-2.4 2.2-2.4 3.7M12 17h.01"/>',
  };
  const icon = (name, size = 16) =>
    `<svg class="ico" viewBox="0 0 24 24" width="${size}" height="${size}" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${P[name] || ""}</svg>`;

  // „დაწვრილებით“ — გრძელი ახსნა ჩაკეცილად (html უკვე უსაფრთხოა)
  const hint = (html) => `<details class="hint"><summary>${icon("help", 14)} დაწვრილებით</summary><p class="muted small">${html}</p></details>`;

  // ---------- toast ----------
  let toastBox = null;
  function toast(text, kind = "info", ms) {
    if (!toastBox) {
      toastBox = document.createElement("div");
      toastBox.className = "toasts";
      toastBox.setAttribute("aria-live", "polite");
      document.body.appendChild(toastBox);
    }
    const t = document.createElement("div");
    t.className = `toast ${kind}`;
    t.setAttribute("role", kind === "error" ? "alert" : "status");
    const ic = { ok: "check", error: "alert", warn: "alert", info: "info" }[kind] || "info";
    t.innerHTML = `<span class="toast-ico">${icon(ic, 18)}</span><div class="toast-text"></div>
      <button class="toast-x" aria-label="დახურვა">${icon("x", 14)}</button>`;
    t.querySelector(".toast-text").textContent = String(text ?? "");
    const close = () => { t.classList.add("out"); setTimeout(() => t.remove(), 200); };
    t.querySelector(".toast-x").onclick = close;
    toastBox.appendChild(t);
    setTimeout(close, ms ?? (kind === "error" ? 8000 : 4500));
    return close;
  }

  // ---------- დადასტურება (Promise<bool>) ----------
  function confirm(text, opts = {}) {
    return new Promise((resolve) => {
      const d = document.createElement("dialog");
      d.className = "modal";
      const danger = opts.danger ?? /წავშალ|წაშლ|ამოვიღო/.test(text);
      d.innerHTML = `
        <form method="dialog" class="modal-body">
          <div class="modal-ico ${danger ? "bad" : ""}">${icon(danger ? "trash" : "info", 20)}</div>
          <div class="modal-main">
            <h3 class="modal-title">${escT(opts.title || (danger ? "დარწმუნებული ხართ?" : "დადასტურება"))}</h3>
            <p class="modal-text"></p>
          </div>
          <div class="modal-actions">
            <button value="cancel" class="ghost" type="submit">${escT(opts.cancelText || "გაუქმება")}</button>
            <button value="ok" class="primary ${danger ? "danger-solid" : ""}" type="submit">${escT(opts.okText || (danger ? "წაშლა" : "კარგი"))}</button>
          </div>
        </form>`;
      d.querySelector(".modal-text").textContent = text;
      document.body.appendChild(d);
      d.addEventListener("close", () => { resolve(d.returnValue === "ok"); d.remove(); });
      d.addEventListener("click", (e) => { if (e.target === d) d.close("cancel"); });
      d.showModal();
      d.querySelector('button[value="ok"]').focus();
    });
  }

  // ---------- ჩამოსაშლელი მენიუ (⋯) ----------
  // მარკაპი: <div class="dd"><button class="icon-btn" data-dd aria-haspopup="menu" aria-expanded="false">…</button>
  //          <div class="dd-menu" role="menu" hidden><button role="menuitem">…</button></div></div>
  function closeMenus(except) {
    document.querySelectorAll(".dd-menu:not([hidden])").forEach((m) => {
      if (m === except) return;
      m.hidden = true;
      m.parentElement.querySelector("[data-dd]")?.setAttribute("aria-expanded", "false");
    });
  }
  const ddHTML = (items) => `<div class="dd">
      <button class="icon-btn" type="button" data-dd aria-haspopup="menu" aria-expanded="false" aria-label="მეტი მოქმედება" title="მეტი">${icon("dots", 18)}</button>
      <div class="dd-menu" role="menu" hidden>${items.filter(Boolean).join("")}</div></div>`;
  const ddItem = (attrs, ic, label, danger) =>
    `<button type="button" role="menuitem" class="dd-item ${danger ? "danger" : ""}" ${attrs}>${icon(ic)}<span>${escT(label)}</span></button>`;
  document.addEventListener("click", (e) => {
    const t = e.target.closest("[data-dd]");
    if (t) {
      const m = t.parentElement.querySelector(".dd-menu");
      const open = m.hidden;
      closeMenus(m);
      m.hidden = !open;
      t.setAttribute("aria-expanded", String(open));
      if (open) {
        m.classList.remove("up");
        const r = m.getBoundingClientRect();
        if (r.bottom > window.innerHeight - 8 && r.height < t.getBoundingClientRect().top) m.classList.add("up");
        m.querySelector("[role=menuitem]")?.focus();
      }
      return;
    }
    closeMenus();
  });
  document.addEventListener("keydown", (e) => {
    const m = e.target.closest?.(".dd-menu");
    if (e.key === "Escape") {
      const open = document.querySelector(".dd-menu:not([hidden])");
      if (open) { closeMenus(); open.parentElement.querySelector("[data-dd]")?.focus(); }
      closeDrawer();
    } else if (m && (e.key === "ArrowDown" || e.key === "ArrowUp")) {
      e.preventDefault();
      const items = [...m.querySelectorAll("[role=menuitem]")];
      const i = items.indexOf(document.activeElement);
      items[(i + (e.key === "ArrowDown" ? 1 : -1) + items.length) % items.length]?.focus();
    }
  });

  // ---------- ქვე-ჩანართები ----------
  // ღილაკები: <div class="subtabs" data-subtabs="voices"><button data-subtab="listen">…</button></div>
  // პანელები: გვერდის შიგნით ნებისმიერი ელემენტი data-pane="listen" (დინამიკურადაც) — CSS მალავს სხვებს.
  function setSub(page, key, silent) {
    const bar = document.querySelector(`[data-subtabs="${page}"]`);
    if (!bar) return;
    const btn = bar.querySelector(`[data-subtab="${key}"]`) || bar.querySelector("[data-subtab]");
    key = btn.dataset.subtab;
    bar.querySelectorAll("[data-subtab]").forEach((b) => {
      const on = b === btn;
      b.setAttribute("aria-selected", String(on));
      b.tabIndex = on ? 0 : -1;
    });
    document.getElementById(`tab-${page}`)?.setAttribute("data-sub-active", key);
    store.set(`sub-${page}`, key);
    updateTitle();
    if (!silent) document.dispatchEvent(new CustomEvent("subtab", { detail: { page, key } }));
  }
  const getSub = (page) => document.getElementById(`tab-${page}`)?.dataset.subActive || null;
  function initSubtabs() {
    document.querySelectorAll("[data-subtabs]").forEach((bar) => {
      const page = bar.dataset.subtabs;
      bar.setAttribute("role", "tablist");
      bar.querySelectorAll("[data-subtab]").forEach((b) => {
        b.setAttribute("role", "tab");
        b.type = "button";
        b.addEventListener("click", () => setSub(page, b.dataset.subtab));
      });
      bar.addEventListener("keydown", (e) => {
        if (!["ArrowLeft", "ArrowRight"].includes(e.key)) return;
        const bs = [...bar.querySelectorAll("[data-subtab]")];
        const i = bs.indexOf(document.activeElement);
        if (i < 0) return;
        const n = bs[(i + (e.key === "ArrowRight" ? 1 : -1) + bs.length) % bs.length];
        n.focus(); n.click();
      });
      setSub(page, store.get(`sub-${page}`), true);
    });
  }

  // ---------- ნავიგაცია, სათაური, გვერდითი მენიუ ----------
  const navBtn = (page) => document.querySelector(`.tabs button[data-tab="${page}"]`);
  function go(page, sub) {
    navBtn(page)?.click();
    if (sub) setSub(page, sub);
    document.querySelector("main")?.scrollTo?.(0, 0);
    window.scrollTo(0, 0);
  }
  function currentPage() {
    return document.querySelector(".tabs button[aria-current='page']")?.dataset.tab
      || document.querySelector(".tab:not([hidden])")?.id.replace("tab-", "") || "overview";
  }
  function updateTitle() {
    const page = currentPage(), b = navBtn(page);
    if (!b) return;
    const label = b.querySelector(".nav-label")?.textContent.trim() || b.textContent.trim();
    const group = b.dataset.group || "";
    const sub = document.querySelector(`[data-subtabs="${page}"] [aria-selected="true"]`)?.textContent.trim();
    const t = document.getElementById("page-title");
    if (t) t.textContent = label;
    const c = document.getElementById("crumbs");
    if (c) c.innerHTML = [group !== label ? group : "", label, sub !== label ? sub : ""].filter(Boolean).map((x, i, a) =>
      `<span${i === a.length - 1 ? ' aria-current="location"' : ""}>${escT(x)}</span>`).join('<span class="sep">/</span>');
    document.title = `${label} · AI ხმები`;
  }

  const root = document.documentElement;
  function closeDrawer() { document.body.classList.remove("drawer-open"); document.getElementById("nav-open")?.setAttribute("aria-expanded", "false"); }
  function initShell() {
    document.querySelectorAll(".tabs button[data-tab]").forEach((b) => {
      b.addEventListener("click", () => { setTimeout(updateTitle, 0); closeDrawer(); });
    });
    // გვერდითი მენიუს ჩაკეცვა (დესკტოპი)
    const col = document.getElementById("sb-collapse");
    const applyCol = (on) => {
      root.classList.toggle("sb-collapsed", on);
      col?.setAttribute("aria-pressed", String(on));
      col?.setAttribute("title", on ? "მენიუს გაშლა" : "მენიუს ჩაკეცვა");
    };
    applyCol(store.get("sidebar-collapsed") === "1");
    col?.addEventListener("click", () => {
      const on = !root.classList.contains("sb-collapsed");
      applyCol(on); store.set("sidebar-collapsed", on ? "1" : "0");
    });
    // მობილური: drawer
    const open = document.getElementById("nav-open");
    open?.addEventListener("click", () => {
      const on = !document.body.classList.contains("drawer-open");
      document.body.classList.toggle("drawer-open", on);
      open.setAttribute("aria-expanded", String(on));
    });
    document.getElementById("scrim")?.addEventListener("click", closeDrawer);
    // თემა
    const tb = document.getElementById("theme-toggle");
    const isDark = () => root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    const syncTheme = () => {
      const d = isDark();
      tb?.setAttribute("aria-label", d ? "ნათელი თემა" : "მუქი თემა");
      tb?.setAttribute("title", d ? "ნათელი თემა" : "მუქი თემა");
      tb?.classList.toggle("is-dark", d);
    };
    tb?.addEventListener("click", () => {
      const next = isDark() ? "light" : "dark";
      root.dataset.theme = next; store.set("theme", next); syncTheme();
    });
    matchMedia("(prefers-color-scheme: dark)").addEventListener?.("change", syncTheme);
    syncTheme();
    initSubtabs();
    updateTitle();
  }

  // ზედა ზოლის სტატუსი: setStatus("jobs", {text, kind, title, go:["train"]}) ან null — დამალვა
  function setStatus(id, s) {
    const el = document.getElementById(`st-${id}`);
    if (!el) return;
    el.hidden = !s;
    if (!s) return;
    el.className = `status-pill ${s.kind || ""}`;
    el.querySelector(".pill-text").textContent = s.text;
    el.title = s.title || s.text;
    el.onclick = s.go ? () => go(...s.go) : null;
  }

  initShell();  // სკრიპტი <body>-ის ბოლოშია — DOM უკვე მზადაა
  // დანარჩენი სკრიპტების ჩატვირთვის შემდეგ — ქვე-ჩანართის მოვლენა მიმდინარე მდგომარეობისთვის
  window.addEventListener("DOMContentLoaded", () => {
    document.querySelectorAll("[data-subtabs]").forEach((bar) =>
      document.dispatchEvent(new CustomEvent("subtab", { detail: { page: bar.dataset.subtabs, key: getSub(bar.dataset.subtabs), initial: true } })));
  });

  return { toast, confirm, icon, hint, dd: ddHTML, ddItem, closeMenus, setSub, getSub, go, setStatus, updateTitle };
})();
