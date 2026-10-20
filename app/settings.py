"""„პარამეტრები“ და პირველი დაყენების ოსტატი.

/api/settings          — კონფიგურაციის სქემა + მიმდინარე მნიშვნელობები (app/config.py)
/api/settings (POST)   — შენახვა aivoice.toml-ში; ცვლილება ძალაში შედის პანელის გადატვირთვის შემდეგ
/api/settings/secret   — გასაღების ჩაწერა (მხოლოდ წერა: მნიშვნელობა უკან არასოდეს ბრუნდება)
/api/settings/restart  — პანელის გადატვირთვა (systemd / `aivoice serve` თავიდან უშვებს)
/api/setup/status      — დაყენების შემოწმება: ხელსაწყოები, საქაღალდეები, მოდელები, ანგარიშები
/api/update            — ახალი ვერსია GitHub-ზე? (GET) / განახლება (POST, tools/update.py apply) / ლოგი
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from . import config as C

router = APIRouter(prefix="/api")
RESTART_CODE = 3  # `aivoice serve` ამ კოდზე თავიდან უშვებს; systemd (Restart=on-failure) — ასევე

# გასაღებები: სახელი → (სათაური, სად ინახება, ახსნა)
SECRETS = {
    "KAGGLE_API_TOKEN": ("Kaggle API token", "~/.kaggle/access_token",
                         "kaggle.com → Settings → API → „Generate New Token“. საჭიროა ახალი ხმების შესაქმნელად (Kaggle-ის უფასო GPU)."),
    "HF_TOKEN": ("Hugging Face token", "secrets.env",
                 "huggingface.co → Settings → Access Tokens. მოდელების რეპოზიტორია პირადია: ჩამოსატვირთად — Read "
                 "(წვდომას ავტორი გაძლევთ), ასატვირთად — Write."),
    "MDC_API_KEY": ("Mozilla Data Collective API key", "secrets.env",
                    "მხოლოდ საკუთარი საბაზო მოდელისთვის: Common Voice-ის ჩამოტვირთვა („მონაცემები“ → „ხარისხი“). mozilladatacollective.com → API key."),
}
LEGACY_ENV = Path.home() / ".config" / "aivoice" / "mdc.env"  # ძველი ადგილი (ამ კომპიუტერზე)


def _env_files() -> dict:
    out = {}
    for f in (LEGACY_ENV, C.SECRETS_FILE):
        if f.exists():
            for line in f.read_text(encoding="utf-8").splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    k, v = line.split("=", 1)
                    out[k.strip()] = v.strip()
    return out


def secret_present(name: str) -> bool:
    if name == "KAGGLE_API_TOKEN":
        return (Path.home() / ".kaggle" / "access_token").exists() or (Path.home() / ".kaggle" / "kaggle.json").exists() \
            or bool(os.environ.get(name))
    env = _env_files()
    return bool(os.environ.get(name) or env.get(name) or (name == "HF_TOKEN" and env.get("HF_API_KEY")))


def _write_private(p: Path, text: str):
    p.parent.mkdir(parents=True, exist_ok=True)
    os.chmod(p.parent, 0o700)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.chmod(tmp, 0o600)
    tmp.replace(p)


@router.get("/settings")
def get_settings():
    c = C.cfg()
    sections = []
    for s, keys in C.SCHEMA.items():
        fields = []
        for k, (default, label, help_) in keys.items():
            v = c[s].get(k, default)
            f = {"key": k, "label": label, "help": help_, "value": v, "default": default,
                 "type": "int" if isinstance(default, int) and not isinstance(default, bool) else "text"}
            if s == "paths":
                p = C.expand(v)
                f.update(resolved=str(p), exists=p.exists())
            fields.append(f)
        sections.append({"key": s, "fields": fields})
    return {"file": str(C.CONFIG_FILE), "configured": C.configured(), "sections": sections,
            "secrets": [{"name": n, "label": t, "where": w, "help": h, "present": secret_present(n)}
                        for n, (t, w, h) in SECRETS.items()]}


class SettingsIn(BaseModel):
    values: dict[str, dict[str, str | int]]


@router.post("/settings")
def save_settings(body: SettingsIn):
    for s, vals in body.values.items():
        if s not in C.SCHEMA or any(k not in C.SCHEMA[s] for k in vals):
            raise HTTPException(400, f"უცნობი პარამეტრი: {s}")
    C.save(body.values)
    C.ensure_dirs()
    return {"ok": True, "restart_needed": True, "file": str(C.CONFIG_FILE)}


class SecretIn(BaseModel):
    name: str
    value: str


@router.post("/settings/secret")
def save_secret(body: SecretIn):
    if body.name not in SECRETS or not body.value.strip() or "\n" in body.value:
        raise HTTPException(400, "არასწორი გასაღები")
    v = body.value.strip()
    if body.name == "KAGGLE_API_TOKEN":
        _write_private(Path.home() / ".kaggle" / "access_token", v + "\n")
    else:
        env = {}
        if C.SECRETS_FILE.exists():
            for line in C.SECRETS_FILE.read_text(encoding="utf-8").splitlines():
                if "=" in line:
                    k, val = line.split("=", 1)
                    env[k.strip()] = val.strip()
        env[body.name] = v
        _write_private(C.SECRETS_FILE, "".join(f"{k}={val}\n" for k, val in env.items()))
    return {"ok": True, "present": True}


@router.post("/settings/restart")
def restart():
    threading.Thread(target=lambda: (time.sleep(0.5), os._exit(RESTART_CODE)), daemon=True).start()
    return {"ok": True}


# ---------- განახლება (tools/update.py) ----------
_update: dict = {"proc": None}
UPDATE_LOG = C.path("runs") / "_update.log"


@router.get("/update")
def update_check():
    try:
        r = subprocess.run([str(C.ROOT / ".venv" / "bin" / "python"), str(C.ROOT / "tools" / "update.py"), "check", "--json"],
                           cwd=C.ROOT, capture_output=True, text=True, timeout=90)
        return json.loads(r.stdout)
    except (OSError, subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        return {"ok": False, "error": f"შემოწმება ვერ მოხერხდა: {e}"}


@router.post("/update")
def update_apply():
    p = _update["proc"]
    if p and p.poll() is None:
        raise HTTPException(409, "განახლება უკვე მიმდინარეობს")
    UPDATE_LOG.parent.mkdir(parents=True, exist_ok=True)
    log = UPDATE_LOG.open("w", encoding="utf-8")
    _update["proc"] = subprocess.Popen([str(C.ROOT / "aivoice"), "update", "apply"], cwd=C.ROOT, stdout=log,
                                       stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
                                       env={**os.environ, "PYTHONUNBUFFERED": "1"})
    return {"ok": True}


@router.get("/update/log")
def update_log():
    p = _update["proc"]
    busy = bool(p and p.poll() is None)
    return {"busy": busy, "code": None if busy or not p else p.returncode,
            "log": UPDATE_LOG.read_text(encoding="utf-8", errors="replace")[-4000:] if UPDATE_LOG.exists() else ""}


def _cmd(args, timeout=10) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


# ქსელური შემოწმებები (Kaggle, Google Cloud) — უფასო, მხოლოდ კითხვა; 10 წთ ქეშდება, რომ ოსტატი სწრაფად გაიხსნას
_net_cache: dict = {}
NET_TTL = 600


def _cached(key, fn):
    hit = _net_cache.get(key)
    if hit and time.time() - hit[0] < NET_TTL:
        return hit[1]
    v = fn()
    _net_cache[key] = (time.time(), v)
    return v


def kaggle_base_check() -> tuple[str, str]:
    slug = C.cfg()["kaggle"]["base_dataset"]
    kg = str(C.ROOT / ".venv" / "bin" / "kaggle")

    def run():
        try:
            r = subprocess.run([kg, "datasets", "list", "--mine", "-s", slug, "--csv"], capture_output=True, text=True, timeout=30)
            u = next((l.split(":", 1)[1].strip() for l in _cmd([kg, "config", "view"], 20).splitlines() if "username" in l), "")
        except (OSError, subprocess.TimeoutExpired):
            return None
        if r.returncode != 0 or not u:
            return None
        return any(l.split(",", 1)[0].strip() == f"{u}/{slug}" for l in r.stdout.splitlines())

    have = _cached(("kaggle_base", slug), run)
    if have:
        return "ok", f"{slug} — თქვენს ანგარიშზეა"
    if have is False:
        return "ok", f"{slug} — შეიქმნება ავტომატურად პირველ წვრთნაზე (~800 MB ატვირთვა)"
    return "warn", "ვერ შევამოწმე (ქსელი ან token) — პირველ წვრთნაზე ავტომატურად შეიქმნება, თუ არ არის"


def gcp_checks(gpath: str) -> list[tuple[str, str, str]]:
    """(სახელი, მდგომარეობა, ახსნა) — ფასიანი არაფერი: მხოლოდ კონფიგურაციის და API/კვოტის წაკითხვა."""
    out = []
    acct = _cmd([gpath, "auth", "list", "--filter=status:ACTIVE", "--format=value(account)"], 20)
    imp = _cmd([gpath, "config", "get-value", "auth/impersonate_service_account"], 20)
    out.append(("Google Cloud: ანგარიში", "ok" if acct else "warn",
                ("შესულია" + (" (service account-ის impersonation)" if imp else "")) if acct
                else "gcloud auth login"))
    proj = str(C.cfg()["gcp"].get("project") or "").strip() or _cmd([gpath, "config", "get-value", "project"], 20)
    out.append(("Google Cloud: პროექტი", "ok" if proj else "warn",
                proj if proj else "gcloud config set project <ID> — ან „პარამეტრები → gcp → Google Cloud პროექტი“"))
    if not (acct and proj):
        return out

    def api_and_quota():
        en = _cmd([gpath, "services", "list", "--enabled", "--project", proj, "--filter=config.name=compute.googleapis.com",
                   "--format=value(config.name)"], 40)
        reg = (C.cfg()["gcp"]["regions"].split(",") or ["us-central1"])[0].strip()
        quota = l4 = None
        if en:
            try:
                info = json.loads(_cmd([gpath, "compute", "project-info", "describe", "--project", proj, "--format=json"], 40) or "{}")
                quota = next((q.get("limit") for q in info.get("quotas", []) if q.get("metric") == "GPUS_ALL_REGIONS"), None)
                rinfo = json.loads(_cmd([gpath, "compute", "regions", "describe", reg, "--project", proj, "--format=json"], 40) or "{}")
                l4 = next((q.get("limit") for q in rinfo.get("quotas", []) if q.get("metric") == "NVIDIA_L4_GPUS"), None)
            except json.JSONDecodeError:
                pass
        return bool(en), quota, l4, reg

    en, quota, l4, reg = api_and_quota()
    out.append(("Google Cloud: Compute Engine API", "ok" if en else "warn",
                "ჩართულია" if en else "ჩართეთ: console.cloud.google.com → APIs & Services → Compute Engine API (ან ვერ შევამოწმე)"))
    if en:
        ok = (quota or 0) >= 1 and (l4 or 0) >= 1
        out.append(("Google Cloud: GPU კვოტა (L4)", "ok" if ok else "warn",
                    f"GPUs (all regions) = {quota or 0:g} · NVIDIA L4 ({reg}) = {l4 or 0:g}"
                    + ("" if ok else " — მოითხოვეთ 1: IAM & Admin → Quotas (README → Google Cloud)")))
    return out


@router.get("/setup/status")
def setup_status():
    """პირველი დაყენების ოსტატი: რა არის მზად და რა აკლია (ok / warn / missing)."""
    items = []

    def add(group, name, state, detail="", action=None):
        items.append({"group": group, "name": name, "state": state, "detail": detail, "action": action})

    add("კონფიგურაცია", "aivoice.toml", "ok" if C.configured() else "missing",
        str(C.CONFIG_FILE) if C.configured() else "ჯერ არ არის — აირჩიეთ მონაცემების საქაღალდე და შეინახეთ", "settings")
    root = C.path("data_root")
    free = shutil.disk_usage(root).free / 1e9 if root.exists() else None
    add("კონფიგურაცია", "მონაცემების საქაღალდე", "ok" if root.exists() and os.access(root, os.W_OK) else "missing",
        f"{root}" + (f" · თავისუფალი {free:.0f} GB" if free is not None else ""), "settings")

    ff = _cmd(["ffmpeg", "-version"]).split("\n")[0]
    add("ხელსაწყოები", "ffmpeg", "ok" if ff else "missing", ff[:60] or "macOS: brew install ffmpeg · Linux: apt/dnf install ffmpeg")
    esp = _cmd(["espeak-ng", "--version"])
    add("ხელსაწყოები", "espeak-ng", "ok" if esp else "warn",
        esp[:60] or "Piper-ს თავისი აქვს; ცალკე საჭიროა მხოლოდ ტექსტის შესამოწმებლად (brew install espeak-ng)")
    tpy = C.ROOT / "training" / ".venv" / "bin" / "python"
    add("ხელსაწყოები", "წვრთნის გარემო (training/.venv)", "ok" if tpy.exists() else "warn",
        "" if tpy.exists() else "საჭიროა ხმის მომზადებისთვის/შემქმნელის ექსპორტისთვის — ./install.sh --training")
    # ჩვენი შესწორებები (piper_overlay/) დადებულია თუ არა წვრთნის გარემოს Piper-ზე
    cands = list((C.ROOT / "training" / ".venv").glob("lib*/python3*/site-packages/piper/train/vits/dataset.py"))
    cands += [C.ROOT / "training" / "piper1-gpl" / "src" / "piper" / "train" / "vits" / "dataset.py"]
    patched = any(c.exists() and "aivoice: with DDP" in c.read_text(encoding="utf-8", errors="replace") for c in cands)
    add("ხელსაწყოები", "Piper (ჩვენი შესწორებებით)", "ok" if patched else "warn",
        "" if patched else "./install.sh --training ადებს piper_overlay/-ს ოფიციალურ Piper-ზე")

    # ხმის შემქმნელი: ძველი (designer/), ახალი (designer_base_ka_v*) ან საჯარო (…_public) — რომელიმე საკმარისია
    dirs = [d for d in [C.path("designer"), *sorted(C.path("designer").parent.glob("designer_base_ka_v*"))]
            if (d / "designer.onnx").exists()]
    names = ", ".join("v1" if d.name == "designer" else d.name.removeprefix("designer_base_ka_").replace("_public", " საჯარო")
                      for d in dirs)
    add("მოდელები", "ხმის შემქმნელის მოდელი", "ok" if dirs else "missing",
        names if dirs else "ჩამოტვირთვა Hugging Face-იდან (მოდელები → ჩამოტვირთვა)", "models")
    voices = [p for p in (C.ROOT / "voices").glob("*/*.onnx")]
    add("მოდელები", "მზა ხმები", "ok" if voices else "missing",
        ", ".join(p.parent.name for p in voices) or "ჯერ არცერთი — ჩამოტვირთვა Hugging Face-იდან", "models")

    # ყველა ანგარიში არასავალდებულოა: მზა ხმები და ხმის შემქმნელი მათ გარეშეც მუშაობს
    for n, (t, _, h) in SECRETS.items():
        add("ანგარიშები", t, "ok" if secret_present(n) else "warn", h, "secrets")
    if secret_present("KAGGLE_API_TOKEN"):
        st, detail = kaggle_base_check()
        add("ანგარიშები", "Kaggle: დამხმარე dataset", st, detail)
    gc = C.cfg()["gcp"]["gcloud"]
    gpath = str(C.expand(gc)) if "/" in gc else (shutil.which(gc) or "")
    have_g = bool(gpath and Path(gpath).exists())
    add("ანგარიშები", "Google Cloud (არასავალდებულო)", "ok" if have_g else "warn",
        gpath if have_g else "საჭიროა მხოლოდ საბაზო მოდელის წვრთნისთვის (ფასიანი): https://cloud.google.com/sdk/docs/install · README → Google Cloud")
    if have_g:
        for name, st, detail in _cached(("gcp_checks", gpath, str(C.cfg()["gcp"].get("project") or "")),
                                        lambda: gcp_checks(gpath)):
            add("ანგარიშები", name, st, detail)

    order = {"missing": 0, "warn": 1, "ok": 2}
    return {"ready": all(i["state"] != "missing" for i in items), "configured": C.configured(),
            "items": items, "worst": min((order[i["state"]] for i in items), default=2)}
