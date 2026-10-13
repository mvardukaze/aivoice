"""„პარამეტრები“ და პირველი დაყენების ოსტატი.

/api/settings          — კონფიგურაციის სქემა + მიმდინარე მნიშვნელობები (app/config.py)
/api/settings (POST)   — შენახვა aivoice.toml-ში; ცვლილება ძალაში შედის პანელის გადატვირთვის შემდეგ
/api/settings/secret   — გასაღების ჩაწერა (მხოლოდ წერა: მნიშვნელობა უკან არასოდეს ბრუნდება)
/api/settings/restart  — პანელის გადატვირთვა (systemd / `aivoice serve` თავიდან უშვებს)
/api/setup/status      — დაყენების შემოწმება: ხელსაწყოები, საქაღალდეები, მოდელები, ანგარიშები
"""

from __future__ import annotations

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
                         "kaggle.com → Settings → API → „Generate New Token“. საჭიროა Kaggle-ზე წვრთნისთვის."),
    "HF_TOKEN": ("Hugging Face token", "secrets.env",
                 "huggingface.co → Settings → Access Tokens. საჭიროა მოდელების ჩამოსატვირთად (Read) ან ასატვირთად (Write)."),
    "MDC_API_KEY": ("Mozilla Data Collective API key", "secrets.env",
                    "მხოლოდ საბაზო მოდელის თავიდან წვრთნისთვის (Common Voice-ის ჩამოტვირთვა)."),
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


def _cmd(args, timeout=10) -> str:
    try:
        return subprocess.run(args, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


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
    pg = C.ROOT / "training" / "piper1-gpl"
    add("ხელსაწყოები", "Piper (ჩვენი შესწორებებით)", "ok" if (pg / "src").exists() else "warn",
        "" if (pg / "src").exists() else "./install.sh --training კლონავს და patches/piper1-gpl.patch-ს ადებს")

    des = C.path("designer") / "designer.onnx"
    add("მოდელები", "ხმის შემქმნელის მოდელი", "ok" if des.exists() else "missing",
        str(des) if des.exists() else "ჩამოტვირთვა Hugging Face-იდან (მოდელები → ჩამოტვირთვა)", "models")
    voices = [p for p in (C.ROOT / "voices").glob("*/*.onnx")]
    add("მოდელები", "მზა ხმები", "ok" if voices else "missing",
        ", ".join(p.parent.name for p in voices) or "ჯერ არცერთი — ჩამოტვირთვა Hugging Face-იდან", "models")

    for n, (t, _, h) in SECRETS.items():
        add("ანგარიშები", t, "ok" if secret_present(n) else ("warn" if n != "KAGGLE_API_TOKEN" else "missing"), h, "secrets")
    gc = C.cfg()["gcp"]["gcloud"]
    gpath = str(C.expand(gc)) if "/" in gc else (shutil.which(gc) or "")
    add("ანგარიშები", "Google Cloud (არასავალდებულო)", "ok" if gpath and Path(gpath).exists() else "warn",
        gpath or "საჭიროა მხოლოდ Google Cloud-ზე წვრთნისთვის: https://cloud.google.com/sdk/docs/install")

    order = {"missing": 0, "warn": 1, "ok": 2}
    return {"ready": all(i["state"] != "missing" for i in items), "configured": C.configured(),
            "items": items, "worst": min((order[i["state"]] for i in items), default=2)}
