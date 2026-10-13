"""„წვრთნის“ ჩანართი: ხმის შექმნა ნარევიდან, მომზადება, გაშვება (Kaggle / ეს კომპიუტერი), ლოდინი, გაჩერება.

ყველა მოქმედება ეშვება `./aivoice train ...`-ით (tools/train_ctl.py) ცალკე პროცესად — პანელის
გადატვირთვა მას არ აჩერებს. ლოგი: <paths.runs>/<slot>/ctl.log
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from .config import ROOT, path as cpath

router = APIRouter(prefix="/api/train")
JOBS = ROOT / "data" / "kaggle_jobs"
SLOT_RE = re.compile(r"[a-z][a-z0-9_]{1,30}")


def read_json(p: Path, default=None):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def alive(pid) -> bool:
    if not pid:
        return False
    try:
        os.kill(int(pid), 0)
        return True
    except (OSError, ValueError):
        return False


def tail(p: Path, n=4000) -> str:
    if not p.exists():
        return ""
    with p.open("rb") as f:
        f.seek(max(0, p.stat().st_size - n))
        return f.read().decode("utf-8", "replace").replace("\r", "\n")


def spawn(slot: str, *args: str):
    """./aivoice train <args> — ცალკე სესიაში (პანელის გადატვირთვისას არ ჩერდება)."""
    run = cpath("runs") / slot
    run.mkdir(parents=True, exist_ok=True)
    log = (run / "ctl.log").open("a", encoding="utf-8")
    log.write(f"\n==== პანელიდან: ./aivoice train {' '.join(args)}\n")
    log.flush()
    p = subprocess.Popen([str(ROOT / "aivoice"), "train", *args], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True)
    threading.Thread(target=p.wait, daemon=True).start()  # zombie-ს არ ვტოვებთ (თორემ „მიმდინარეობს“ სამუდამოდ)
    c = read_json(run / "ctl.json", {}) or {}
    c.update(pid=p.pid, action=args[0])
    (run / "ctl.json").write_text(json.dumps(c, ensure_ascii=False, indent=1), encoding="utf-8")
    return p.pid


def check_slot(slot: str):
    if not SLOT_RE.fullmatch(slot) or not (JOBS / f"{slot}.json").exists():
        raise HTTPException(404, "ასეთი ხმა არ არის")


@router.get("/jobs")
def jobs():
    cat = {v["id"]: v for v in (read_json(ROOT / "voices" / "catalog.json", {"voices": []}) or {"voices": []})["voices"]}
    out = []
    for p in sorted(JOBS.glob("*.json")):
        slot, j = p.stem, read_json(p, {}) or {}
        run = cpath("runs") / slot
        st = read_json(run / "status.json", {}) or {}
        c = read_json(run / "ctl.json", {}) or {}
        loc = read_json(run / "local" / "state.json", {}) or {}
        target = c.get("target") or st.get("device")
        epoch = loc.get("epoch") if target == "local" and loc.get("epoch") is not None else st.get("epoch")
        ds = Path(j.get("source_dir") or j.get("dataset_dir") or "")
        out.append({
            "slot": slot, "name": cat.get(slot, {}).get("name") or slot, "design": j.get("design"),
            "kind": "base" if (j.get("num_speakers") or 1) > 1 else "voice",
            "stage": c.get("stage") or st.get("status"), "target": target, "note": c.get("note"),
            "error": c.get("error") or st.get("error"), "busy": alive(c.get("pid")), "action": c.get("action"),
            "epoch": (epoch + 1) if epoch is not None else 0, "max_epochs": j.get("max_epochs"),
            "kaggle_state": st.get("kaggle_state"), "local": {k: loc.get(k) for k in ("running", "step_in_epoch", "steps_per_epoch", "epoch_hours")},
            "clips": sum(1 for _ in (ds / "metadata.csv").open(encoding="utf-8")) if (ds / "metadata.csv").exists() else 0,
            "init": (Path(j["init_dir"]) / j.get("warmstart", "init.ckpt")).exists() if j.get("init_dir") else None,
            "ready": cat.get(slot, {}).get("status") == "ready", "updated": c.get("updated") or st.get("updated"),
        })
    return {"jobs": out}


class NewVoice(BaseModel):
    slot: str = Field(pattern=r"^[a-z][a-z0-9_]{1,30}$")
    design: str
    name: str = ""
    gender: str = ""
    epochs: int = Field(150, ge=10, le=2000)


@router.post("/new")
def new(v: NewVoice):
    r = subprocess.run([str(ROOT / "aivoice"), "train", "new", v.slot, "--design", v.design, "--name", v.name or v.design,
                        "--gender", v.gender, "--epochs", str(v.epochs)], cwd=ROOT, capture_output=True, text=True)
    if r.returncode != 0:
        raise HTTPException(400, (r.stderr or r.stdout).strip().splitlines()[-1] if (r.stderr or r.stdout).strip() else "შეცდომა")
    return {"ok": True}


@router.post("/{slot}/prepare")
def prepare(slot: str):
    check_slot(slot)
    return {"pid": spawn(slot, "prepare", slot)}


class StartIn(BaseModel):
    on: str = Field(pattern=r"^(kaggle|local)$")
    epochs: int | None = Field(None, ge=10, le=2000)


@router.post("/{slot}/start")
def start(slot: str, s: StartIn):
    check_slot(slot)
    args = ["start", slot, "--on", s.on, "--wait"]
    if s.epochs:
        args += ["--epochs", str(s.epochs)]
    return {"pid": spawn(slot, *args)}


@router.post("/{slot}/{action}")
def act(slot: str, action: str):
    check_slot(slot)
    if action not in ("stop", "fetch", "wait"):
        raise HTTPException(404)
    if action == "stop":
        c = read_json(cpath("runs") / slot / "ctl.json", {}) or {}
        if c.get("action") in ("prepare", "start", "wait") and alive(c.get("pid")):
            try:
                os.killpg(int(c["pid"]), 15)  # მომზადება/ლოდინი — ჩვენი პროცესი
            except OSError:
                pass
    return {"pid": spawn(slot, action, slot)}


@router.get("/{slot}/log")
def log(slot: str):
    check_slot(slot)
    run = cpath("runs") / slot
    return {"ctl": tail(run / "ctl.log"), "train": tail(run / "local" / "train.log", 2500)}
