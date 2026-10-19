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


def spawn(slot: str, *args: str, cmd: str = "train"):
    """./aivoice train|base <args> — ცალკე სესიაში (პანელის გადატვირთვისას არ ჩერდება)."""
    run = cpath("runs") / slot
    run.mkdir(parents=True, exist_ok=True)
    log = (run / "ctl.log").open("a", encoding="utf-8")
    log.write(f"\n==== პანელიდან: ./aivoice {cmd} {' '.join(args)}\n")
    log.flush()
    p = subprocess.Popen([str(ROOT / "aivoice"), cmd, *args], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
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
        target = c.get("target") or st.get("device")
        loc = read_json(run / ("gpu" if target == "gpu" else "local") / "state.json", {}) or {}
        epoch = loc.get("epoch") if target in ("local", "gpu") and loc.get("epoch") is not None else st.get("epoch")
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
            "ready": cat.get(slot, {}).get("status") in ("ready", "testing"), "voice_status": cat.get(slot, {}).get("status"), "updated": c.get("updated") or st.get("updated"),
            "base": j.get("base_meta"),
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


# ---------------- საბაზო მოდელი (tools/base_ctl.py, ./aivoice base ...) ----------------
def base_ctl():
    import sys
    if str(ROOT / "tools") not in sys.path:
        sys.path.insert(0, str(ROOT / "tools"))
    import base_ctl as b
    return b


def base_check(fn, *args):
    """base_ctl-ის შემოწმება — SystemExit(ტექსტი) → 400."""
    try:
        return fn(*args)
    except SystemExit as e:
        raise HTTPException(400, str(e.code))


def base_run(slot: str, *args: str, dry_run: bool = False) -> dict:
    """ფონური ./aivoice base …; dry_run — სინქრონულად, --dry-run-ით (არაფერს ქმნის/უშვებს), გამოსავალი პასუხში."""
    if not dry_run:
        return {"pid": spawn(slot, *args, cmd="base")}
    r = subprocess.run([str(ROOT / "aivoice"), "base", *args, "--dry-run"], cwd=ROOT, capture_output=True, text=True, timeout=120)
    return {"dry_run": (r.stdout + r.stderr).strip(), "rc": r.returncode}


def check_base_slot(slot: str) -> dict:
    j = read_json(JOBS / f"{slot}.json", {}) if SLOT_RE.fullmatch(slot) else None
    if not j or (j.get("num_speakers") or 1) <= 1:
        raise HTTPException(404, "ასეთი საბაზო მოდელი არ არის")
    return j


@router.get("/base/options")
def base_options():
    return base_ctl().options()


class NewBase(BaseModel):
    slot: str = Field(pattern=r"^base_ka_v\d{1,3}$")
    mode: str = Field(pattern=r"^(scratch|continue)$")
    from_model: str = ""
    hours: float = Field(40, ge=1, le=400)
    short: bool = False
    punct: bool = False
    clusters: bool = False
    selection: str = Field("", pattern=r"^[A-Za-z0-9_]{0,60}$")
    epochs: int = Field(..., ge=1, le=2000)
    on: str = Field("gcp", pattern=r"^(gcp|local)$")


@router.post("/base/new")
def base_new(v: NewBase, dry_run: bool = False):
    import argparse
    b = base_ctl()
    a = argparse.Namespace(slot=v.slot, mode=v.mode, from_=v.from_model or None, hours=v.hours, short=v.short, punct=v.punct,
                           clusters=v.clusters, selection=v.selection or None, epochs=v.epochs, on=v.on, force=False)
    plan = base_check(b.check_new, a)
    args = ["new", v.slot, "--mode", v.mode, "--hours", str(v.hours), "--epochs", str(v.epochs), "--on", v.on]
    args += ["--from", v.from_model] if v.mode == "continue" else []
    args += ["--selection", v.selection] if v.selection else []
    args += [f"--{k}" for k in ("short", "punct", "clusters") if getattr(v, k)]
    return {**base_run(v.slot, *args, dry_run=dry_run), "estimate": plan["estimate"], "max_epochs": plan["max_epochs"]}


@router.post("/base/{slot}/prepare")
def base_prepare(slot: str, dry_run: bool = False):
    check_base_slot(slot)
    return base_run(slot, "prepare", slot, dry_run=dry_run)


class StartBase(BaseModel):
    on: str = Field(pattern=r"^(gcp|local)$")


@router.post("/base/{slot}/start")
def base_start(slot: str, s: StartBase, dry_run: bool = False):
    check_base_slot(slot)
    base_check(base_ctl().check_start, slot, s.on)  # GPU / Google Cloud დაკავებულია / dataset / დინამიკების რიგი
    return base_run(slot, "start", slot, "--on", s.on, dry_run=dry_run)


@router.post("/base/{slot}/stop")
def base_stop(slot: str, dry_run: bool = False):
    check_base_slot(slot)
    if dry_run:
        return base_run(slot, "stop", slot, dry_run=True)
    c = read_json(cpath("runs") / slot / "ctl.json", {}) or {}
    if c.get("action") in ("new", "prepare", "start") and alive(c.get("pid")):
        try:
            os.killpg(int(c["pid"]), 15)  # შერჩევა / dataset / ატვირთვა / ლოკალური წვრთნა — ჩვენი პროცესი
        except OSError:
            pass
    return base_run(slot, "stop", slot)


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
    return {"ctl": tail(run / "ctl.log"), "train": tail(run / "local" / "train.log", 2500) or tail(run / "gpu" / "train.log", 2500)}
