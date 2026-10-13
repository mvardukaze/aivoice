"""მოდელების სინქრონიზაცია Hugging Face-თან პანელიდან (tools/models.py).

/api/models/status   — რეპოზიტორია ↔ ეს კომპიუტერი (ფაილების სია)
/api/models/{action} — pull / push (base: საბაზო checkpoint-ებიც) — ცალკე პროცესად, ლოგი: <runs>/_models/ctl.log
/api/models/log      — ლოგი + მიმდინარეობს თუ არა
"""

from __future__ import annotations

import json
import os
import subprocess
import threading

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from .config import ROOT, cfg, path as cpath

router = APIRouter(prefix="/api/models")
RUN = cpath("runs") / "_models"


def _pid() -> int | None:
    try:
        pid = int((RUN / "pid").read_text())
        os.kill(pid, 0)
        return pid
    except (FileNotFoundError, ValueError, OSError):
        return None


@router.get("/status")
def status():
    if not cfg()["hf"]["models_repo"]:
        return {"repo": "", "files": [], "error": "რეპოზიტორია არ არის მითითებული (პარამეტრები → Hugging Face)"}
    r = subprocess.run([str(ROOT / ".venv" / "bin" / "python"), str(ROOT / "tools" / "models.py"), "status", "--json"],
                       cwd=ROOT, capture_output=True, text=True, timeout=600)
    if r.returncode != 0:
        err = (r.stderr or r.stdout).strip().splitlines()
        return {"repo": cfg()["hf"]["models_repo"], "files": [], "error": err[-1] if err else "შეცდომა"}
    return {**json.loads(r.stdout), "busy": bool(_pid())}


class SyncIn(BaseModel):
    base: bool = False


@router.post("/{action}")
def sync(action: str, s: SyncIn):
    if action not in ("pull", "push"):
        raise HTTPException(404)
    if _pid():
        raise HTTPException(409, "სინქრონიზაცია უკვე მიმდინარეობს")
    RUN.mkdir(parents=True, exist_ok=True)
    args = [str(ROOT / "aivoice"), "models", action] + (["--base"] if s.base else [])
    log = (RUN / "ctl.log").open("w", encoding="utf-8")
    log.write(f"==== ./aivoice models {action}{' --base' if s.base else ''}\n")
    log.flush()
    p = subprocess.Popen(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                         start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    threading.Thread(target=p.wait, daemon=True).start()  # zombie-ს არ ვტოვებთ (თორემ „მიმდინარეობს“ სამუდამოდ)
    (RUN / "pid").write_text(str(p.pid))
    return {"pid": p.pid}


@router.get("/log")
def log():
    p = RUN / "ctl.log"
    return {"busy": bool(_pid()), "log": p.read_text(encoding="utf-8", errors="replace")[-6000:] if p.exists() else ""}
