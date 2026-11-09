"""OmniVoice — მეორე ძრავი (არასავალდებულო): პანელის მხარე. სერვერი — tools/omni_server.py (omnivoice/.venv).

GET  /api/omni/status            დაყენებულია? ეშვება? მოდელი ჩატვირთულია? მოწყობილობა, რიგი
POST /api/omni/start             სერვერის გაშვება (თუ არ ეშვება)
GET  /api/omni/options           აღწერის ველები (სქესი, ასაკი, სიმაღლე, სტილი) ქართულად
POST /api/omni/design            {text, gender, age, pitch, style, seed} → ერთი ვარიანტი {id, url, seconds}
GET  /api/omni/design/{id}.wav   ვარიანტის მოსმენა
POST /api/omni/voices            {design_id, name, gender, age, character} → ხმა ბიბლიოთეკაში (სტატუსი „სატესტო“)

ხმა catalog.json-ში: {"engine": "omnivoice", ...}; ფაილები: voices/<id>/prompt.pt, ref.wav, ref.txt, omni.json.
სერვერი, თუ არ ეშვება, პანელი თავად უშვებს (`./aivoice omni serve`, ფონურად); ან systemd: aivoice-omni.service.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import threading
import time
import urllib.error
import urllib.request

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .config import ROOT, cfg, path as cpath
from .textnorm import ka_text

router = APIRouter(prefix="/api/omni")
OPY = ROOT / "omnivoice" / ".venv" / "bin" / "python"
DESIGNS = cpath("cache") / "omni" / "designs"
LOG = ROOT / "logs" / "omni_server.log"
PIDFILE = cpath("tmp") / "omni_server.pid"
SAMPLE_RATE = 24000
INSTALL_CMD = "./install.sh --omnivoice"

# აღწერის ველები: ქართული → OmniVoice-ის instruct (ზუსტად მისი სიტყვები; სხვას არ იღებს)
OPTIONS = {
    "gender": {"label": "სქესი", "items": {"female": "ქალი", "male": "კაცი"}},
    "age": {"label": "ასაკი", "items": {"child": "ბავშვი", "teenager": "მოზარდი", "young adult": "ახალგაზრდა",
                                         "middle-aged": "საშუალო ასაკის", "elderly": "ხანდაზმული"}},
    "pitch": {"label": "სიმაღლე", "items": {"very low pitch": "ძალიან დაბალი", "low pitch": "დაბალი", "moderate pitch": "საშუალო",
                                             "high pitch": "მაღალი", "very high pitch": "ძალიან მაღალი"}},
    "style": {"label": "სტილი", "items": {"": "ჩვეულებრივი", "whisper": "ჩურჩული"}},
}
_start_lock = threading.Lock()
_proc: dict = {"p": None}


def installed() -> bool:
    return OPY.exists()


def enabled() -> bool:
    return bool(cfg()["omnivoice"]["enabled"])


def _url(sub: str) -> str:
    return f"http://127.0.0.1:{int(cfg()['omnivoice']['port'])}{sub}"


def _http(method: str, sub: str, data: dict | None = None, timeout: float = 1800) -> tuple[int, str, bytes]:
    body = json.dumps(data, ensure_ascii=False).encode() if data is not None else None
    req = urllib.request.Request(_url(sub), data=body, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.headers.get("Content-Type", ""), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type", ""), e.read()


def server_status(timeout: float = 1.5) -> dict | None:
    try:
        code, _, b = _http("GET", "/status", timeout=timeout)
        return json.loads(b) if code == 200 else None
    except (OSError, ValueError):
        return None


def _pid_alive() -> bool:
    try:
        os.kill(int(PIDFILE.read_text().strip()), 0)
        return True
    except (OSError, ValueError):
        return False


def ensure_running(wait: float = 60) -> dict:
    """სერვერი უნდა ეშვებოდეს; თუ არა — ფონურად ვუშვებთ (`./aivoice omni serve`) და ველოდებით."""
    if not installed():
        raise HTTPException(503, f"OmniVoice არ არის დაყენებული — გაუშვით: {INSTALL_CMD}")
    if not enabled():
        raise HTTPException(503, "OmniVoice გამორთულია („პარამეტრები“ → OmniVoice)")
    st = server_status()
    if st:
        return st
    with _start_lock:
        st = server_status()
        if st:
            return st
        p = _proc["p"]
        # PID ცოცხალია → სერვერი ახლა თავიდან ეშვება (მოდელის გამოტვირთვა) ან systemd-ით — მეორეს არ ვუშვებთ
        if not _pid_alive() and (p is None or p.poll() is not None):
            LOG.parent.mkdir(parents=True, exist_ok=True)
            _proc["p"] = subprocess.Popen([str(ROOT / "aivoice"), "omni", "serve"], cwd=ROOT,
                                          stdout=LOG.open("a", encoding="utf-8"), stderr=subprocess.STDOUT,
                                          stdin=subprocess.DEVNULL, start_new_session=True,
                                          env={**os.environ, "PYTHONUNBUFFERED": "1"})
        t = time.time()
        while time.time() - t < wait:
            time.sleep(1)
            st = server_status()
            if st:
                return st
    raise HTTPException(503, f"OmniVoice-ის სერვერი ვერ ჩაირთო — იხ. {LOG.relative_to(ROOT)}")


def call(sub: str, data: dict, timeout: float = 1800) -> bytes:
    """POST სერვერზე (საჭიროებისას გაშვებით); შეცდომა → HTTPException ქართული ტექსტით."""
    ensure_running()
    for attempt in (1, 2):
        try:
            code, ctype, b = _http("POST", sub, data, timeout)
            break
        except OSError:  # სერვერი სწორედ ახლა გადაიტვირთა (უსაქმობის შემდეგ) — ერთხელ კიდევ
            if attempt == 2:
                raise HTTPException(503, "OmniVoice-ის სერვერი არ პასუხობს")
            ensure_running()
    if code != 200:
        try:
            detail = json.loads(b).get("detail")
        except ValueError:
            detail = b[:200].decode(errors="replace")
        raise HTTPException(code if 400 <= code < 600 else 502, detail if isinstance(detail, str) else f"OmniVoice: {detail}")
    return b


def voice_files_ok(voice_id: str) -> bool:
    return (ROOT / "voices" / voice_id / "prompt.pt").exists()


def synth_wav(v: dict, text: str, speed: float, volume: float, num_step: int | None = None) -> bytes:
    """შენახული ხმით გახმოვანება → WAV (24 kHz). text — უკვე გასწორებული (ka_text). num_step=None — კონფიგურაციიდან."""
    return call("/synth", {"voice": v["id"], "text": text, "speed": round(speed, 3), "volume": round(volume, 3), "seed": 0,
                           "num_step": num_step})


# ---------- API ----------
@router.get("/status")
def status():
    st = server_status() if installed() else None
    return {"installed": installed(), "enabled": enabled(), "running": bool(st), "server": st,
            "install_cmd": INSTALL_CMD, "port": int(cfg()["omnivoice"]["port"]),
            "voices": sum(1 for v in _catalog()["voices"] if v.get("engine") == "omnivoice")}


@router.post("/start")
def start():
    return ensure_running()


@router.get("/options")
def options():
    return OPTIONS


class DesignIn(BaseModel):
    text: str = Field(min_length=1, max_length=400)
    gender: str = "female"
    age: str = "young adult"
    pitch: str = "moderate pitch"
    style: str = ""
    seed: int = Field(0, ge=0, le=2**31 - 1)


def instruct_of(d) -> str:
    parts = []
    for k in ("gender", "age", "pitch", "style"):
        val = getattr(d, k)
        if val not in OPTIONS[k]["items"]:
            raise HTTPException(400, f"უცნობი მნიშვნელობა: {OPTIONS[k]['label']}")
        if val:
            parts.append(val)
    return ", ".join(parts)


@router.post("/design")
def design(d: DesignIn):
    meta = json.loads(call("/design", {"text": ka_text(d.text.strip()), "instruct": instruct_of(d), "seed": d.seed}))
    return {**meta, "url": f"/api/omni/design/{meta['id']}.wav"}


@router.get("/design/{name}")
def design_audio(name: str):
    f = DESIGNS / name
    if not re.fullmatch(r"[0-9a-f]{16}\.wav", name) or not f.exists():
        raise HTTPException(404)
    return FileResponse(f, media_type="audio/wav")


def _catalog() -> dict:
    try:
        return json.loads((ROOT / "voices" / "catalog.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"voices": []}


class VoiceIn(BaseModel):
    design_id: str = Field(pattern=r"^[0-9a-f]{16}$")
    id: str | None = Field(None, pattern=r"^[a-z0-9][a-z0-9_]{1,39}$")
    name: str = Field(min_length=1, max_length=60)
    gender: str = Field("", max_length=20)
    age: str = Field("", max_length=20)
    character: str = Field("", max_length=200)


@router.post("/voices")
def save_voice(b: VoiceIn):
    """ვარიანტი → ხმა ბიბლიოთეკაში (სატესტო). ტემბრი ამის შემდეგ ფიქსირებულია (prompt.pt)."""
    from app import main as M
    cat = _catalog()
    ids = {v["id"] for v in cat["voices"]}
    vid = b.id
    if not vid:
        n = 1
        while f"omni_{n:02}" in ids or (M.VOICES / f"omni_{n:02}").exists():
            n += 1
        vid = f"omni_{n:02}"
    if vid in ids or (M.VOICES / vid).exists():
        raise HTTPException(409, "ასეთი id უკვე არსებობს")
    info = json.loads(call("/voices", {"design_id": b.design_id, "voice_id": vid}))
    cat = _catalog()  # გენერაციის დროს შეიძლება შეიცვალა
    cat["voices"].append({
        "id": vid, "name": b.name.strip(), "gender": b.gender.strip(), "age": b.age.strip(), "character": b.character.strip(),
        "status": "testing", "engine": "omnivoice",
        "source": f"OmniVoice: ხმა აღწერით — „{info.get('instruct') or 'ავტომატური'}“, seed {info.get('seed')} (მხოლოდ არაკომერციულად)",
        "license": "cc-by-nc",  # OmniVoice-ის წონები CC-BY-NC-ია (კოდი — Apache-2.0)
    })
    M.write_json(M.VOICES / "catalog.json", cat)
    return {"ok": True, "id": vid}
