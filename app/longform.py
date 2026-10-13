"""გრძელი ტექსტის გახმოვანება: აბზაცები (თითოს თავისი ხმა/ემოცია) → ერთი WAV/MP3 + SRT სუბტიტრები.

POST /api/longform            {blocks: [{text, voice, emotion}], pause, sentence_pause, mp3} → {id}
GET  /api/longform/{id}       მდგომარეობა: status, done/total (წინადადებები), seconds, files
GET  /api/longform/{id}/{f}   audio.wav / audio.mp3 / subtitles.srt

წინადადებები ცალ-ცალკე ხმოვანდება (ქეშით: იგივე წინადადება თავიდან აღარ ითვლება) — ამიტომ SRT-ის დრო ზუსტია.
ერთდროულად ერთი სამუშაო; დანარჩენი რიგში დგება.
"""

from __future__ import annotations

import hashlib
import json
import queue
import re
import shutil
import subprocess
import threading
import time
import wave
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .config import path as cpath
from .textnorm import ka_text

router = APIRouter(prefix="/api/longform")
OUT = cpath("cache") / "longform"
SENT_CACHE = cpath("cache") / "sentences"
MAX_CHARS = 100_000
_q: queue.Queue = queue.Queue()
_jobs: dict[str, dict] = {}

# წერტილი შემოკლების/ინიციალის შემდეგ წინადადებას არ ამთავრებს
_NO_SPLIT = re.compile(r"(?:^|[\s(])(?:ქ|მაგ|ტელ|დაახლ|სხვ|გვ|საუკ|ბ-ნ|ქ-ნ|ა\.შ|ე\.ი|ა\.წ|ძვ\.წ|ახ\.წ|წ|სთ|წთ|წმ|კმ|კგ|გრ|[ა-ჰA-Z])\.$")


def sentences(text: str) -> list[str]:
    """აბზაცი → წინადადებები (. ! ? … და ახალი ხაზი)."""
    out = []
    for line in text.splitlines():
        buf = ""
        for part in re.split(r"(?<=[.!?…])\s+", line.strip()):
            buf = f"{buf} {part}".strip() if buf else part
            if not _NO_SPLIT.search(buf):
                out.append(buf)
                buf = ""
        if buf:
            out.append(buf)
    return [s for s in out if re.search(r"[\wა-ჰ]", s)]


class Block(BaseModel):
    text: str = Field(min_length=1, max_length=20_000)
    voice: str
    emotion: str = "neutral"


class LongIn(BaseModel):
    blocks: list[Block] = Field(min_length=1, max_length=500)
    pause: float = Field(0.7, ge=0, le=5)           # აბზაცებს შორის, წმ
    sentence_pause: float = Field(0.25, ge=0, le=3)  # წინადადებებს შორის, წმ
    mp3: bool = True


def _synth_sentence(M, v: dict, text: str, params: dict) -> np.ndarray:
    norm = ka_text(text)
    key = hashlib.sha1(json.dumps([v["model"], norm, params], ensure_ascii=False).encode()).hexdigest()[:20]
    f = SENT_CACHE / f"{key}.npy"
    if f.exists():
        return np.load(f)
    from piper import SynthesisConfig
    voice = M.load_voice(v["model"])
    parts = [c.audio_int16_array for c in voice.synthesize(norm, syn_config=SynthesisConfig(**params))]
    a = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int16)
    SENT_CACHE.mkdir(parents=True, exist_ok=True)
    np.save(f, a)
    return a


def _ts(sec: float) -> str:
    ms = int(round(sec * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def _run(jid: str, req: LongIn):
    from app import main as M  # ხმების ფუნქციები (წრიული import-ის გარეშე)
    st = _jobs[jid]
    d = OUT / jid
    try:
        st["status"] = "running"
        plan = []
        for bi, b in enumerate(req.blocks):
            v = M.find_voice(b.voice)
            params = M.emotion_params(v, b.emotion) if b.emotion in M.EMOTIONS else M.voice_defaults(v)
            for s in sentences(b.text):
                plan.append((bi, v, s, params))
        st["total"] = len(plan)
        sr = None
        audio, srt, t = [], [], 0.0
        prev_block = None
        for i, (bi, v, s, params) in enumerate(plan):
            if st.get("cancel"):
                raise RuntimeError("გაუქმებულია")
            a = _synth_sentence(M, v, s, params)
            rate = M.read_json((M.VOICES / v["model"]).with_suffix(".onnx.json"), {}).get("audio", {}).get("sample_rate", 22050)
            if sr is None:
                sr = rate
            elif rate != sr:  # სხვა სიხშირის ხმა — მარტივი გადათვლა
                a = np.interp(np.linspace(0, len(a), int(len(a) * sr / rate), endpoint=False), np.arange(len(a)), a).astype(np.int16)
            if prev_block is not None:
                gap = req.pause if bi != prev_block else req.sentence_pause
                audio.append(np.zeros(int(gap * sr), dtype=np.int16))
                t += gap
            dur = len(a) / sr
            srt.append(f"{len(srt) + 1}\n{_ts(t)} --> {_ts(t + dur)}\n{s}\n")
            audio.append(a)
            t += dur
            prev_block = bi
            st.update(done=i + 1, seconds=round(t, 1))
        d.mkdir(parents=True, exist_ok=True)
        with wave.open(str(d / "audio.wav"), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sr or 22050)
            w.writeframes(np.concatenate(audio).tobytes() if audio else b"")
        (d / "subtitles.srt").write_text("\n".join(srt), encoding="utf-8")
        files = ["audio.wav", "subtitles.srt"]
        if req.mp3 and shutil.which("ffmpeg"):
            st["status"] = "encoding"
            r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(d / "audio.wav"), "-codec:a", "libmp3lame",
                                "-q:a", "2", str(d / "audio.mp3")], capture_output=True, text=True)
            if r.returncode == 0:
                files.insert(0, "audio.mp3")
        st.update(status="done", files=files, finished=time.time())
    except Exception as e:  # noqa: BLE001 — შეცდომა მომხმარებელს უნდა ეჩვენოს
        st.update(status="failed", error=str(getattr(e, "detail", e)))
    (OUT / jid).mkdir(parents=True, exist_ok=True)
    (OUT / jid / "state.json").write_text(json.dumps(st, ensure_ascii=False), encoding="utf-8")


def _worker():
    while True:
        jid, req = _q.get()
        _run(jid, req)


threading.Thread(target=_worker, daemon=True, name="longform").start()


@router.post("")
def start(req: LongIn):
    if sum(len(b.text) for b in req.blocks) > MAX_CHARS:
        raise HTTPException(400, f"ტექსტი ძალიან გრძელია (მაქს. {MAX_CHARS:,} სიმბოლო) — დაყავით ნაწილებად")
    from app import main as M
    for b in req.blocks:
        v = M.find_voice(b.voice)
        if v.get("status") not in ("ready", "testing") or not v.get("model"):
            raise HTTPException(409, f"ხმა „{v['name']}“ ჯერ არ არის მზად")
    jid = time.strftime("%Y%m%d-%H%M%S-") + hashlib.sha1(req.model_dump_json().encode()).hexdigest()[:6]
    _jobs[jid] = {"id": jid, "status": "queued", "done": 0, "total": 0, "seconds": 0, "queued": _q.qsize(), "started": time.time()}
    _q.put((jid, req))
    return {"id": jid}


def _state(jid: str) -> dict:
    if not re.fullmatch(r"[\w-]+", jid):
        raise HTTPException(404)
    if jid in _jobs:
        return _jobs[jid]
    p = OUT / jid / "state.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    raise HTTPException(404, "ვერ მოიძებნა")


@router.get("/{jid}")
def status(jid: str):
    return _state(jid)


@router.post("/{jid}/cancel")
def cancel(jid: str):
    _state(jid)["cancel"] = True
    return {"ok": True}


@router.get("/{jid}/{name}")
def file(jid: str, name: str):
    _state(jid)
    if name not in ("audio.wav", "audio.mp3", "subtitles.srt"):
        raise HTTPException(404)
    f = OUT / jid / name
    if not f.exists():
        raise HTTPException(404)
    media = {"wav": "audio/wav", "mp3": "audio/mpeg", "srt": "application/x-subrip"}[name.rsplit(".", 1)[1]]
    return FileResponse(f, media_type=media, filename=f"aivoice-{jid}.{name.rsplit('.', 1)[1]}")
