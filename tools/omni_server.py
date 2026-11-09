"""OmniVoice-ის სერვერი — მეორე ძრავი (ეშვება omnivoice/.venv-ში): ./aivoice omni serve

k2-fsa/OmniVoice: diffusion-LM, ქართული (language="ka"), 24 kHz. კოდი Apache-2.0, წონები CC-BY-NC (არაკომერციული).
ხმა იქმნება აღწერით (instruct: სქესი, ასაკი, სიმაღლე, ჩურჩული) და ინახება „პრომპტად“ (voices/<id>/prompt.pt) —
შემდეგ ის ტემბრი ფიქსირებულია, როგორც Piper-ის ხმა. პანელი (app/omni.py) უკავშირდება 127.0.0.1:<omnivoice.port>-ზე.

GET  /status    ჩატვირთულია? მოწყობილობა, რიგი, დაკავებულია
POST /design    {text, instruct, seed, num_step} → {id, seconds}   ფაილი: <cache>/omni/designs/<id>.wav (+ .json)
POST /synth     {voice, text, speed, volume, seed, num_step} → audio/wav (შენახული ხმით)
POST /voices    {design_id, voice_id} → voices/<id>/prompt.pt, ref.wav, ref.txt, omni.json

- ტექსტს პანელი ამზადებს (app.textnorm.ka_text) — აქ უკვე გასწორებული მოდის.
- მოდელი იტვირთება პირველ მოთხოვნაზე; idle_unload_min წუთის უსაქმობის შემდეგ პროცესი თავს თავიდან უშვებს
  (execv, იგივე PID და პორტი) — ასე GPU მთლიანად თავისუფლდება (CUDA-ს კონტექსტიც).
- ერთდროულად ერთი გენერაცია; დანარჩენი რიგში ელოდება.
- ნამდვილი ადამიანის ხმის კლონირება გამორთულია: ატვირთვის endpoint არ არსებობს, ხმა ინახება მხოლოდ
  ამავე სერვერის მიერ აღწერით შექმნილი ჩანაწერიდან (design id → სერვერის საკუთარი ფაილი).
"""

from __future__ import annotations

import gc
import io
import json
import os
import re
import secrets
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.config import cfg, path as cpath  # noqa: E402

OC = cfg()["omnivoice"]
HF = cpath("models") / "hf"
os.environ.setdefault("HF_HOME", str(HF))  # დიდი ფაილები — მონაცემების დისკზე
if (HF / "hub" / "models--k2-fsa--OmniVoice").is_dir():
    os.environ.setdefault("HF_HUB_OFFLINE", "1")  # უკვე ჩამოტვირთულია — ქსელს არ ვეკითხებით
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402
import soundfile as sf  # noqa: E402
from fastapi import FastAPI, HTTPException  # noqa: E402
from fastapi.responses import Response  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

MODEL_ID = "k2-fsa/OmniVoice"
LANG = "ka"
VOICES = ROOT / "voices"
DESIGNS = cpath("cache") / "omni" / "designs"
PIDFILE = cpath("tmp") / "omni_server.pid"
ID_RE = re.compile(r"[a-z0-9][a-z0-9_]{1,39}")
DESIGN_RE = re.compile(r"[0-9a-f]{16}")

app = FastAPI(title="aivoice-omni", docs_url=None, redoc_url=None)
S: dict = {"model": None, "device": None, "loaded_at": None, "load_seconds": None, "last_used": time.time(),
           "waiting": 0, "busy": None, "started": time.time(), "error": None}
_gen = threading.Lock()   # ერთი გენერაცია ერთდროულად
_prompts: dict = {}       # voice id → (mtime, VoiceClonePrompt)


def _device() -> str:
    import torch
    want = str(OC.get("device") or "auto").lower()
    if want in ("cuda", "auto") and torch.cuda.is_available():
        return "cuda:0"
    if want in ("mps", "auto") and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def model():
    """მოდელი (ზარმაცად: პირველ მოთხოვნაზე). გამოიძახება მხოლოდ _gen-ის ქვეშ."""
    if S["model"] is None:
        t = time.time()
        import torch
        from omnivoice import OmniVoice
        dev = _device()
        if dev == "cpu":
            torch.set_num_threads(max(1, (os.cpu_count() or 2) // 2))  # კომპიუტერი არ უნდა გაიჭედოს
        S["model"] = OmniVoice.from_pretrained(MODEL_ID, device_map=dev,
                                               dtype=torch.float16 if dev.startswith("cuda") else torch.float32)
        S.update(device=dev, loaded_at=time.time(), load_seconds=round(time.time() - t, 1))
        print(f"მოდელი ჩაიტვირთა: {dev}, {S['load_seconds']} წმ", flush=True)
    return S["model"]


class _Slot:
    """რიგი: ერთი გენერაცია ერთდროულად; S["waiting"] — რამდენი ელოდება."""

    def __init__(self, what: str):
        self.what = what

    def __enter__(self):
        S["waiting"] += 1
        _gen.acquire()
        S["waiting"] -= 1
        S["busy"] = self.what
        return self

    def __exit__(self, *exc):
        S["busy"] = None
        S["last_used"] = time.time()
        _gen.release()


def generate(text: str, seed: int, num_step: int | None, **kw) -> np.ndarray:
    import torch
    m = model()
    torch.manual_seed(int(seed))  # ერთი და იგივე seed → იგივე ვარიანტი
    t = time.time()
    try:
        a = m.generate(text=text, language=LANG, num_step=int(num_step or OC["num_step"]), **kw)[0]
    except RuntimeError as e:  # მაგ. GPU-ს მეხსიერება არ ეყო
        S["error"] = str(e)[:300]
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        raise HTTPException(500, f"OmniVoice: {str(e)[:200]}")
    S["error"] = None
    a = np.asarray(a, dtype=np.float32).reshape(-1)
    sec = len(a) / m.sampling_rate
    print(f"{S['busy']}: {sec:.1f} წმ აუდიო — {time.time() - t:.1f} წმ ({(time.time() - t) / max(sec, 0.1):.1f}× რეალური დრო)", flush=True)
    return a


def wav_bytes(a: np.ndarray, sr: int) -> bytes:
    buf = io.BytesIO()
    sf.write(buf, a, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


@app.get("/status")
def status():
    return {"ok": True, "pid": os.getpid(), "loaded": S["model"] is not None, "device": S["device"],
            "queue": S["waiting"], "busy": S["busy"], "load_seconds": S["load_seconds"],
            "idle_unload_min": OC["idle_unload_min"], "num_step": OC["num_step"], "allow_clone": bool(OC["allow_clone"]),
            "idle_seconds": round(time.time() - S["last_used"]), "error": S["error"]}


class DesignIn(BaseModel):
    text: str = Field(min_length=1, max_length=600)
    instruct: str = Field("", max_length=200)
    seed: int = Field(0, ge=0, le=2**31 - 1)
    num_step: int | None = Field(None, ge=4, le=64)


@app.post("/design")
def design(d: DesignIn):
    """ახალი ვარიანტი აღწერით — ფაილს id-ს სერვერი აძლევს (კლიენტი მისამართს ვერ მიუთითებს)."""
    with _Slot("design"):
        try:
            a = generate(d.text, d.seed, d.num_step, instruct=d.instruct or None)
        except ValueError as e:  # არასწორი instruct (OmniVoice ამოწმებს)
            raise HTTPException(400, str(e).split("\n")[0])
        sr = S["model"].sampling_rate
    did = secrets.token_hex(8)
    DESIGNS.mkdir(parents=True, exist_ok=True)
    for old in DESIGNS.glob("*.*"):  # შეუნახავი ვარიანტები ერთ კვირაში იშლება
        if time.time() - old.stat().st_mtime > 7 * 86400:
            old.unlink(missing_ok=True)
    sf.write(DESIGNS / f"{did}.wav", a, sr, subtype="PCM_16")
    meta = {"id": did, "text": d.text, "instruct": d.instruct, "seed": d.seed, "num_step": d.num_step or OC["num_step"],
            "sample_rate": sr, "seconds": round(len(a) / sr, 2), "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "generated_by": "omni_server"}
    (DESIGNS / f"{did}.json").write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    return meta


def prompt_for(voice: str):
    from omnivoice import VoiceClonePrompt
    f = VOICES / voice / "prompt.pt"
    if not ID_RE.fullmatch(voice) or not f.exists():
        raise HTTPException(404, "OmniVoice-ის ხმა ვერ მოიძებნა")
    m = f.stat().st_mtime
    hit = _prompts.get(voice)
    if not hit or hit[0] != m:
        _prompts[voice] = hit = (m, VoiceClonePrompt.load(str(f)))
    return hit[1]


class SynthIn(BaseModel):
    voice: str
    text: str = Field(min_length=1, max_length=4000)
    speed: float = Field(1.0, ge=0.5, le=2.0)
    volume: float = Field(1.0, ge=0.1, le=2.0)
    seed: int = Field(0, ge=0, le=2**31 - 1)
    num_step: int | None = Field(None, ge=4, le=64)


@app.post("/synth")
def synth(r: SynthIn):
    with _Slot("synth"):
        p = prompt_for(r.voice)
        a = generate(r.text, r.seed, r.num_step, voice_clone_prompt=p, speed=r.speed)
        sr = S["model"].sampling_rate
    a = np.clip(a * r.volume, -1.0, 1.0)
    return Response(wav_bytes(a, sr), media_type="audio/wav")


class VoiceIn(BaseModel):
    design_id: str
    voice_id: str


@app.post("/voices")
def create_voice(v: VoiceIn):
    """შენახული ხმა: მხოლოდ ამ სერვერის მიერ აღწერით შექმნილი ჩანაწერიდან (ატვირთული აუდიოდან — არასოდეს)."""
    if not DESIGN_RE.fullmatch(v.design_id) or not ID_RE.fullmatch(v.voice_id):
        raise HTTPException(400, "არასწორი id")
    wav, meta_f = DESIGNS / f"{v.design_id}.wav", DESIGNS / f"{v.design_id}.json"
    meta = json.loads(meta_f.read_text(encoding="utf-8")) if meta_f.exists() else {}
    if not wav.exists() or meta.get("generated_by") != "omni_server" or meta.get("id") != v.design_id:
        raise HTTPException(404, "ვარიანტი ვერ მოიძებნა (ხმა ინახება მხოლოდ აქ, აღწერით შექმნილი ვარიანტიდან)")
    d = VOICES / v.voice_id
    if (d / "prompt.pt").exists():
        raise HTTPException(409, "ასეთი ხმა უკვე არსებობს")
    with _Slot("voice"):
        p = model().create_voice_clone_prompt(ref_audio=str(wav), ref_text=meta["text"])
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "prompt.pt.tmp"
    p.save(str(tmp))
    tmp.replace(d / "prompt.pt")
    (d / "ref.wav").write_bytes(wav.read_bytes())
    (d / "ref.txt").write_text(meta["text"] + "\n", encoding="utf-8")
    info = {"engine": "omnivoice", "model": MODEL_ID, "instruct": meta.get("instruct"), "seed": meta.get("seed"),
            "design_id": v.design_id, "sample_rate": meta.get("sample_rate"), "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (d / "omni.json").write_text(json.dumps(info, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return {"ok": True, **info}


def _idle_watch():
    """უსაქმობისას მოდელის გამოტვირთვა: პროცესი თავს თავიდან უშვებს (GPU მთლიანად თავისუფლდება)."""
    limit = max(1, int(OC["idle_unload_min"])) * 60
    while True:
        time.sleep(15)
        if S["model"] is None or S["waiting"] or time.time() - S["last_used"] < limit:
            continue
        if not _gen.acquire(blocking=False):
            continue
        print(f"{limit // 60} წთ უსაქმოდ — მოდელი ამოიტვირთება (პროცესი თავიდან ეშვება)", flush=True)
        S["model"] = None
        _prompts.clear()
        gc.collect()
        sys.stdout.flush()
        os.execv(sys.executable, [sys.executable, *sys.argv])  # იგივე PID — systemd/PID-ფაილი სწორი რჩება


def main():
    import uvicorn
    try:
        os.nice(max(0, 10 - os.nice(0)))  # წინა პლანის საქმეები არ შეფერხდეს (execv-ის შემდეგ აღარ მატულობს)
    except OSError:
        pass
    PIDFILE.parent.mkdir(parents=True, exist_ok=True)
    PIDFILE.write_text(str(os.getpid()), encoding="utf-8")
    threading.Thread(target=_idle_watch, daemon=True, name="idle").start()
    uvicorn.run(app, host="127.0.0.1", port=int(OC["port"]), log_level="warning")


if __name__ == "__main__":
    main()
