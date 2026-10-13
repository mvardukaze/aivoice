"""aivoice.nikora.dev — Piper-ის ქართული ხმების სამუშაო პანელი.

ხმის ტემბრი ფიქსირებულია: თითო ხმა ცალკე გაწვრთნილი Piper-ის მოდელია.
ემოცია მხოლოდ სინთეზის პარამეტრებით იცვლება, შეზღუდულ ფარგლებში (EMOTION_LIMITS),
რომ ასაკი/სქესი ვერასდროს შეიცვალოს — videoka-ს DSP-ტემბრის შეცდომა აქ აღარ მეორდება.
"""

import datetime as dt
import hashlib
import json
import shutil
import subprocess
import wave
from functools import lru_cache
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from piper import PiperVoice, SynthesisConfig
from pydantic import BaseModel, Field

from app.dashboard import router as dash_router
from app.designer import router as designer_router
from app.recorder import router as rec_router
from app.textnorm import ka_text

from app.config import ROOT, path as cpath  # noqa: E402

VOICES = ROOT / "voices"
DATASETS = cpath("datasets")
TRAINING = ROOT / "training"
CACHE = cpath("cache")
CACHE.mkdir(parents=True, exist_ok=True)

# (min, max) — ემოციის ცვლილების დაშვებული დიაპაზონი
EMOTION_LIMITS = {
    "length_scale": (0.80, 1.25),
    "noise_scale": (0.30, 0.90),
    "noise_w_scale": (0.40, 1.10),
    "volume": (0.40, 1.40),
}

EMOTIONS = {
    "neutral": {"label": "ნეიტრალური", "length_scale": 1.00, "noise_scale": 0.667, "noise_w_scale": 0.80, "volume": 1.0},
    "calm":    {"label": "მშვიდი",     "length_scale": 1.12, "noise_scale": 0.50,  "noise_w_scale": 0.60, "volume": 0.85},
    "sad":     {"label": "სევდიანი",   "length_scale": 1.20, "noise_scale": 0.45,  "noise_w_scale": 0.55, "volume": 0.70},
    "happy":   {"label": "მხიარული",   "length_scale": 0.92, "noise_scale": 0.80,  "noise_w_scale": 0.95, "volume": 1.05},
    "excited": {"label": "აღელვებული", "length_scale": 0.85, "noise_scale": 0.85,  "noise_w_scale": 1.05, "volume": 1.20},
    "angry":   {"label": "გაბრაზებული", "length_scale": 0.90, "noise_scale": 0.75, "noise_w_scale": 1.00, "volume": 1.35},
    "quiet":   {"label": "ჩუმი, დაძაბული", "length_scale": 1.08, "noise_scale": 0.55, "noise_w_scale": 0.70, "volume": 0.50},
}

app = FastAPI(title="aivoice", docs_url="/api/docs", openapi_url="/api/openapi.json")
app.include_router(rec_router)
app.include_router(dash_router)
app.include_router(designer_router)


@app.middleware("http")
async def no_cache_static(request, call_next):
    # ბრაუზერი index.html/app.js-ის ძველ ვერსიას არ უნდა ინახავდეს (ახალი ველები "#undefined"-ად ჩანდა)
    response = await call_next(request)
    if not request.url.path.startswith("/api/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return default


def catalog() -> list[dict]:
    return read_json(VOICES / "catalog.json", {"voices": []})["voices"]


def find_voice(voice_id: str) -> dict:
    for v in catalog():
        if v["id"] == voice_id:
            return v
    raise HTTPException(404, "ხმა ვერ მოიძებნა")


@lru_cache(maxsize=8)
def load_voice(model_rel: str) -> PiperVoice:
    return PiperVoice.load(VOICES / model_rel)


def wav_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as w:
        return w.getnframes() / w.getframerate()


# ---------- API ----------

@app.get("/api/voices")
def api_voices():
    out = []
    for v in catalog():
        v = dict(v)
        if v.get("model"):
            model = VOICES / v["model"]
            v["available"] = model.exists()
            cfg = read_json(model.with_suffix(".onnx.json"), {}) if model.exists() else {}
            v["sample_rate"] = cfg.get("audio", {}).get("sample_rate")
            v["quality"] = cfg.get("audio", {}).get("quality")
            v["size_mb"] = round(model.stat().st_size / 1e6, 1) if model.exists() else None
        else:
            v["available"] = False
        out.append(v)
    return out


@app.get("/api/emotions")
def api_emotions():
    return {"presets": EMOTIONS, "limits": EMOTION_LIMITS}


class SynthRequest(BaseModel):
    voice: str
    text: str = Field(min_length=1, max_length=2000)
    length_scale: float = 1.0
    noise_scale: float = 0.667
    noise_w_scale: float = 0.8
    volume: float = 1.0


@app.post("/api/synthesize")
def api_synthesize(req: SynthRequest):
    v = find_voice(req.voice)
    if v.get("status") != "ready" or not v.get("model"):
        raise HTTPException(409, "ეს ხმა ჯერ არ არის მზად")

    params = {}
    for key, (lo, hi) in EMOTION_LIMITS.items():
        params[key] = round(min(hi, max(lo, getattr(req, key))), 3)

    text = ka_text(req.text.strip())
    key = hashlib.sha1(json.dumps([v["model"], text, params], ensure_ascii=False).encode()).hexdigest()[:16]
    out = CACHE / f"{key}.wav"
    if not out.exists():
        voice = load_voice(v["model"])
        tmp = out.with_suffix(".tmp")
        with wave.open(str(tmp), "wb") as w:
            voice.synthesize_wav(text, w, syn_config=SynthesisConfig(**params))
        tmp.rename(out)
    return {"url": f"/api/audio/{out.name}", "seconds": round(wav_seconds(out), 2), "params": params}


@app.get("/api/audio/{name}")
def api_audio(name: str):
    path = CACHE / name
    if "/" in name or not name.endswith(".wav") or not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="audio/wav")


@app.get("/api/roadmap")
def api_roadmap():
    return read_json(ROOT / "data" / "roadmap.json", {"stages": []})


@app.get("/api/datasets")
def api_datasets():
    """datasets/<name>/ — Piper-ის ფორმატი: metadata.csv + wav/. stats.json-ს pipeline წერს."""
    out = []
    for d in sorted(p for p in DATASETS.iterdir() if p.is_dir()):
        meta = d / "metadata.csv"
        stats = read_json(d / "stats.json", {})
        clips = sum(1 for _ in meta.open(encoding="utf-8")) if meta.exists() else 0
        out.append({"name": d.name, "clips": clips, **stats})
    return out


SOURCES = cpath("sources")
AGE_KA = {"teens": "თინეიჯერი", "twenties": "20-იანი", "thirties": "30-იანი", "fourties": "40-იანი",
          "fifties": "50-იანი", "sixties": "60-იანი", "seventies": "70-იანი", "eighties": "80-იანი"}


def suggest_slot(s: dict) -> str:
    """კანდიდატის სავარაუდო სლოტი catalog.json-ში — სქესის, ასაკის და F0-ის მიხედვით."""
    g, age, f0 = s.get("gender", ""), s.get("age", ""), s.get("f0_median") or 0
    if not g and f0:
        g = "male" if f0 < 160 else "female"
    old = age in ("sixties", "seventies", "eighties")
    if g.startswith("male"):
        if old or (age == "fifties" and f0 and f0 < 110):
            return "m_old_01"
        return "m_young_01" if age in ("teens", "twenties") else "m_adult_01"
    if g.startswith("female"):
        if old:
            return "f_old_01"
        return "f_young_01" if age in ("teens", "twenties") else "f_adult_01"
    return ""


@app.get("/api/sources")
def api_sources():
    out = []
    for d in sorted(p for p in SOURCES.iterdir() if p.is_dir()) if SOURCES.exists() else []:
        data = read_json(d / "speakers.json")
        if not data:
            continue
        decisions = read_json(d / "decisions.json", {})
        speakers = [s for s in data["speakers"] if "samples" in s]
        for s in speakers:
            dec = decisions.get(s["id"], {})
            s["verdict"], s["verdict_note"] = dec.get("verdict", ""), dec.get("note", "")
            s.pop("client_id", None)
            s["slot"] = dec.get("slot") or suggest_slot(s)
            s["age_ka"] = AGE_KA.get(s.get("age", ""), s.get("age", ""))
        out.append({"name": d.name, "summary": data["summary"], "speakers": speakers})
    return out


@app.get("/api/sample/{source}/{speaker}/{name}")
def api_sample(source: str, speaker: str, name: str):
    path = (SOURCES / source / "samples" / speaker / name).resolve()
    if not path.is_relative_to(SOURCES.resolve()) or not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, media_type="audio/mpeg")


@app.get("/api/training")
def api_training():
    """training/runs/<name>/status.json — წვრთნის სკრიპტი წერს."""
    runs_dir = TRAINING / "runs"
    if not runs_dir.exists():
        return []
    runs = []
    for d in sorted(runs_dir.iterdir()):
        if not (d.is_dir() and (d / "status.json").exists()):
            continue
        r = {"name": d.name, **read_json(d / "status.json", {})}
        # სტატუსს წვრთნა ყოველ ~20 წამში წერს; დიდხანს უცვლელი "training" = პროცესი აღარ მუშაობს
        if r.get("status") in ("training", "exporting") and r.get("updated"):
            try:
                if (dt.datetime.now() - dt.datetime.fromisoformat(r["updated"])).total_seconds() > 900:
                    r["status"] = "paused"
            except ValueError:
                pass
        # ნიმუშები (meta.json, tools/make_previews.py / kaggle fetch) — ეტალონი პირველი, შემდეგ დროის მიხედვით
        extra = [read_json(p, {}) for p in sorted((d / "previews").glob("*/meta.json"))] if (d / "previews").exists() else []
        r["previews"] = sorted(r.get("previews", []) + [e for e in extra if e],
                               key=lambda p: (not p.get("reference"), p.get("time", "")))
        runs.append(r)
    order = {"training": 0, "exporting": 0, "failed": 1, "done": 2}
    runs.sort(key=lambda r: (order.get(r.get("status"), 3), r.get("started", "")))
    return {"runs": runs, "queue": read_json(ROOT / "data" / "train_queue.json", {})}


@app.get("/api/preview/{slot}/{n}/{name}")
def api_preview(slot: str, n: str, name: str):
    path = (TRAINING / "runs" / slot / "previews" / n / name).resolve()
    if not path.is_relative_to((TRAINING / "runs").resolve()) or not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, media_type="audio/wav")


def _run(cmd: list[str], timeout=10) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


@lru_cache(maxsize=1)
def _static_env() -> list[dict]:
    home = Path.home()
    train_py = ROOT / "training" / ".venv" / "bin" / "python"
    checks = []

    def add(group, name, ok, detail=""):
        checks.append({"group": group, "name": name, "ok": ok, "detail": detail})

    ff = _run(["ffmpeg", "-version"]).splitlines()
    add("აუდიო", "ffmpeg", bool(ff), ff[0].split(" Copyright")[0] if ff else "")
    add("აუდიო", "rubberband", bool(shutil.which("rubberband")))
    add("აუდიო", "demucs (ხმის გამოყოფა)", bool(shutil.which("demucs") or (home / ".local/bin/demucs").exists()))

    hub = home / ".cache/huggingface/hub"
    for m in ("large-v3", "small"):
        add("ამოცნობა", f"faster-whisper {m}", (hub / f"models--Systran--faster-whisper-{m}").exists())

    gpu = _run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"])
    add("წვრთნა", "GPU", bool(gpu), gpu)
    torch_cuda = ""
    if train_py.exists():
        torch_cuda = _run([str(train_py), "-c", "import torch;print(torch.__version__, torch.cuda.is_available())"], 60)
    add("წვრთნა", "წვრთნის venv (torch + CUDA)", torch_cuda.endswith("True"), torch_cuda or "training/.venv არ არსებობს")
    lightning = _run([str(train_py), "-c", "import lightning;print(lightning.__version__)"], 60) if train_py.exists() else ""
    add("წვრთნა", "lightning", bool(lightning), lightning)
    ckpts = list((TRAINING / "checkpoints").glob("**/*.ckpt")) if (TRAINING / "checkpoints").exists() else []
    add("წვრთნა", "საწყისი checkpoint (.ckpt)", bool(ckpts), ", ".join(c.name for c in ckpts[:3]) or "არ გვაქვს")
    return checks


@app.get("/api/env")
def api_env(refresh: bool = False):
    if refresh:
        _static_env.cache_clear()
    du = shutil.disk_usage(ROOT)
    return {"checks": _static_env(), "disk_free_gb": round(du.free / 1e9)}


app.mount("/", StaticFiles(directory=ROOT / "app" / "static", html=True), name="static")
