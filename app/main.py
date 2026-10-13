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
from app.models_api import router as models_router
from app.longform import router as longform_router
from app.quality_api import router as quality_router
from app.recorder import router as rec_router
from app.settings import router as settings_router
from app.train_api import router as train_router
from app.textnorm import LEXICON, LEXICON_DEFAULT, ka_text, lexicon

from app.config import ROOT, path as cpath  # noqa: E402

VOICES = ROOT / "voices"
DATASETS = cpath("datasets")
TRAINING = ROOT / "training"
CACHE = cpath("cache")
CACHE.mkdir(parents=True, exist_ok=True)

# (min, max) — ემოციის ცვლილების დაშვებული დიაპაზონი
EMOTION_LIMITS = {
    "length_scale": (0.80, 1.25),
    "noise_scale": (0.15, 0.90),
    "noise_w_scale": (0.40, 1.10),
    "volume": (0.40, 1.40),
}

# ემოციები ფარდობითია: ხმის საკუთარი ნორმა (onnx.json-ის "inference") × (პრესეტი / "neutral").
# მაგ. „ქალი 1“-ის ინტონაცია 0.31 → „მხიარული“ 0.31 × 0.80/0.667 ≈ 0.37
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
app.include_router(settings_router)
app.include_router(train_router)
app.include_router(models_router)
app.include_router(longform_router)
app.include_router(quality_router)


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

VOICE_STATUS = ("ready", "testing", "hidden", "planned")
PARAM_KEYS = ("length_scale", "noise_scale", "noise_w_scale", "volume")


def write_json(path: Path, data):
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(path)


def clamp(params: dict) -> dict:
    return {k: round(min(hi, max(lo, float(params[k]))), 3) for k, (lo, hi) in EMOTION_LIMITS.items()}


def voice_defaults(v: dict) -> dict:
    """ხმის ნორმა: Piper-ის onnx.json → "inference" (+ ხმამაღლობა catalog-იდან)."""
    inf = read_json((VOICES / v["model"]).with_suffix(".onnx.json"), {}).get("inference", {}) if v.get("model") else {}
    n = EMOTIONS["neutral"]
    return clamp({"length_scale": inf.get("length_scale", n["length_scale"]), "noise_scale": inf.get("noise_scale", n["noise_scale"]),
                  "noise_w_scale": inf.get("noise_w", n["noise_w_scale"]), "volume": v.get("volume", 1.0)})


def emotion_params(v: dict, emotion: str) -> dict:
    d, n, p = voice_defaults(v), EMOTIONS["neutral"], EMOTIONS[emotion]
    return clamp({k: d[k] * p[k] / n[k] for k in PARAM_KEYS})


@app.get("/api/voices")
def api_voices(all: bool = False):
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
            v["defaults"] = voice_defaults(v)
            v["emotions"] = {k: emotion_params(v, k) for k in EMOTIONS}
        else:
            v["available"] = False
        out.append(v)
    return out


class VoiceEdit(BaseModel):
    name: str | None = Field(None, min_length=1, max_length=60)
    gender: str | None = Field(None, max_length=20)
    age: str | None = Field(None, max_length=20)
    character: str | None = Field(None, max_length=200)
    status: str | None = None
    defaults: dict[str, float] | None = None  # ნორმა: length_scale, noise_scale, noise_w_scale, volume


@app.post("/api/voices/{voice_id}")
def api_voice_edit(voice_id: str, e: VoiceEdit):
    cat = read_json(VOICES / "catalog.json", {"voices": []})
    v = next((x for x in cat["voices"] if x["id"] == voice_id), None)
    if not v:
        raise HTTPException(404, "ხმა ვერ მოიძებნა")
    if e.status is not None and e.status not in VOICE_STATUS:
        raise HTTPException(400, "უცნობი სტატუსი")
    for k in ("name", "gender", "age", "character", "status"):
        if getattr(e, k) is not None:
            v[k] = getattr(e, k).strip()
    if e.defaults is not None:
        if not v.get("model") or not (VOICES / v["model"]).exists():
            raise HTTPException(409, "ხმის მოდელი არ არის")
        d = clamp({**voice_defaults(v), **{k: x for k, x in e.defaults.items() if k in PARAM_KEYS}})
        cp = (VOICES / v["model"]).with_suffix(".onnx.json")
        c = read_json(cp, {})
        c.setdefault("_inference_trained", c.get("inference", {}))  # პირველადი მნიშვნელობა — „დაბრუნებისთვის“
        c["inference"] = {"noise_scale": d["noise_scale"], "length_scale": d["length_scale"], "noise_w": d["noise_w_scale"]}
        write_json(cp, c)
        v["volume"] = d["volume"]
    write_json(VOICES / "catalog.json", cat)
    return {"ok": True}


@app.post("/api/voices/{voice_id}/reset")
def api_voice_reset(voice_id: str):
    """ნორმის დაბრუნება წვრთნისას დაყენებულზე."""
    v = find_voice(voice_id)
    cp = (VOICES / v["model"]).with_suffix(".onnx.json")
    c = read_json(cp, {})
    if c.get("_inference_trained"):
        c["inference"] = c.pop("_inference_trained")
        write_json(cp, c)
    cat = read_json(VOICES / "catalog.json", {"voices": []})
    for x in cat["voices"]:
        if x["id"] == voice_id:
            x.pop("volume", None)
    write_json(VOICES / "catalog.json", cat)
    return {"ok": True}


@app.delete("/api/voices/{voice_id}")
def api_voice_delete(voice_id: str):
    """კატალოგიდან ამოღება; ფაილები გადადის <data_root>/trash/-ში (არ იშლება საბოლოოდ)."""
    cat = read_json(VOICES / "catalog.json", {"voices": []})
    if not any(x["id"] == voice_id for x in cat["voices"]):
        raise HTTPException(404, "ხმა ვერ მოიძებნა")
    cat["voices"] = [x for x in cat["voices"] if x["id"] != voice_id]
    moved = None
    vdir = VOICES / voice_id
    if vdir.is_dir() and voice_id and "/" not in voice_id and ".." not in voice_id:
        trash = cpath("data_root") / "trash" / f"{voice_id}-{dt.datetime.now():%Y%m%d-%H%M%S}"
        trash.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(vdir), str(trash))
        moved = str(trash)
    write_json(VOICES / "catalog.json", cat)
    load_voice.cache_clear()
    return {"ok": True, "moved_to": moved}


# ---------- შემოწმება (tools/voice_check.py): სატესტო ფრაზები ხმით და ნარევით + MMS ----------
_check_procs: dict[str, subprocess.Popen] = {}


def _check_dir(voice_id: str) -> Path:
    if not voice_id.replace("_", "").isalnum():
        raise HTTPException(404)
    return cpath("runs") / voice_id / "check"


@app.post("/api/voices/{voice_id}/check")
def api_check_start(voice_id: str):
    find_voice(voice_id)
    d = _check_dir(voice_id)
    p = _check_procs.get(voice_id)
    if p and p.poll() is None:
        raise HTTPException(409, "შემოწმება უკვე მიმდინარეობს")
    d.parent.mkdir(parents=True, exist_ok=True)
    log = (d.parent / "check.log").open("w", encoding="utf-8")
    _check_procs[voice_id] = subprocess.Popen([str(ROOT / "aivoice"), "check", voice_id], cwd=ROOT, stdout=log,
                                              stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
                                              env={**__import__("os").environ, "PYTHONUNBUFFERED": "1"})
    return {"ok": True}


@app.get("/api/voices/{voice_id}/check")
def api_check(voice_id: str):
    d = _check_dir(voice_id)
    p = _check_procs.get(voice_id)
    busy = bool(p and p.poll() is None)
    log = (d.parent / "check.log")
    return {"busy": busy, "report": read_json(d / "report.json"),
            "log": log.read_text(encoding="utf-8", errors="replace")[-1500:] if log.exists() else ""}


@app.get("/api/voices/{voice_id}/check/{name}")
def api_check_audio(voice_id: str, name: str):
    f = _check_dir(voice_id) / name
    if "/" in name or not name.endswith(".wav") or not f.exists():
        raise HTTPException(404)
    return FileResponse(f, media_type="audio/wav")


@app.get("/api/emotions")
def api_emotions():
    return {"presets": EMOTIONS, "limits": EMOTION_LIMITS}


class SynthRequest(BaseModel):
    voice: str
    text: str = Field(min_length=1, max_length=2000)
    emotion: str | None = None  # თუ მოცემულია — ხმის ნორმიდან გამოთვლილი პარამეტრები
    length_scale: float | None = None  # ცალკე მოცემული მნიშვნელობა ემოციისას ანაცვლებს
    noise_scale: float | None = None
    noise_w_scale: float | None = None
    volume: float | None = None


@app.post("/api/synthesize")
def api_synthesize(req: SynthRequest):
    v = find_voice(req.voice)
    if v.get("status") not in ("ready", "testing", "hidden") or not v.get("model"):
        raise HTTPException(409, "ეს ხმა ჯერ არ არის მზად")
    if req.emotion and req.emotion not in EMOTIONS:
        raise HTTPException(400, "უცნობი ემოცია")

    base = emotion_params(v, req.emotion) if req.emotion else voice_defaults(v)
    params = clamp({k: getattr(req, k) if getattr(req, k) is not None else base[k] for k in PARAM_KEYS})

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


# ---------- ტექსტი: როგორ წაიკითხავს + ლექსიკონი (app/textnorm.py) ----------
class TextIn(BaseModel):
    text: str = Field(max_length=5000)


@app.post("/api/textnorm")
def api_textnorm(t: TextIn):
    return {"text": ka_text(t.text)}


@app.get("/api/lexicon")
def api_lexicon():
    d = {k: v for k, v in read_json(LEXICON_DEFAULT, {}).items() if not k.startswith("_")}
    local = read_json(LEXICON, {})
    return {"entries": [{"word": k, "say": v, "local": k in local} for k, v in sorted(lexicon().items(), key=lambda x: x[0].lower())],
            "default_count": len(d), "local_count": len(local)}


class LexIn(BaseModel):
    word: str = Field(min_length=1, max_length=60)
    say: str = Field(min_length=1, max_length=120)


@app.post("/api/lexicon")
def api_lexicon_add(e: LexIn):
    local = read_json(LEXICON, {})
    local[e.word.strip()] = e.say.strip()
    write_json(LEXICON, local)
    return {"ok": True}


@app.delete("/api/lexicon/{word}")
def api_lexicon_del(word: str):
    local = read_json(LEXICON, {})
    if word in local:
        del local[word]
    elif word in read_json(LEXICON_DEFAULT, {}):
        local[word] = word  # ნაგულისხმევის გაუქმება: სიტყვა როგორც წერია (ლათინური → ავტომატური გადაწერა)
    write_json(LEXICON, local)
    return {"ok": True}


@app.get("/api/audio/{name}")
def api_audio(name: str):
    path = CACHE / name
    if "/" in name or not name.endswith(".wav") or not path.exists():
        raise HTTPException(404)
    return FileResponse(path, media_type="audio/wav")


@app.get("/api/roadmap")
def api_roadmap():
    return read_json(ROOT / "data" / "roadmap.json", {"stages": []})


@app.delete("/api/preview/{slot}/{n}")
def api_preview_delete(slot: str, n: str):
    """ნიმუშების ერთი თაობის წაშლა (საბაზო მოდელი → „მოსასმენი ნიმუშები“ → ✕)."""
    d = (cpath("runs") / slot / "previews" / n).resolve()
    if not d.is_relative_to(cpath("runs").resolve()) or d.parent.name != "previews" or not d.is_dir():
        raise HTTPException(404)
    shutil.rmtree(d)
    return {"ok": True}


@app.get("/api/preview/{slot}/{n}/{name}")
def api_preview(slot: str, n: str, name: str):
    path = (cpath("runs") / slot / "previews" / n / name).resolve()
    if not path.is_relative_to(cpath("runs").resolve()) or not path.is_file():
        raise HTTPException(404)
    return FileResponse(path, media_type="audio/wav")


app.mount("/", StaticFiles(directory=ROOT / "app" / "static", html=True), name="static")
