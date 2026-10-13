"""ხმის შემქმნელი: საბაზო მოდელის დინამიკების ვექტორების შერევა → ახალი, არავის ხმა.

მონაცემები (tools/export_designer.py): <paths.designer>/{designer.onnx, embeddings.npy, config.json, meta.json}
დიზაინები: data/designs/<id>.json — სახელი, წილები, სინთეზის პარამეტრები.

„უნიკალურობა“ — ნარევის ვექტორის კოსინუსური მსგავსება ყველაზე ახლო რეალურ დინამიკთან. ეს სწრაფი
მაჩვენებელია მოდელის შიგნით; საბოლოო შემოწმება (ECAPA, აუდიოზე) ხმის შენახვისას მოხდება.
"""

import hashlib
import io
import json
import re
import threading
import time
import wave
from pathlib import Path

import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .textnorm import ka_text

from .config import ROOT, path as cpath

DES = cpath("designer")  # იწერება tools/export_designer.py-ით
DESIGNS = ROOT / "data" / "designs"
CACHE = cpath("cache") / "designer"
BASE_SPEAKERS = cpath("datasets") / "base_ka_v1" / "speakers.json"
SR = 22050

router = APIRouter(prefix="/api/designer")
_lock = threading.Lock()
_state = {"mtime": None}


def read_json(p: Path, default=None):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def map_layout(emb: np.ndarray, f0s: list) -> np.ndarray:
    """2D რუკა: X = ხმის სიმაღლე (რანგით — თანაბრად გაშლილი), Y = ვექტორის მთავარი კომპონენტი, რომელიც
    სიმაღლეს არ იმეორებს. ბოლოს წერტილები ერთმანეთს „განზე სწევს“, რომ არ გადაფარონ."""
    n = len(emb)
    f = np.array([v if v else np.nan for v in f0s], dtype=float)
    f[np.isnan(f)] = np.nanmedian(f) if np.isfinite(f).any() else 0.0
    rank = lambda v: (np.argsort(np.argsort(v, kind="stable")) + 0.5) / len(v)
    c = emb - emb.mean(axis=0)
    fz = (f - f.mean()) / (f.std() + 1e-9)
    c = c - np.outer(fz, fz @ c) / n  # სიმაღლესთან დაკავშირებული მიმართულების მოცილება
    _, _, vt = np.linalg.svd(c, full_matrices=False)
    xy = np.stack([rank(f), rank(c @ vt[0])], axis=1)
    d_min = 0.8 / np.sqrt(n)
    for _ in range(200):
        diff = xy[:, None, :] - xy[None, :, :]
        dist = np.sqrt((diff ** 2).sum(-1)) + np.eye(n)
        push = np.clip(d_min - dist, 0, None) / dist
        np.fill_diagonal(push, 0)
        if not push.any():
            break
        xy = np.clip(xy + 0.5 * (diff * push[..., None]).sum(1), 0, 1)
    return xy


def state():
    """მოდელი იტვირთება პირველ მოთხოვნაზე და თავიდან — როცა ახალი სესიის შემდეგ ხელახლა დაექსპორტდება."""
    meta_p = DES / "meta.json"
    if not meta_p.exists():
        raise HTTPException(409, "საბაზო მოდელი ჯერ არ არის დაექსპორტებული შემქმნელისთვის")
    mtime = meta_p.stat().st_mtime
    with _lock:
        if _state["mtime"] != mtime:
            from piper import PiperVoice
            voice = PiperVoice.load(DES / "designer.onnx", DES / "config.json")
            emb = np.load(DES / "embeddings.npy")
            cfg = read_json(DES / "config.json", {})
            smap = cfg.get("speaker_id_map", {})
            names = [None] * len(smap)
            for n, i in smap.items():
                names[i] = n
            info = read_json(BASE_SPEAKERS, {}) or {}
            xy = map_layout(emb, [(info.get(n) or {}).get("f0_median") for n in names])
            unit = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9)
            _state.update(mtime=mtime, voice=voice, emb=emb, unit=unit, names=names, smap=smap,
                          info=info, xy=xy, meta=read_json(meta_p, {}))
        return _state


@router.get("/status")
def status():
    meta = read_json(DES / "meta.json")
    return {"ready": meta is not None, "meta": meta,
            "designs": len(list(DESIGNS.glob("*.json"))) if DESIGNS.exists() else 0}


@router.get("/speakers")
def speakers():
    s = state()
    out = []
    for i, n in enumerate(s["names"]):
        inf = s["info"].get(n, {})
        out.append({"name": n, "x": round(float(s["xy"][i, 0]), 4), "y": round(float(s["xy"][i, 1]), 4),
                    "f0": inf.get("f0_median"), "minutes": inf.get("minutes")})
    return {"speakers": out, "meta": s["meta"]}


def mix_vector(s, weights: dict[str, float]) -> tuple[np.ndarray, dict]:
    w = {n: float(v) for n, v in weights.items() if n in s["smap"] and float(v) > 0}
    if not w:
        raise HTTPException(400, "აირჩიეთ ერთი დინამიკი მაინც")
    tot = sum(w.values())
    g = sum((v / tot) * s["emb"][s["smap"][n]] for n, v in w.items())
    # ყველაზე ახლო რეალური დინამიკი (კოსინუსი) — „ვისი ხმაა?“ სწრაფი მაჩვენებელი
    u = g / (np.linalg.norm(g) + 1e-9)
    sims = s["unit"] @ u
    order = np.argsort(-sims)[:3]
    nearest = [{"name": s["names"][i], "sim": round(float(sims[i]), 3)} for i in order]
    f0s = [(s["info"].get(n, {}).get("f0_median"), v / tot) for n, v in w.items()]
    f0_est = sum(f * v for f, v in f0s if f) / max(1e-9, sum(v for f, v in f0s if f)) if any(f for f, _ in f0s) else None
    return g.astype(np.float32), {"nearest": nearest, "f0_estimate": round(f0_est) if f0_est else None,
                                  "weights": {n: round(v / tot, 3) for n, v in w.items()}}


class SynthReq(BaseModel):
    weights: dict[str, float]
    text: str = Field(min_length=1, max_length=1000)
    length_scale: float = 1.0
    noise_scale: float = 0.667
    noise_w_scale: float = 0.8


@router.post("/synth")
def synth(req: SynthReq):
    s = state()
    g, info = mix_vector(s, req.weights)
    params = [min(1.0, max(0.2, req.noise_scale)), min(1.4, max(0.7, req.length_scale)), min(1.2, max(0.3, req.noise_w_scale))]
    key = hashlib.sha1(json.dumps([s["meta"].get("exported"), info["weights"], ka_text(req.text), params]).encode()).hexdigest()[:16]
    CACHE.mkdir(parents=True, exist_ok=True)
    out = CACHE / f"{key}.wav"
    if not out.exists():
        voice = s["voice"]
        chunks = []
        for ph in voice.phonemize(ka_text(req.text)):
            if not ph:
                continue
            ids = np.array([voice.phonemes_to_ids(ph)], dtype=np.int64)
            audio = voice.session.run(None, {
                "input": ids, "input_lengths": np.array([ids.shape[1]], dtype=np.int64),
                "scales": np.array(params, dtype=np.float32), "g": g[None, :]})[0].squeeze()
            chunks += [audio, np.zeros(int(SR * 0.25), dtype=np.float32)]
        if not chunks:
            raise HTTPException(400, "ტექსტიდან ბგერები ვერ მივიღე")
        a = np.concatenate(chunks)
        a = a / (np.max(np.abs(a)) + 1e-8) * 0.9
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SR)
            w.writeframes((a * 32767).astype(np.int16).tobytes())
        tmp = out.with_suffix(".tmp")
        tmp.write_bytes(buf.getvalue())
        tmp.rename(out)
    return {"url": f"/api/designer/audio/{out.name}", **info, "model_epoch": s["meta"].get("epoch")}


@router.get("/audio/{name}")
def audio(name: str):
    p = CACHE / name
    if not re.fullmatch(r"[0-9a-f]{16}\.wav", name) or not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="audio/wav")


@router.get("/suggest")
def suggest(mode: str = "neutral", k: int = 4, seed: int | None = None):
    """შემთხვევითი ნარევი: neutral — ორი დაბალი + ორი მაღალი ხმა შუა სიმაღლესთან ახლოს;
    female / male — მხოლოდ მაღალი / დაბალი ხმები; random — ნებისმიერი."""
    s = state()
    rng = np.random.default_rng(seed if seed is not None else int(time.time() * 1000) % 2**32)
    pool = [(n, s["info"].get(n, {}).get("f0_median")) for n in s["names"]]
    pool = [(n, f) for n, f in pool if f and (s["info"].get(n, {}).get("minutes") or 0) >= 10]
    if mode == "neutral":
        low = sorted([p for p in pool if 120 <= p[1] < 160], key=lambda p: -p[1])[:30]
        high = sorted([p for p in pool if 165 <= p[1] < 205], key=lambda p: p[1])[:30]
        pick = [low[i] for i in rng.choice(len(low), 2, replace=False)] + [high[i] for i in rng.choice(len(high), 2, replace=False)]
    else:
        cand = [p for p in pool if (mode == "female" and p[1] >= 175) or (mode == "male" and p[1] < 140) or mode == "random"]
        pick = [cand[i] for i in rng.choice(len(cand), min(k, len(cand)), replace=False)]
    w = rng.dirichlet(np.ones(len(pick)) * 2.0)
    return {"weights": {n: round(float(x), 3) for (n, _), x in zip(pick, w)}}


class Design(BaseModel):
    name: str = Field(min_length=1, max_length=60)
    weights: dict[str, float]
    length_scale: float = 1.0
    noise_scale: float = 0.667
    noise_w_scale: float = 0.8
    note: str = ""


@router.get("/designs")
def designs():
    return [read_json(p) for p in sorted(DESIGNS.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)] \
        if DESIGNS.exists() else []


@router.post("/designs")
def save_design(d: Design):
    s = state()
    _, info = mix_vector(s, d.weights)
    DESIGNS.mkdir(parents=True, exist_ok=True)
    did = re.sub(r"[^a-z0-9]+", "_", d.name.lower()).strip("_") or "voice"
    did = f"{did}_{int(time.time())}"
    rec = {"id": did, **d.model_dump(), "weights": info["weights"], "nearest": info["nearest"],
           "f0_estimate": info["f0_estimate"], "model_epoch": s["meta"].get("epoch"),
           "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
    (DESIGNS / f"{did}.json").write_text(json.dumps(rec, ensure_ascii=False, indent=1), encoding="utf-8")
    return rec


@router.delete("/designs/{did}")
def delete_design(did: str):
    p = DESIGNS / f"{did}.json"
    if not re.fullmatch(r"[a-z0-9_]+", did) or not p.exists():
        raise HTTPException(404)
    p.unlink()
    return {"ok": True}


# ---------- ყველა ხმის მოსმენა: შეფასებები (👍 / 👎), Claude-საც ეს ფაილი ეხმარება ნარევის შერჩევაში ----------
RATINGS = ROOT / "data" / "designer_ratings.json"


class Rating(BaseModel):
    name: str
    rating: int = Field(ge=-1, le=1)  # 1 — მომწონს, -1 — არა, 0 — მოხსნა
    phrase: int | None = None
    note: str = Field(default="", max_length=300)


@router.get("/ratings")
def ratings():
    return read_json(RATINGS, {}) or {}


@router.post("/ratings")
def rate(r: Rating):
    s = state()
    if r.name not in s["smap"]:
        raise HTTPException(404, "ასეთი დინამიკი არ არის")
    with _lock:
        cur = read_json(RATINGS, {}) or {}
        if r.rating == 0:
            cur.pop(r.name, None)
        else:
            cur[r.name] = {"rating": r.rating, "phrase": r.phrase, "note": r.note,
                           "model_epoch": s["meta"].get("epoch"), "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        tmp = RATINGS.with_suffix(".tmp")
        tmp.write_text(json.dumps(cur, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(RATINGS)
    return {"ok": True, "count": len(cur)}


RANK = ROOT / "data" / "mms_rank.json"  # tools/mms_rank.py


@router.get("/rank")
def rank():
    """tools/mms_rank.py: ყველა ხმა და ნარევი, 5 წინადადება — საუკეთესოდან (ნაკლები შეცდომა) უარესისკენ."""
    d = read_json(RANK, {}) or {}
    rows = sorted(d.values(), key=lambda r: r["cer"])
    return {"sentences": ["გამარჯობა! დღეს მშვენიერი ამინდია, არა?",
                          "ოდესღაც, ძალიან დიდი ხნის წინ, ერთ პატარა სოფელში ცხოვრობდა მოხუცი მეთევზე.",
                          "შენ აქ რას აკეთებ? ხომ გითხარი, არ მოსულიყავი!",
                          "ზღვის ღრმა ტალღებმა ნავი ნაპირისკენ გააქანა.",
                          "მწვრთნელმა ბავშვებს წყალში ცურვა ასწავლა."],
            "rows": rows, "total": len(rows)}
