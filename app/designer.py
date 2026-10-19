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

DESIGNS = ROOT / "data" / "designs"
CACHE = cpath("cache") / "designer"
SR = 22050
ACTIVE = ROOT / "data" / "designer_model.json"  # რომელი საბაზო მოდელით მუშაობს შემქმნელი (პანელიდან იცვლება)

router = APIRouter(prefix="/api/designer")
_lock = threading.Lock()
_state: dict = {}  # მოდელი → ჩატვირთული მდგომარეობა


def models() -> dict:
    """საბაზო მოდელები, რომლებიც შემქმნელისთვის დაექსპორტებულია (tools/export_designer.py).
    თითოს თავისი რუკა (დინამიკები), რეიტინგი და შეფასებები აქვს; ნარევები მოდელითაა მონიშნული."""
    out = {}
    entries = [("base_ka", "ძველი (v1)", cpath("designer"), "base_ka_v1", "mms_rank.json", "designer_ratings.json")]
    for d in sorted(cpath("designer").parent.glob("designer_base_ka_v*")):  # designer_base_ka_v2, _v3 … (export_designer)
        slot = d.name.removeprefix("designer_")
        v = slot.rsplit("_", 1)[-1]
        newest = slot == max(p.name for p in d.parent.glob("designer_base_ka_v*")).removeprefix("designer_")
        entries.append((slot, f"ახალი ({v})" if newest else v,
                        d, slot, f"mms_rank_{slot}.json", f"designer_ratings_{slot}.json"))
    for slot, label, d, ds, rank, rat in entries:
        if (d / "meta.json").exists():
            meta = read_json(d / "meta.json", {}) or {}
            n = len((read_json(d / "config.json", {}) or {}).get("speaker_id_map", {}))
            out[slot] = {"slot": slot, "label": label, "dir": d, "speakers": cpath("datasets") / ds / "speakers.json",
                         "rank": ROOT / "data" / rank, "ratings": ROOT / "data" / rat,
                         "epoch": meta.get("epoch"), "voices": n}
    return out


def active() -> str:
    ms = models()
    a = (read_json(ACTIVE, {}) or {}).get("model")
    if a in ms:
        return a
    return sorted(ms)[-1] if ms else "base_ka"  # ნაგულისხმევად — უახლესი


def M() -> dict:
    ms = models()
    if not ms:
        raise HTTPException(409, "საბაზო მოდელი ჯერ არ არის დაექსპორტებული შემქმნელისთვის")
    return ms[active()]


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


def state(slot: str | None = None):
    """მოდელი იტვირთება პირველ მოთხოვნაზე და თავიდან — როცა ახალი სესიის შემდეგ ხელახლა დაექსპორტდება."""
    if slot:  # კონკრეტული მოდელი (მაგ. რეიტინგი ახალ მოდელზე) — აქტიურის შეცვლის გარეშე
        ms = models()
        if slot not in ms:
            raise HTTPException(404, "ასეთი მოდელი არ არის")
        m = ms[slot]
    else:
        m = M()
    des = m["dir"]
    meta_p = des / "meta.json"
    mtime = meta_p.stat().st_mtime
    with _lock:
        st = _state.setdefault(m["slot"], {"mtime": None})
        if st["mtime"] != mtime:
            from piper import PiperVoice
            voice = PiperVoice.load(des / "designer.onnx", des / "config.json")
            emb = np.load(des / "embeddings.npy")
            cfg = read_json(des / "config.json", {})
            smap = cfg.get("speaker_id_map", {})
            # v2-ს 600 ადგილი აქვს (მომავალი ხმებისთვის) — რუკაზე მხოლოდ ნასწავლი დინამიკები
            names = [n for n, _ in sorted(smap.items(), key=lambda x: x[1])]
            used = emb[[smap[n] for n in names]]
            info = read_json(m["speakers"], {}) or {}
            xy = map_layout(used, [(info.get(n) or {}).get("f0_median") for n in names])
            unit = used / (np.linalg.norm(used, axis=1, keepdims=True) + 1e-9)
            st.update(mtime=mtime, voice=voice, emb=emb, unit=unit, names=names, smap=smap,
                      info=info, xy=xy, meta=read_json(meta_p, {}), slot=m["slot"])
        return st


@router.get("/status")
def status():
    ms = models()
    meta = read_json(ms[active()]["dir"] / "meta.json") if ms else None
    return {"ready": meta is not None, "meta": meta, "model": active() if ms else None,
            "designs": len(list(DESIGNS.glob("*.json"))) if DESIGNS.exists() else 0}


@router.get("/models")
def list_models():
    a = active()
    return {"active": a, "models": [{k: v for k, v in m.items() if k in ("slot", "label", "epoch", "voices")}
                                    | {"active": k == a, "rank_file": m["rank"].name} for k, m in models().items()]}


class ModelIn(BaseModel):
    model: str


@router.post("/model")
def set_model(b: ModelIn):
    if b.model not in models():
        raise HTTPException(404, "ასეთი მოდელი არ არის")
    ACTIVE.write_text(json.dumps({"model": b.model}), encoding="utf-8")
    return {"ok": True, "active": b.model}


def traits() -> dict:
    """ტემბრით შეფასებული სქესი/ასაკი (tools/voice_traits.py → data/voice_traits_<model>.json).
    ასაკი სუსტად ფასდება (~74%) — გაურკვეველი შემთხვევა „?“-ია."""
    d = read_json(ROOT / "data" / f"voice_traits_{active()}.json", {}) or {}
    out = {}
    for n, t in (d.get("speakers") or {}).items():
        age = "young" if t["p_young"] >= 0.65 else "adult" if t["p_young"] <= 0.35 else "?"
        out[n] = {"gender": t["gender"], "age": age, "p_female": t["p_female"], "p_young": t["p_young"],
                  "label_gender": t.get("label_gender"), "label_age": t.get("label_age"), "mismatch": t.get("mismatch", [])}
    return out


@router.get("/traits")
def traits_info():
    d = read_json(ROOT / "data" / f"voice_traits_{active()}.json", {}) or {}
    return {"accuracy": d.get("accuracy"), "labeled": d.get("labeled"), "ready": bool(d)}


@router.get("/speakers")
def speakers(model: str | None = None):
    s = state(model)
    tr = traits()
    out = []
    for i, n in enumerate(s["names"]):
        inf = s["info"].get(n, {})
        out.append({"name": n, "x": round(float(s["xy"][i, 0]), 4), "y": round(float(s["xy"][i, 1]), 4),
                    "f0": inf.get("f0_median"), "minutes": inf.get("minutes"), **tr.get(n, {})})
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
    model: str | None = None  # ცარიელი — აქტიური მოდელი
    text: str = Field(min_length=1, max_length=1000)
    length_scale: float = 1.0
    noise_scale: float = 0.667
    noise_w_scale: float = 0.8


@router.post("/synth")
def synth(req: SynthReq):
    s = state(req.model)
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
    # საკმარისად ნასწავლი ხმები: v1-ში ≥ 10 წთ; v2-ში თითო ხმას ≤ 8 წთ აქვს (შერჩევის ზღვარი) → ≥ 5 წთ
    mins = [s["info"].get(n, {}).get("minutes") or 0 for n, _ in pool]
    thr = 10 if max(mins, default=0) > 20 else 5
    pool = [(n, f) for n, f in pool if f and (s["info"].get(n, {}).get("minutes") or 0) >= thr]
    if mode == "neutral":
        low = sorted([p for p in pool if 120 <= p[1] < 160], key=lambda p: -p[1])[:30]
        high = sorted([p for p in pool if 165 <= p[1] < 205], key=lambda p: p[1])[:30]
        if len(low) < 2 or len(high) < 2:
            raise HTTPException(409, "საკმარისი ხმა ვერ მოიძებნა")
        pick = [low[i] for i in rng.choice(len(low), 2, replace=False)] + [high[i] for i in rng.choice(len(high), 2, replace=False)]
    elif mode in ("female", "male", "female_young", "female_adult", "male_young", "male_adult") and (tr := traits()):
        g, _, ag = mode.partition("_")  # ტემბრით შეფასებული სქესი/ასაკი (არა Common Voice-ის მითითება)
        # ასაკი აქ — კლასიფიკატორის გადაწყვეტილებით (0.5); „?“ ზოლი მხოლოდ ჩვენებისთვისაა
        age_of = lambda n: "young" if tr.get(n, {}).get("p_young", 0.5) >= 0.5 else "adult"  # noqa: E731
        cand = [p for p in pool if tr.get(p[0], {}).get("gender") == g and (not ag or age_of(p[0]) == ag)]
        # რეიტინგი თუ არის — ჯგუფის უკეთესი ნახევრიდან (MMS-ით სუფთად მოლაპარაკეები)
        rk = {v["name"]: v["cer"] for v in (read_json(M()["rank"], {}) or {}).values() if v.get("kind") == "speaker"}
        if rk and len(cand) >= 8:
            cer = sorted(rk.get(n, 1) for n, _ in cand)
            cand = [c for c in cand if rk.get(c[0], 1) <= cer[len(cer) // 2]]
        if len(cand) < 2:
            raise HTTPException(409, "ამ ჯგუფში საკმარისი ხმა ვერ მოიძებნა")
        pick = [cand[i] for i in rng.choice(len(cand), min(k, len(cand)), replace=False)]
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
    """ყველა ნარევი; "model" — რომელ საბაზო მოდელზეა შექმნილი (ძველებს ველი არ აქვთ → base_ka)."""
    out = [read_json(p) for p in sorted(DESIGNS.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)] \
        if DESIGNS.exists() else []
    for d in out:
        d.setdefault("model", "base_ka")
    return out


@router.post("/designs")
def save_design(d: Design):
    s = state()
    _, info = mix_vector(s, d.weights)
    DESIGNS.mkdir(parents=True, exist_ok=True)
    did = re.sub(r"[^a-z0-9]+", "_", d.name.lower()).strip("_") or "voice"
    did = f"{did}_{int(time.time())}"
    rec = {"id": did, **d.model_dump(), "weights": info["weights"], "nearest": info["nearest"],
           "f0_estimate": info["f0_estimate"], "model_epoch": s["meta"].get("epoch"), "model": s["slot"],
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


class Rating(BaseModel):
    name: str
    rating: int = Field(ge=-1, le=1)  # 1 — მომწონს, -1 — არა, 0 — მოხსნა
    phrase: int | None = None
    note: str = Field(default="", max_length=300)


@router.get("/ratings")
def ratings():
    return read_json(M()["ratings"], {}) or {}


@router.post("/ratings")
def rate(r: Rating):
    s = state()
    if r.name not in s["smap"]:
        raise HTTPException(404, "ასეთი დინამიკი არ არის")
    with _lock:
        cur = read_json(M()["ratings"], {}) or {}
        if r.rating == 0:
            cur.pop(r.name, None)
        else:
            cur[r.name] = {"rating": r.rating, "phrase": r.phrase, "note": r.note,
                           "model_epoch": s["meta"].get("epoch"), "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
        tmp = M()["ratings"].with_suffix(".tmp")
        tmp.write_text(json.dumps(cur, ensure_ascii=False, indent=1), encoding="utf-8")
        tmp.replace(M()["ratings"])
    return {"ok": True, "count": len(cur)}




@router.get("/rank")
def rank():
    """tools/mms_rank.py: ყველა ხმა და ნარევი, 5 წინადადება — საუკეთესოდან (ნაკლები შეცდომა) უარესისკენ."""
    d = read_json(M()["rank"], {}) or {}
    names = {x.get("name") for x in designs() if x["model"] == active()}
    rows = sorted((r for r in d.values() if r["kind"] != "mix" or r["name"] in names), key=lambda r: r["cer"])
    return {"sentences": ["გამარჯობა! დღეს მშვენიერი ამინდია, არა?",
                          "ოდესღაც, ძალიან დიდი ხნის წინ, ერთ პატარა სოფელში ცხოვრობდა მოხუცი მეთევზე.",
                          "შენ აქ რას აკეთებ? ხომ გითხარი, არ მოსულიყავი!",
                          "ზღვის ღრმა ტალღებმა ნავი ნაპირისკენ გააქანა.",
                          "მწვრთნელმა ბავშვებს წყალში ცურვა ასწავლა."],
            "rows": [{**r, **(tr.get(r["name"], {}) if r["kind"] == "speaker" else {})} for r in rows]
            if (tr := traits()) else rows, "total": len(rows), "job": rank_job(),
            "model_epoch": M()["epoch"], "model": active()}


# ---------- რეიტინგის გაშვება პანელიდან (./aivoice rank) ----------
RANK_PROGRESS = ROOT / "data" / "mms_rank.progress.json"
RANK_LOG = cpath("runs") / "_rank.log"
_rank: dict = {"proc": None}


def rank_job() -> dict:
    p = _rank["proc"]
    busy = bool(p and p.poll() is None)
    prog = read_json(RANK_PROGRESS, {}) or {}
    log = RANK_LOG.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-1:] if RANK_LOG.exists() else []
    return {"busy": busy, **prog, "last": log[0] if log else "", "code": None if busy or not p else p.returncode}


class RankRun(BaseModel):
    kind: str = Field("mix", pattern=r"^(all|speaker|mix)$")
    fresh: bool = False


@router.post("/rank/run")
def rank_run(r: RankRun):
    import subprocess
    p = _rank["proc"]
    if p and p.poll() is None:
        raise HTTPException(409, "რეიტინგი უკვე მიმდინარეობს")
    RANK_PROGRESS.unlink(missing_ok=True)
    RANK_LOG.parent.mkdir(parents=True, exist_ok=True)
    log = RANK_LOG.open("w", encoding="utf-8")
    args = ["nice", "-n", "10", str(ROOT / "aivoice"), "rank", "--kind", r.kind] + (["--fresh"] if r.fresh else [])
    _rank["proc"] = p = subprocess.Popen(args, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                         start_new_session=True, env={**__import__("os").environ, "PYTHONUNBUFFERED": "1"})
    threading.Thread(target=p.wait, daemon=True).start()
    return {"ok": True}


@router.post("/rank/stop")
def rank_stop():
    import os
    import signal
    p = _rank["proc"]
    if p and p.poll() is None:
        os.killpg(p.pid, signal.SIGTERM)  # შედეგები ყოველ ხმაზე ინახება — შემდეგ ჯერზე იქიდან გააგრძელებს
    return {"ok": True}


@router.delete("/rank")
def rank_clear(kind: str = "all"):
    if _rank["proc"] and _rank["proc"].poll() is None:
        raise HTTPException(409, "ჯერ გააჩერეთ მიმდინარე შეფასება")
    d = read_json(M()["rank"], {}) or {}
    d = {k: v for k, v in d.items() if kind != "all" and v["kind"] != kind}
    M()["rank"].write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    RANK_PROGRESS.unlink(missing_ok=True)
    return {"ok": True, "left": len(d)}
