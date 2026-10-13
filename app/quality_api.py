"""„მონაცემები → ხარისხი“: Common Voice 27-ის ჩანაწერების შემოწმების შედეგები (tools/clip_quality.py).

GET  /api/quality                 შეჯამება + დინამიკები (ხარისხით) + მიმდინარეობა
GET  /api/quality/speaker/{spk}   დინამიკის ჩანაწერები (ყველაზე ცუდიდან)
GET  /api/quality/clip/{id}       mp3 მოსასმენად
POST /api/quality/decision        {spk, decision: include|exclude|auto} — თქვენი გადაწყვეტილება (data/dataset_decisions.json)
POST /api/quality/run · /stop     შემოწმების გაშვება / გაჩერება (./aivoice quality, გაგრძელებადი)

ჩანაწერი „ცუდია“, თუ ერთი წესი მაინც ირღვევა (BAD); დინამიკი „საეჭვოა“, თუ ცუდი ჩანაწერები ბევრია (SPEAKER).
"""

from __future__ import annotations

import json
import os
import re
import signal
import statistics as st
import subprocess
import threading
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from .config import ROOT, path as cpath

router = APIRouter(prefix="/api/quality")
DIR = cpath("data_root") / "analysis" / "cv27"
DECISIONS = ROOT / "data" / "dataset_decisions.json"
CORPUS = cpath("sources") / "cv27_ka" / "corpus.json"

# ჩანაწერის წესები: (ველი, პირობა, ახსნა)
BAD = {
    # MMS მოკლე ტექსტზე (< 8 ასო) არასანდოა — იქ არ ვითვლით
    "cer": (lambda m: m.get("cer", 0) > 0.35 and len(m.get("text", "")) >= 10, "ტექსტს არ ემთხვევა (MMS)"),
    "snr": (lambda m: m.get("snr", 99) < 25, "ხმაური"),
    "hiss": (lambda m: m.get("hiss", -99) > -10, "შიშინი"),
    "quiet": (lambda m: m.get("speech_db", 0) < -38, "ჩუმი"),
    "clip": (lambda m: m.get("clip", 0) > 0.001, "გადატვირთული"),
    "cps": (lambda m: not 5 <= m.get("cps", 12) <= 22, "უცნაური სიჩქარე"),
}
SPEAKER_BAD_SHARE = 0.4   # ცუდი ჩანაწერების წილი, რომლის ზემოთ დინამიკი „საეჭვოა“
MIN_GOOD_MIN = 3          # კარგი წუთები, რომლის ქვემოთ დინამიკი dataset-ში არ შედის

_cache: dict = {"key": None, "data": None}
_job: dict = {"proc": None}


def read_json(p: Path, default=None):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def reasons(m: dict) -> list[str]:
    if m.get("err"):
        return ["err"]
    return [k for k, (f, _) in BAD.items() if f(m)]


MMS_MIN_LETTERS, MMS_MIN_CLIPS = 10, 10


def cer_med(C: list) -> float | None:
    """დინამიკის MMS შეცდომა: მხოლოდ ≥10-ასოიანი ტექსტები (ერთ სიტყვაზე MMS არასანდოა), თითო ჩანაწერი ≤100%,
    და მხოლოდ მაშინ, როცა ≥10 ასეთი ჩანაწერი უკვე შემოწმდა."""
    v = [min(1.0, c["cer"]) for c in C if c.get("cer") is not None and len(c.get("text", "")) >= MMS_MIN_LETTERS]
    return round(st.median(v), 3) if len(v) >= MMS_MIN_CLIPS else None


def load() -> dict:
    f, fm = DIR / "clips.jsonl", DIR / "mms.jsonl"
    key = tuple((x.stat().st_mtime, x.stat().st_size) if x.exists() else None for x in (f, fm))
    if key[0] is None:
        key = None
    if key and _cache["key"] == key:
        return _cache["data"]
    corpus = {f"cv_{s['client_id'][:10]}": s for s in (read_json(CORPUS, {}) or {}).get("speakers", [])}
    mms = {}
    for line in (fm.read_text(encoding="utf-8").splitlines() if fm.exists() else []):
        try:
            x = json.loads(line)
            mms[x["id"]] = x
        except (json.JSONDecodeError, KeyError):
            pass
    spk: dict[str, dict] = {}
    for line in (f.read_text(encoding="utf-8").splitlines() if f.exists() else []):
        try:
            m = json.loads(line)
        except json.JSONDecodeError:
            continue
        if m["id"] in mms:
            m.update(cer=mms[m["id"]]["cer"], heard=mms[m["id"]]["heard"])
        s = spk.setdefault(m["spk"], {"spk": m["spk"], "current": m.get("current"), "clips": [], "why": {}})
        r = reasons(m)
        m["bad"] = r
        s["clips"].append(m)
        for x in r:
            s["why"][x] = s["why"].get(x, 0) + 1
    rows = []
    for name, s in spk.items():
        C = [c for c in s["clips"] if not c.get("err")]
        good = [c for c in C if not c["bad"]]
        med = lambda k: round(st.median(v), 3) if (v := [c[k] for c in C if c.get(k) is not None]) else None  # noqa: E731
        c = corpus.get(name, {})
        bad_share = round(1 - len(good) / len(s["clips"]), 3) if s["clips"] else 0
        good_min = round(sum(x["dur"] for x in good) / 60, 1)
        rows.append({"spk": name, "current": s["current"], "n": len(s["clips"]), "bad_share": bad_share,
                     "good_min": good_min, "min": round(sum(x["dur"] for x in C) / 60, 1),
                     "cer": cer_med(C), "mms_n": sum(1 for x in C if x.get("cer") is not None), "snr": med("snr"), "hiss": med("hiss"), "speech_db": med("speech_db"),
                     "f0": c.get("f0_median"), "why": s["why"],
                     "suggest": "exclude" if bad_share > SPEAKER_BAD_SHARE or good_min < MIN_GOOD_MIN else "include"})
    data = {"rows": rows, "clips": {n: s["clips"] for n, s in spk.items()}}
    _cache.update(key=key, data=data)
    return data


@router.get("")
def summary():
    d = load()
    dec = read_json(DECISIONS, {}) or {}
    rows = [{**r, "decision": dec.get(r["spk"], {}).get("decision", "auto")} for r in d["rows"]]
    for r in rows:
        r["final"] = r["suggest"] if r["decision"] == "auto" else r["decision"]
    clips = [c for cs in d["clips"].values() for c in cs]
    good = [c for c in clips if not c["bad"]]
    why: dict[str, int] = {}
    for c in clips:
        for x in c["bad"]:
            why[x] = why.get(x, 0) + 1
    incl = [r for r in rows if r["final"] == "include"]
    p = _job["proc"]
    return {
        "progress": read_json(DIR / "progress.json", {}) or {}, "busy": bool(p and p.poll() is None),
        "mms_done": sum(1 for c in clips if c.get("cer") is not None),
        "rules": {k: t for k, (_, t) in BAD.items()} | {"err": "ვერ გაიხსნა/სიჩუმე"},
        "totals": {"clips": len(clips), "good": len(good), "hours": round(sum(c.get("dur", 0) for c in clips) / 3600, 1),
                   "good_hours": round(sum(c["dur"] for c in good) / 3600, 1), "why": why,
                   "speakers": len(rows), "include": len(incl),
                   "include_low": sum(1 for r in incl if (r["f0"] or 999) < 165),
                   "include_hours": round(sum(r["good_min"] for r in incl) / 60, 1)},
        "speakers": sorted(rows, key=lambda r: -r["bad_share"]),
    }


@router.get("/speaker/{spk}")
def speaker(spk: str):
    clips = load()["clips"].get(spk)
    if clips is None:
        raise HTTPException(404)
    order = sorted(clips, key=lambda c: (-len(c["bad"]), -(c.get("cer") or 0)))
    return {"spk": spk, "clips": [{k: c.get(k) for k in ("id", "text", "heard", "kind", "dur", "cer", "snr", "hiss",
                                                          "speech_db", "cps", "bad")} for c in order]}


@router.get("/clip/{cid}")
def clip(cid: str):
    if not re.fullmatch(r"[\w.-]+\.mp3", cid):
        raise HTTPException(404)
    f = DIR / "clips" / cid
    if not f.exists():
        raise HTTPException(404)
    return FileResponse(f, media_type="audio/mpeg")


class Decision(BaseModel):
    spk: str = Field(pattern=r"^cv_[0-9a-f]{10}$")
    decision: str = Field(pattern=r"^(include|exclude|auto)$")


@router.post("/decision")
def decision(d: Decision):
    dec = read_json(DECISIONS, {}) or {}
    if d.decision == "auto":
        dec.pop(d.spk, None)
    else:
        dec[d.spk] = {"decision": d.decision, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    DECISIONS.write_text(json.dumps(dec, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"ok": True}


@router.post("/run")
def run():
    p = _job["proc"]
    if p and p.poll() is None:
        raise HTTPException(409, "შემოწმება უკვე მიმდინარეობს")
    log = (DIR.parent / "quality.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    _job["proc"] = p = subprocess.Popen([str(ROOT / "aivoice"), "quality"], cwd=ROOT, stdout=log.open("a", encoding="utf-8"),
                                        stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, start_new_session=True,
                                        env={**os.environ, "PYTHONUNBUFFERED": "1"})
    threading.Thread(target=p.wait, daemon=True).start()
    return {"ok": True}


@router.post("/stop")
def stop():
    p = _job["proc"]
    if p and p.poll() is None:
        os.killpg(p.pid, signal.SIGTERM)  # შედეგები თითო ჩანაწერზე ინახება — შემდეგ ჯერზე გააგრძელებს
    return {"ok": True}
