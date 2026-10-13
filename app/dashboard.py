"""პანელის ახალი ჩანართების API: მიმოხილვა, საბაზო მოდელი, მონაცემები."""

import datetime as dt
import json
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter

from .config import ROOT, path as cpath

RUNS = cpath("runs")
DDATA = cpath("datasets")
BASE_SLOT = "base_ka"
KAGGLE_WEEKLY_HOURS = 30
GCP_STATE = cpath("gcp") / "state.json"
GCP_SEC_PER_BATCH = 1.05  # L4, batch 32 — გაზომილი 2026-10-08
GCP_BATCH = 32


def gcp_info(st: dict) -> dict | None:
    """Google Cloud-ის მდგომარეობა (tools/gcp_train.py) + პროგრესი/დარჩენილი დრო პანელისთვის."""
    g = read_json(GCP_STATE)
    if not g:
        return None
    p = g.get("progress") or {}
    bpe = int((st.get("clips") or 62282) * 0.99 / GCP_BATCH) or 1
    out = {k: g.get(k) for k in ("phase", "zone", "started", "restarts", "last_restart", "error", "fetched_at", "updated")}
    out.update(epoch=p.get("epoch"), batch=p.get("batch"), batches_per_epoch=bpe, sec_per_batch=GCP_SEC_PER_BATCH)
    if g.get("started"):
        try:
            out["hours"] = round((dt.datetime.now() - dt.datetime.fromisoformat(g["started"])).total_seconds() / 3600, 1)
        except ValueError:
            pass
    return out

router = APIRouter(prefix="/api")


def read_json(p: Path, default=None):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return default


def run_status(slot: str) -> dict | None:
    st = read_json(RUNS / slot / "status.json")
    if not st:
        return None
    # ადგილობრივი წვრთნა სტატუსს ~20 წამში ერთხელ წერს; Kaggle-ის მეთვალყურე — 10 წუთში ერთხელ
    stale = 1800 if st.get("device") == "kaggle" else 900
    if st.get("status") in ("training", "exporting") and st.get("updated"):
        try:
            if (dt.datetime.now() - dt.datetime.fromisoformat(st["updated"])).total_seconds() > stale:
                st["status"] = "paused"
        except ValueError:
            pass
    prev_dir = RUNS / slot / "previews"
    extra = [read_json(p, {}) for p in sorted(prev_dir.glob("*/meta.json"))] if prev_dir.exists() else []
    if st.get("device") == "gcp":
        g = gcp_info(st)
        if g:
            st["gcp"] = g
            job = read_json(ROOT / "data" / "kaggle_jobs" / f"{slot}.json", {}) or {}
            st["max_epochs"] = job.get("max_epochs") or st.get("max_epochs")
            if g.get("batch") is not None:
                st["steps_per_epoch"], st["step_in_epoch"] = g["batches_per_epoch"], min(g["batch"], g["batches_per_epoch"])
                left = (st["max_epochs"] - (st.get("epoch") or 0)) * g["batches_per_epoch"] - st["step_in_epoch"]
                st["eta_hours"] = round(max(0, left) * GCP_SEC_PER_BATCH / 3600, 1)
    loc = read_json(RUNS / slot / "local" / "state.json")
    if loc:
        if loc.get("running") and loc.get("updated"):
            try:
                if (dt.datetime.now() - dt.datetime.fromisoformat(loc["updated"])).total_seconds() > 900:
                    loc["running"] = False  # კომპიუტერი გაითიშა / სერვისი გაჩერდა
            except ValueError:
                pass
        st["local"] = {k: loc.get(k) for k in ("running", "epoch", "step_in_epoch", "steps_per_epoch", "sec_per_step",
                                                "epoch_hours", "started", "stopped", "note", "error", "designer_epoch", "updated")}
    st["previews"] = sorted(st.get("previews", []) + [e for e in extra if e],
                            key=lambda p: (not p.get("reference"), p.get("time", "")))
    return {"name": slot, **st}


def kaggle_week_hours() -> float:
    """ბოლო 7 დღეში Kaggle-ზე დახარჯული GPU-საათები (იგივე მოძრავი ფანჯარა, რასაც kaggle_train.py იყენებს)."""
    cutoff = (dt.datetime.now() - dt.timedelta(days=7)).isoformat(timespec="seconds")
    total = 0.0
    for p in RUNS.glob("*/status.json"):
        for s in (read_json(p, {}) or {}).get("sessions", []):
            if (s.get("ended") or "") >= cutoff:
                total += s.get("hours") or 0
    return round(total, 1)


@router.get("/overview")
def overview():
    jobs = []
    for p in sorted(RUNS.glob("*/status.json")):
        r = run_status(p.parent.name)
        if r and r.get("status") in ("training", "exporting", "partial", "paused", "failed"):
            r.pop("previews", None)
            jobs.append(r)
    return {"jobs": jobs, "kaggle_week_hours": kaggle_week_hours(), "kaggle_week_limit": KAGGLE_WEEKLY_HOURS}


@lru_cache(maxsize=4)
def _speakers_summary(path: str, mtime: float):
    spk = read_json(Path(path), {}) or {}
    f0 = [v.get("f0_median") for v in spk.values() if v.get("f0_median")]
    bins = list(range(80, 281, 20))
    hist = [{"from": b, "to": b + 20, "n": sum(1 for f in f0 if b <= f < b + 20)} for b in bins[:-1]]
    mins = sorted((v["minutes"] for v in spk.values()), reverse=True)
    return {"speakers": len(spk), "f0_hist": hist,
            "low": sum(1 for f in f0 if f < 165), "high": sum(1 for f in f0 if f >= 165),
            "minutes_top": mins[:5], "minutes_median": mins[len(mins) // 2] if mins else 0}


@router.get("/base")
def base():
    job = read_json(ROOT / "data" / "kaggle_jobs" / f"{BASE_SLOT}.json", {}) or {}
    ds = Path(job.get("dataset_dir") or DDATA / "base_ka_v1")
    sp = ds / "speakers.json"
    summary = _speakers_summary(str(sp), sp.stat().st_mtime) if sp.exists() else {}
    names = read_json(sp, {}) or {}
    return {
        "job": {k: v for k, v in job.items() if not k.startswith("_")},
        "note": job.get("_note"),
        "dataset": {**(read_json(ds / "stats.json", {}) or {}), **summary},
        "run": run_status(BASE_SLOT),
        "preview_speakers": {s: names.get(s, {}) for s in job.get("preview_speakers", [])},
        "onnx_ready": (RUNS / BASE_SLOT / "kaggle_out" / f"{BASE_SLOT}.onnx").exists(),
        "kaggle_week_hours": kaggle_week_hours(), "kaggle_week_limit": KAGGLE_WEEKLY_HOURS,
    }


@lru_cache(maxsize=4)
def _corpus_summary(path: str, mtime: float):
    S = (read_json(Path(path), {}) or {}).get("speakers", [])
    measured = [s for s in S if "snr_db" in s]

    def good(s):
        return (s.get("snr_db", 0) >= 25 and s.get("clip", 1) < 0.001 and s["audio_min"] >= 5
                and (s.get("f0_spread") is None or s["f0_spread"] <= 3) and s["votes_down_ratio"] < 0.2)

    g = [s for s in measured if good(s)]
    return {"speakers": len(S), "hours": round(sum(s["audio_min"] for s in S) / 60, 1),
            "validated_hours": round(sum(s["validated_min"] for s in S) / 60, 1),
            "measured": len(measured), "good": len(g),
            "good_hours_cap3": round(sum(min(s["audio_min"], 180) for s in g) / 60, 1),
            "multi_person": sum(1 for s in measured if (s.get("f0_spread") or 0) > 3)}


@router.get("/data")
def data():
    datasets = []
    for d in sorted([p for p in DDATA.iterdir() if p.is_dir() and not p.name.endswith("_unused")] if DDATA.exists() else []):
        st = read_json(d / "stats.json", {}) or {}
        meta = d / "metadata.csv"
        datasets.append({"name": d.name, "location": str(d.parent),
                         "clips": st.get("clips") or (sum(1 for _ in meta.open(encoding="utf-8")) if meta.exists() else 0),
                         "hours": st.get("hours"), "speakers": st.get("speakers"),
                         "speaker": st.get("speaker"), "source": st.get("source"), "status": st.get("status")})
    sources = []
    for name, path, label in [("cv27_ka", cpath("sources") / "cv27_ka" / "corpus.json", "Common Voice 27 (Mozilla Data Collective)")]:
        if path.exists():
            sources.append({"name": name, "label": label, **_corpus_summary(str(path), path.stat().st_mtime)})
    return {"datasets": datasets, "sources": sources}
