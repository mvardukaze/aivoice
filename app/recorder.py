"""ხმის ჩამწერი: ბრაუზერიდან ჩაწერა → ხარისხის შემოწმება → Piper-ის dataset.

recordings/<speaker>/                (D:-ზე, symlink)
  speaker.json                       სახელი, თანხმობა, წინადადებების სია
  raw/<pid>.flac                     ორიგინალი (მიკროფონის სიხშირით, როგორც მოვიდა)
  wav/<pid>.wav                      22050 Hz მონო, კიდეებზე სიჩუმე მოჭრილი — წვრთნისთვის
  meta/<pid>.json                    QC: ხანგრძლივობა, დონე, SNR, clipping, whisper CER

whisper-ით შემოწმება ფონურ ნაკადშია (large-v3, CPU, 4 ნაკადი) — მოდელი საჭიროებისას იტვირთება
და 10 წუთის უქმობის შემდეგ მეხსიერებიდან იშლება (წვრთნას RAM სჭირდება).
"""

import io
import json
import re
import shutil
import subprocess
import threading
import time
import unicodedata
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parent.parent
REC = ROOT / "recordings"
PROMPTS = ROOT / "data" / "prompts"
SR = 22050
MAX_CER = 0.30

router = APIRouter(prefix="/api/rec")


def read_json(p: Path, default=None):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write_json(p: Path, data):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(p)


def prompts(name="ka_v1"):
    return read_json(PROMPTS / f"{name}.json")["prompts"]


def spk_dir(spk: str) -> Path:
    if not re.fullmatch(r"[a-z0-9_]{2,40}", spk) or not (REC / spk / "speaker.json").exists():
        raise HTTPException(404, "დინამიკი ვერ მოიძებნა")
    return REC / spk


def slugify(name: str) -> str:
    tr = str.maketrans("აბგდევზთიკლმნოპჟრსტუფქღყშჩცძწჭხჯჰ",
                       "abgdevztiklmnopjrstufqgyscczwcxjh")
    s = re.sub(r"[^a-z0-9]+", "_", unicodedata.normalize("NFKD", name.lower().translate(tr))).strip("_")
    return (s or "speaker")[:30]


# ---------- ხარისხის შემოწმება ----------

def analyze(y: np.ndarray, sr: int) -> dict:
    frame = int(sr * 0.02)
    n = max(1, len(y) // frame)
    rms = np.sqrt(np.mean(y[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    db = 20 * np.log10(rms)
    speech_db, noise_db = float(np.percentile(db, 90)), float(np.percentile(db, 10))
    voiced = np.where(db > max(noise_db + 12, speech_db - 35))[0]
    return {
        "dur": len(y) / sr,
        "speech_db": round(speech_db, 1),
        "noise_db": round(noise_db, 1),
        "snr_db": round(speech_db - noise_db, 1),
        "peak": round(float(np.max(np.abs(y))), 3),
        "clip_ratio": float(np.mean(np.abs(y) > 0.99)),
        "voiced": (int(voiced[0]) * frame, int(voiced[-1] + 1) * frame) if len(voiced) else None,
    }


def qc_warnings(a: dict, text: str) -> list[str]:
    w = []
    if a["voiced"] is None:
        return ["ხმა არ ისმის — მიკროფონი შეამოწმეთ"]
    speech_sec = (a["voiced"][1] - a["voiced"][0]) / a["sr"]
    chars_per_sec = len(text) / max(speech_sec, 0.1)
    if a["clip_ratio"] > 0.0005 or a["peak"] >= 0.99:
        w.append("ხმა ზედმეტად ხმამაღალია (clipping) — მიკროფონი ოდნავ მოაშორეთ ან დონე დაუწიეთ")
    if a["speech_db"] < -35:
        w.append("ძალიან ჩუმია — მიუახლოვდით მიკროფონს")
    if a["snr_db"] < 25:
        w.append(f"ფონური ხმაური მაღალია (SNR {a['snr_db']} dB)")
    if chars_per_sec > 22:
        w.append("ძალიან სწრაფად — შეიძლება ტექსტი ბოლომდე არ იყოს წაკითხული")
    elif chars_per_sec < 6:
        w.append("ძალიან ნელა, ან დიდი პაუზებია")
    if a["voiced"][0] < 0.05 * a["sr"]:
        w.append("ჩაწერა ლაპარაკის დაწყებისთანავე დაიწყო — დასაწყისი შეიძლება მოჭრილი იყოს")
    return w


def cer(ref: str, hyp: str) -> float:
    norm = lambda t: " ".join("".join(c for c in t.lower() if c.isalnum() or c == " ").split())
    a, b = norm(ref), norm(hyp)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / max(1, len(a))


# ---------- whisper ფონური შემოწმება ----------

class WhisperWorker:
    def __init__(self):
        self.queue: list[tuple[str, str]] = []
        self.cv = threading.Condition()
        self.model = None
        self.last_used = 0.0
        threading.Thread(target=self.loop, daemon=True).start()

    def add(self, spk, pid):
        with self.cv:
            if (spk, pid) not in self.queue:
                self.queue.append((spk, pid))
            self.cv.notify()

    def pending(self, spk):
        with self.cv:
            return sum(1 for s, _ in self.queue if s == spk)

    def loop(self):
        while True:
            with self.cv:
                while not self.queue:
                    if self.model is not None and time.time() - self.last_used > 600:
                        self.model = None  # RAM-ის გათავისუფლება
                    self.cv.wait(timeout=60)
                spk, pid = self.queue.pop(0)
            try:
                self.check(spk, pid)
            except Exception as e:  # ერთი ფაილის შეცდომამ რიგი არ უნდა გააჩეროს
                meta_p = REC / spk / "meta" / f"{pid}.json"
                m = read_json(meta_p, {})
                m["whisper_error"] = str(e)[:200]
                write_json(meta_p, m)

    def check(self, spk, pid):
        wav = REC / spk / "wav" / f"{pid}.wav"
        meta_p = REC / spk / "meta" / f"{pid}.json"
        m = read_json(meta_p)
        if not wav.exists() or m is None:
            return
        recorded = m.get("recorded")
        if self.model is None:
            from faster_whisper import WhisperModel
            self.model = WhisperModel("large-v3", device="cpu", compute_type="int8", cpu_threads=4)
        # ffmpeg → 16 kHz float32: faster-whisper-ის PyAV-ით წაკითხვა ახალ av-სთან არ მუშაობს ("metadata_errors")
        pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", str(wav), "-ar", "16000", "-ac", "1", "-f", "f32le", "pipe:1"],
                             capture_output=True, check=True).stdout
        segs, _ = self.model.transcribe(np.frombuffer(pcm, dtype=np.float32), language="ka", beam_size=1,
                                        vad_filter=False)
        hyp = " ".join(s.text for s in segs).strip()
        self.last_used = time.time()
        m = read_json(meta_p)
        if not m or m.get("recorded") != recorded:  # ამასობაში თავიდან ჩაიწერა ან წაიშალა
            return
        m.update(whisper=hyp, cer=round(cer(m["text"], hyp), 3))
        write_json(meta_p, m)


worker = WhisperWorker()


# ---------- API ----------

class NewSpeaker(BaseModel):
    name: str = Field(min_length=2, max_length=60)
    consent: bool
    gender: str = ""
    age: str = ""


def speaker_stats(d: Path) -> dict:
    metas = [read_json(p, {}) for p in (d / "meta").glob("*.json")] if (d / "meta").exists() else []
    good = [m for m in metas if not m.get("warnings") and (m.get("cer") is None or m["cer"] <= MAX_CER)]
    return {
        "recorded": len(metas),
        "total": len(prompts(read_json(d / "speaker.json", {}).get("prompts", "ka_v1"))),
        "minutes": round(sum(m.get("speech_sec", 0) for m in metas) / 60, 1),
        "good": len(good),
        "flagged": len(metas) - len(good),
        "checking": worker.pending(d.name),
    }


@router.get("/speakers")
def list_speakers():
    out = []
    for d in sorted(REC.iterdir()) if REC.exists() else []:
        sp = read_json(d / "speaker.json")
        if sp:
            out.append({"id": d.name, **sp, **speaker_stats(d)})
    return out


@router.post("/speakers")
def create_speaker(req: NewSpeaker):
    if not req.consent:
        raise HTTPException(400, "ჩაწერისთვის თანხმობა აუცილებელია")
    base = slugify(req.name)
    spk, i = base, 2
    while (REC / spk).exists():
        spk, i = f"{base}_{i}", i + 1
    write_json(REC / spk / "speaker.json", {
        "name": req.name.strip(), "gender": req.gender, "age": req.age, "prompts": "ka_v1",
        "consent": True, "consent_text": "ვეთანხმები, რომ ჩემი ხმის ჩანაწერები გამოყენებულ იქნას ხმოვანი მოდელის "
                                           "შესაქმნელად და ეს მოდელი შეიძლება საჯაროდ გავრცელდეს.",
        "consent_time": time.strftime("%Y-%m-%dT%H:%M:%S"), "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
    })
    return {"id": spk}


@router.get("/{spk}/state")
def state(spk: str):
    d = spk_dir(spk)
    sp = read_json(d / "speaker.json")
    items = []
    for p in prompts(sp.get("prompts", "ka_v1")):
        m = read_json(d / "meta" / f"{p['id']}.json")
        item = {"id": p["id"], "text": p["text"]}
        if m:
            item.update(recorded=True, warnings=m.get("warnings", []), cer=m.get("cer"),
                        whisper=m.get("whisper"), dur=m.get("speech_sec"))
        items.append(item)
    return {"speaker": {"id": spk, **sp}, "stats": speaker_stats(d), "items": items}


@router.post("/{spk}/export")
def export(spk: str):
    """კარგი ჩანაწერები → datasets/rec_<spk>/ (Piper-ის ფორმატი: wav/ + metadata.csv)."""
    d = spk_dir(spk)
    out = ROOT / "datasets" / f"rec_{spk}"
    shutil.rmtree(out, ignore_errors=True)
    (out / "wav").mkdir(parents=True)
    lines, secs = [], 0.0
    for mp in sorted((d / "meta").glob("*.json")):
        m = read_json(mp, {})
        if m.get("warnings") or (m.get("cer") is not None and m["cer"] > MAX_CER):
            continue
        src = d / "wav" / f"{m['id']}.wav"
        if src.exists():
            shutil.copy2(src, out / "wav" / src.name)
            lines.append(f"{src.name}|{m['text']}")
            secs += m.get("speech_sec", 0)
    (out / "metadata.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    stats = {"speaker": f"recordings/{spk}", "status": "მზადაა", "hours": round(secs / 3600, 2),
             "clips": len(lines), "source": "საკუთარი ჩანაწერი"}
    write_json(out / "stats.json", stats)
    return stats


@router.post("/{spk}/{pid}")
async def upload(spk: str, pid: str, request: Request):
    d = spk_dir(spk)
    text = next((p["text"] for p in prompts(read_json(d / "speaker.json")["prompts"]) if p["id"] == pid), None)
    if text is None:
        raise HTTPException(404, "წინადადება ვერ მოიძებნა")
    body = await request.body()
    if len(body) > 30_000_000:
        raise HTTPException(413, "ფაილი ზედმეტად დიდია")
    try:
        y, sr = sf.read(io.BytesIO(body), dtype="float32", always_2d=True)
    except Exception:
        raise HTTPException(400, "აუდიო ვერ წავიკითხე (WAV უნდა იყოს)")
    y = y.mean(axis=1)

    (d / "raw").mkdir(exist_ok=True)
    sf.write(d / "raw" / f"{pid}.flac", y, sr, subtype="PCM_24" if sr > 44000 else "PCM_16")

    a = analyze(y, sr)
    a["sr"] = sr
    warnings = qc_warnings(a, text)

    # 22050 Hz, სიჩუმე კიდეებზე მოჭრილი (0.25 წმ ბუფერით) — წვრთნისთვის
    (d / "wav").mkdir(exist_ok=True)
    speech_sec = 0.0
    if a["voiced"]:
        s0 = max(0, a["voiced"][0] - int(0.25 * sr))
        s1 = min(len(y), a["voiced"][1] + int(0.25 * sr))
        seg = y[s0:s1]
        speech_sec = len(seg) / sr
        pcm = subprocess.run(["ffmpeg", "-v", "error", "-f", "f32le", "-ar", str(sr), "-ac", "1", "-i", "pipe:0",
                              "-ar", str(SR), "-ac", "1", "-c:a", "pcm_s16le", "-f", "wav", "pipe:1"],
                             input=seg.astype(np.float32).tobytes(), capture_output=True, check=True).stdout
        (d / "wav" / f"{pid}.wav").write_bytes(pcm)

    meta = {"id": pid, "text": text, "recorded": time.strftime("%Y-%m-%dT%H:%M:%S"), "input_sr": sr,
            "speech_sec": round(speech_sec, 2), "warnings": warnings,
            **{k: a[k] for k in ("speech_db", "noise_db", "snr_db", "peak", "clip_ratio")}}
    write_json(d / "meta" / f"{pid}.json", meta)
    if a["voiced"]:
        worker.add(spk, pid)
    return {"warnings": warnings, "speech_sec": meta["speech_sec"], "snr_db": a["snr_db"],
            "speech_db": a["speech_db"], "stats": speaker_stats(d)}


@router.get("/{spk}/audio/{pid}")
def audio(spk: str, pid: str):
    p = spk_dir(spk) / "wav" / f"{pid}.wav"
    if not re.fullmatch(r"[a-z0-9]+", pid) or not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="audio/wav", headers={"Cache-Control": "no-store"})


@router.delete("/{spk}/{pid}")
def delete(spk: str, pid: str):
    d = spk_dir(spk)
    if not re.fullmatch(r"[a-z0-9]+", pid):
        raise HTTPException(400)
    for p in (d / "raw" / f"{pid}.flac", d / "wav" / f"{pid}.wav", d / "meta" / f"{pid}.json"):
        p.unlink(missing_ok=True)
    return {"stats": speaker_stats(d)}
