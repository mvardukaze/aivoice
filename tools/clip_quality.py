"""Common Voice 27-ის ჩანაწერების ხარისხის შემოწმება — საბაზო მოდელის გაგრძელებამდე.

    ./aivoice quality [measure|mms|all] [--cap-hours 4] [--cpu]
შედეგი: <data_root>/analysis/cv27/
    clips.jsonl      თითო ჩანაწერი: dur, snr, hiss, speech_db, clip, lead/trail, cps, zeros (გაგრძელებადი)
    mms.jsonl        cer, heard — ტექსტის დამთხვევა (ცალკე, ნელი ეტაპი; GTX 1050 Ti-ზე ~1 ჩანაწერი/წმ)
    progress.json    პანელისთვის (მონაცემები → ხარისხი)
    clips/<id>.mp3   შემოწმებული ჩანაწერები (მოსასმენად და შემდეგი dataset-ისთვის — არქივის თავიდან გახსნა აღარ სჭირდება)
დინამიკების შეჯამება: tools/quality_summary.py (ან პანელი).

რა შემოწმდება: „კარგი“ დინამიკები corpus.json-იდან (≥5 წთ, SNR ≥ 25, ერთი ადამიანი, < 20% უარყოფითი ხმა) —
ახლანდელი 296 + ახალი კანდიდატები; validated და other (შეუმოწმებელი) ჩანაწერები; თითო დინამიკზე --cap-hours
(ჯერ validated, მერე other; შემთხვევითი, seed 27).

ზომები:
  snr       მეტყველება (90-ე პროცენტილი) − ფონი (10-ე), dB — ხმაური
  hiss      ჩუმი კადრების 4–11 kHz ენერგია მეტყველებასთან შედარებით, dB — შიშინი (რაც მაღალია, მით ცუდი)
  speech_db მეტყველების დონე, dBFS — ჩუმი ჩანაწერი
  clip      გადატვირთული სემპლების წილი
  cps       ასო წამში — ძალიან სწრაფი/ნელი = ტექსტი შეიძლება არ ემთხვეოდეს
  cer       MMS-მა რა გაიგონა ↔ ტექსტი (ასოების შეცდომის წილი)
"""

import argparse
import json
import random
import re
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from app.config import cfg, path as cpath  # noqa: E402
from cv_speakers import rows  # noqa: E402

SRC = cpath("sources") / "cv27_ka"
OUT = cpath("data_root") / "analysis" / "cv27"
SR = 22050
TEXT_OK = re.compile(r"^[ა-ჰ .,!?:;\-—–„“\"'()]+$")
STRIP = str.maketrans("", "", ".,!?;:„“\"'—–-()… ")


def good(s):  # იგივე, რაც build_base_dataset.py-ში
    return (s.get("snr_db", 0) >= 25 and s.get("clip", 1) < 0.001 and s["audio_min"] >= 5
            and (s.get("f0_spread") is None or s["f0_spread"] <= 3) and s["votes_down_ratio"] < 0.2)


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def measure(job):
    """(clip, mp3 bytes, text) → (clip, metrics | None, 16 kHz აუდიო | None). CPU-ს მუშა პროცესში."""
    clip, data, text = job
    try:
        pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", "-ar", str(SR), "-ac", "1", "-f", "f32le", "pipe:1"],
                             input=data, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError:
        return clip, {"err": "decode"}, None
    y = np.frombuffer(pcm, dtype=np.float32)
    frame = int(SR * 0.02)
    n = len(y) // frame
    if n < 20:
        return clip, {"err": "short"}, None
    fr = y[: n * frame].reshape(n, frame)
    db = 20 * np.log10(np.sqrt(np.mean(fr ** 2, axis=1)) + 1e-9)
    real = db > -80  # ციფრული სიჩუმე (ნულები ჩანაწერის თავში/ბოლოში) ფონის ზომას არ უნდა ურევდეს
    if real.sum() < 20:
        return clip, {"err": "silent"}, None
    dbr = db[real]
    speech, noise = float(np.percentile(dbr, 90)), float(np.percentile(dbr, 10))
    voiced = np.where(db > max(noise + 12, speech - 35))[0]
    if not len(voiced):
        return clip, {"err": "silent"}, None
    # შიშინი: ჩუმი (მაგრამ არა ციფრულად ცარიელი) კადრების 4–11 kHz ენერგია მეტყველების დონესთან შედარებით
    quiet = fr[real & (db <= np.percentile(dbr, 20))]
    spec = np.abs(np.fft.rfft(quiet * np.hanning(frame), axis=1)) ** 2
    f = np.fft.rfftfreq(frame, 1 / SR)
    hf = spec[:, (f >= 4000) & (f <= 11000)].sum(axis=1).mean() / frame
    hiss = 10 * np.log10(hf + 1e-12) - speech
    a, b = voiced[0] * frame, (voiced[-1] + 1) * frame
    sec = (b - a) / SR
    letters = len(text.translate(STRIP))
    m = {"dur": round(sec, 2), "snr": round(speech - noise, 1), "hiss": round(float(hiss), 1), "speech_db": round(speech, 1),
         "clip": round(float(np.mean(np.abs(y) > 0.99)), 4), "lead": round(a / SR, 2), "trail": round((len(y) - b) / SR, 2),
         "cps": round(letters / max(sec, 0.1), 1), "zeros": round(float(1 - real.mean()), 2)}
    seg = y[max(0, a - int(0.1 * SR)): b + int(0.1 * SR)]
    y16 = np.interp(np.arange(0, len(seg), SR / 16000), np.arange(len(seg)), seg).astype(np.float32)
    return clip, m, y16


def select(cap_hours: float):
    corpus = {s["client_id"]: s for s in json.loads((SRC / "corpus.json").read_text(encoding="utf-8"))["speakers"]}
    chosen = {cid for cid, s in corpus.items() if good(s)}
    cur = {v["client_id"] for v in json.loads((cpath("datasets") / "base_ka_v1" / "speakers.json").read_text(encoding="utf-8")).values()}
    dur = {r["clip"]: int(r["duration[ms]"]) for r in rows(SRC / "clip_durations.tsv")}
    rng = random.Random(27)
    per = {}
    for kind in ("validated", "other"):
        for r in rows(SRC / f"{kind}.tsv"):
            cid, text = r["client_id"], " ".join((r["sentence"] or "").replace('"', "").split())
            if cid in chosen and TEXT_OK.match(text) and 1500 <= dur.get(r["path"], 0) <= 15500:
                per.setdefault(cid, {"validated": [], "other": []})[kind].append((r["path"], text))
    sel = {}
    for cid, k in per.items():
        tot = 0
        for kind in ("validated", "other"):
            rng.shuffle(k[kind])
            for c, t in k[kind]:
                if tot >= cap_hours * 3600_000:
                    break
                sel[c] = {"spk": f"cv_{cid[:10]}", "text": t, "kind": kind, "current": cid in cur}
                tot += dur[c]
    return sel


def bounded_map(ex, fn, it, window):
    """ex.map, რომელიც შემავალს ზარმაცად კითხულობს (Executor.map ყველაფერს ერთბაშად იღებს → მეხსიერება ივსება)."""
    from collections import deque
    q = deque()
    for x in it:
        q.append(ex.submit(fn, x))
        if len(q) >= window:
            yield q.popleft().result()
    while q:
        yield q.popleft().result()


def load_ids(f: Path) -> set:
    out = set()
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                out.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                pass
    return out


def progress(**kw):
    (OUT / "progress.json").write_text(json.dumps({**kw, "updated": time.strftime("%Y-%m-%dT%H:%M:%S")}, ensure_ascii=False),
                                       encoding="utf-8")


def stage_measure(a):
    """1) არქივიდან ამოღება + ხმის ზომები (პროცესორი, სწრაფი) → clips.jsonl, clips/*.mp3"""
    out_f = OUT / "clips.jsonl"
    done = load_ids(out_f)
    sel = select(a.cap_hours)
    todo = {c: v for c, v in sel.items() if c not in done}
    if a.limit:
        todo = dict(list(todo.items())[: a.limit])
    total, t0, n = len(sel), time.time(), 0
    print(f"ხმის ზომები: {len(todo)} (სულ {total}, გაკეთებული {len(done)})", flush=True)
    progress(stage="measure", total=total, done=len(done))
    if not todo:
        return

    def jobs():
        # ჯერ უკვე ამოღებული mp3-ები (წინა გაშვებიდან), მერე დანარჩენი — არქივიდან
        have = {c for c in todo if (OUT / "clips" / c).exists()}
        for c in have:
            yield c, (OUT / "clips" / c).read_bytes(), todo[c]["text"]
        left = len(todo) - len(have)
        if not left:
            return
        with tarfile.open(next(SRC.glob("*.tar.gz")), "r|gz") as tf:
            for mem in tf:
                c = Path(mem.name).name
                if c in todo and c not in have and mem.isfile():
                    data = tf.extractfile(mem).read()
                    (OUT / "clips" / c).write_bytes(data)
                    yield c, data, todo[c]["text"]
                    left -= 1
                    if not left:
                        return

    with out_f.open("a", encoding="utf-8") as fout, ProcessPoolExecutor(a.workers) as ex:
        for clip, m, _ in bounded_map(ex, measure, jobs(), a.workers * 16):
            fout.write(json.dumps({"id": clip, **sel[clip], **m}, ensure_ascii=False) + "\n")
            n += 1
            if n % 500 == 0:
                fout.flush()
                rate = n / (time.time() - t0)
                progress(stage="measure", total=total, done=len(done) + n, rate=round(rate, 1),
                         eta_h=round((len(todo) - n) / rate / 3600, 1))
                print(f"  {len(done) + n}/{total} · {rate:.0f}/წმ", flush=True)


MMS_FILES = {"cuda": "mms.jsonl", "cpu": "mms.cpu.jsonl"}  # ვიდეობარათი და პროცესორი პარალელურად — ცალ-ცალკე ფაილი


def mms_done() -> set:
    return set().union(*(load_ids(OUT / f) for f in MMS_FILES.values()))


def stage_mms(a):
    """2) ტექსტის დამთხვევა (MMS) შენახულ mp3-ებზე. ვიდეობარათი რიგს თავიდან მიჰყვება, პროცესორი (--cpu) — ბოლოდან;
    ერთმანეთის შედეგებს 200 ჩანაწერში ერთხელ კითხულობენ და გაკეთებულს ტოვებენ — შუაში ხვდებიან."""
    rows = [json.loads(l) for l in (OUT / "clips.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    import torch
    dev = "cuda" if torch.cuda.is_available() and not a.cpu else "cpu"
    out_f = OUT / MMS_FILES[dev]
    done = mms_done()
    todo = [r for r in rows if not r.get("err") and r["id"] not in done]
    # რიგი: ჯერ CV-ში შეუმოწმებელი (other) + თითო დინამიკზე 30 ნიმუში — ამით დინამიკების შეფასება უკვე შესაძლებელია;
    # მერე დანარჩენი დადასტურებული (მათ ტექსტი ისედაც ემთხვევა — ეს მხოლოდ ზუსტი ჩანაწერ-დონის ფილტრისთვისაა)
    seen: dict[str, int] = {}
    for r in rows:
        if r["id"] in done and not r.get("err"):
            seen[r["spk"]] = seen.get(r["spk"], 0) + 1
    first, rest = [], []
    for r in todo:
        if r.get("kind") == "other" or seen.get(r["spk"], 0) < 30:
            first.append(r)
            if r.get("kind") != "other":
                seen[r["spk"]] = seen.get(r["spk"], 0) + 1
        else:
            rest.append(r)
    todo = first + rest
    if dev == "cpu":
        todo.reverse()
    if a.limit:
        todo = todo[: a.limit]
    total = sum(1 for r in rows if not r.get("err"))
    print(f"MMS ({dev}): {len(todo)} (სულ {total}, გაკეთებული {len(done)}; რიგის პირველი ნაწილი {len(first)})", flush=True)
    if not todo:
        return
    from transformers import AutoProcessor, Wav2Vec2ForCTC
    from transformers.utils import logging as tlog
    tlog.set_verbosity_error()
    torch.set_num_threads(a.threads or max(1, min(8, int(cfg()["training"]["local_threads"]) * 2 // 3)))
    proc = AutoProcessor.from_pretrained("facebook/mms-1b-all", target_lang="kat")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", target_lang="kat", ignore_mismatched_sizes=True).eval()
    if dev == "cuda":
        model = model.half().to(dev)
    print("MMS:", dev, torch.get_num_threads(), "ნაკადი", flush=True)
    t0, n = time.time(), 0
    other = done

    def gen():
        for r in todo:
            if r["id"] not in other:
                yield r

    def src():
        for r in gen():
            yield r, (r["id"], (OUT / "clips" / r["id"]).read_bytes(), r["text"])

    with out_f.open("a", encoding="utf-8") as fout, ProcessPoolExecutor(2) as ex:
        pending = src()
        rows_q = []

        def jobs():
            for r, job in pending:
                rows_q.append(r)
                yield job
        for (cid, _, y16) in bounded_map(ex, measure, jobs(), 16):
            r = rows_q.pop(0)
            if y16 is None or r["id"] in other:
                continue
            feats = proc(y16, sampling_rate=16000, return_tensors="pt")
            iv = feats.input_values.to(dev)
            with torch.no_grad():
                hyp = proc.batch_decode(model(iv.half() if dev == "cuda" else iv).logits.argmax(-1).cpu())[0]
            ref = r["text"].translate(STRIP)
            fout.write(json.dumps({"id": r["id"], "cer": round(lev(ref, hyp.translate(STRIP)) / max(1, len(ref)), 3),
                                   "heard": hyp}, ensure_ascii=False) + "\n")
            n += 1
            if n % 100 == 0:
                fout.flush()
                rate = n / (time.time() - t0)
                if n % 200 == 0:
                    other = mms_done()  # მეორე მუშის შედეგები — გაკეთებულს აღარ ვიმეორებთ
                left = sum(1 for x in todo if x["id"] not in other)
                progress(stage="mms", total=total, done=len(other), rate=round(rate, 2), device=dev,
                         eta_h=round(left / rate / 3600, 1))
                print(f"  MMS ({dev}) {len(other)}/{total} · {rate:.2f}/წმ", flush=True)


def main():
    ap = argparse.ArgumentParser(prog="./aivoice quality")
    ap.add_argument("stage", nargs="?", choices=("measure", "mms", "all"), default="measure",
                    help="measure — ხმის ზომები (სწრაფი); mms — ტექსტის დამთხვევა (ნელი); all — ორივე")
    ap.add_argument("--cap-hours", type=float, default=4)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--cpu", action="store_true", help="MMS პროცესორზე (რიგს ბოლოდან იწყებს — ვიდეობარათთან ერთად)")
    ap.add_argument("--threads", type=int, default=0, help="პროცესორის ნაკადები (--cpu)")
    ap.add_argument("--limit", type=int, default=0, help="მხოლოდ N ჩანაწერი (სატესტოდ)")
    a = ap.parse_args()
    (OUT / "clips").mkdir(parents=True, exist_ok=True)
    if a.stage in ("measure", "all"):
        stage_measure(a)
    if a.stage in ("mms", "all"):
        stage_mms(a)
    progress(stage="done")
    print("✓ დასრულდა", flush=True)


if __name__ == "__main__":
    main()
