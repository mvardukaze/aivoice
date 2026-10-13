"""ერთი დინამიკის Piper-ის dataset-ი Common Voice-დან.

გამოყენება:
    python3 tools/build_dataset.py <slot>            # voices/catalog.json-ის slot, მაგ. f_old_01
    python3 tools/build_dataset.py <slot> --max-hours 10
    python3 tools/build_dataset.py <slot> --include-other   # + შეუმოწმებელი კლიპები, whisper-ით გადამოწმებული

რას აკეთებს:
  1. იღებს დინამიკის validated კლიპებს (საზოგადოების მიერ დადასტურებული ტექსტი)
  2. mp3 → 22050 Hz მონო wav (ffmpeg)
  3. ფილტრავს: ხანგრძლივობა 1.5–15 წმ, SNR ≥ 20 dB, clipping
  4. ხმამაღლობას ერთ დონეზე ასწორებს (მეტყველების RMS → -20 dBFS, პიკი ≤ -1 dBFS)
     — ხმადაბლა ჩაწერილ დინამიკებს (4dc613ce3558, 7300a92b4169) ეს სჭირდებათ
  5. --include-other: Common Voice-ის „other“ (ჯერ შეუმოწმებელი) კლიპებსაც იღებს, ოღონდ
     whisper large-v3 (CPU) ამოიცნობს და ტექსტთან სიმბოლოების შეცდომა (CER) ≤ MAX_CER უნდა იყოს.
     ქართულზე whisper თავადაც ცდება, ამიტომ ზღვარი ლმობიერია.
  6. წერს datasets/<slot>/metadata.csv (wav|ტექსტი), wav/ და stats.json

გაშვება სისტემურ python3-ით (numpy, soundfile, librosa).
"""

import argparse
import json
import subprocess
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from cv_speakers import rows  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
SR = 22050
TARGET_RMS_DB = -20.0
MAX_CER = 0.30
PEAK_DB = -1.0


def decode(data: bytes) -> np.ndarray:
    pcm = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", "pipe:0", "-ar", str(SR), "-ac", "1", "-f", "f32le", "pipe:1"],
        input=data, capture_output=True, check=True).stdout
    return np.frombuffer(pcm, dtype=np.float32)


def process(args):
    """(name, mp3 bytes, out_path) -> stats dict ან None (გაფილტრულია)."""
    name, data, out = args
    y = decode(data)
    dur = len(y) / SR
    if not 1.5 <= dur <= 15:
        return {"name": name, "drop": "duration"}
    frame = 512
    n = len(y) // frame
    rms = np.sqrt(np.mean(y[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
    speech, noise = np.percentile(rms, 90), np.percentile(rms, 10)
    snr = 20 * np.log10(speech / max(noise, 1e-6))
    if snr < 20:
        return {"name": name, "drop": "snr"}
    if np.mean(np.abs(y) > 0.99) > 0.001:
        return {"name": name, "drop": "clipping"}
    gain = 10 ** ((TARGET_RMS_DB - 20 * np.log10(speech)) / 20)
    peak = np.max(np.abs(y)) * gain
    if peak > 10 ** (PEAK_DB / 20):
        gain *= 10 ** (PEAK_DB / 20) / peak
    sf.write(out, np.clip(y * gain, -1, 1), SR, subtype="PCM_16")
    return {"name": name, "dur": dur, "snr": float(snr), "gain_db": float(20 * np.log10(gain))}


def cer(ref: str, hyp: str) -> float:
    """სიმბოლოების შეცდომის წილი (პუნქტუაციის გარეშე)."""
    norm = lambda t: "".join(c for c in t.lower() if c.isalnum() or c == " ").split()
    a, b = " ".join(norm(ref)), " ".join(norm(hyp))
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1] / max(1, len(a))


def verify(kept, sentences, wav_dir, threads=8):
    """whisper-ით ამოწმებს „other“ კლიპებს; აბრუნებს (დარჩენილი, ამოგდებული რაოდენობა)."""
    from faster_whisper import WhisperModel
    model = WhisperModel("large-v3", device="cpu", compute_type="int8", cpu_threads=threads)
    out, bad = [], 0
    for r in kept:
        if not r.get("other"):
            out.append(r)
            continue
        wav = wav_dir / r["name"].replace(".mp3", ".wav")
        segs, _ = model.transcribe(str(wav), language="ka", beam_size=1, vad_filter=False)
        r["cer"] = round(cer(sentences[r["name"]], " ".join(s.text for s in segs)), 3)
        if r["cer"] <= MAX_CER:
            out.append(r)
        else:
            bad += 1
            wav.unlink(missing_ok=True)
    return out, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slot")
    ap.add_argument("--max-hours", type=float, default=10.0)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--include-other", action="store_true")
    ap.add_argument("--whisper-threads", type=int, default=8)
    args = ap.parse_args()

    voice = next(v for v in json.loads((ROOT / "voices/catalog.json").read_text(encoding="utf-8"))["voices"]
                 if v["id"] == args.slot)
    source, spk = voice["speaker"].split("/")
    src = ROOT / "sources" / source

    sentences = {}
    for r in rows(src / "validated.tsv"):
        if r["client_id"].startswith(spk):
            sentences[r["path"]] = " ".join(r["sentence"].split())
    other = set()
    if args.include_other:
        for r in rows(src / "other.tsv"):
            if r["client_id"].startswith(spk) and r["path"] not in sentences:
                sentences[r["path"]] = " ".join(r["sentence"].split())
                other.add(r["path"])

    out_dir = ROOT / "datasets" / args.slot
    wav_dir = out_dir / "wav"
    wav_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "stats.json").write_text(json.dumps(
        {"speaker": voice["speaker"], "status": "მზადდება"}, ensure_ascii=False), encoding="utf-8")

    def jobs():
        for t in sorted((src / "audio").glob("*.tar")):
            with tarfile.open(t) as tf:
                for m in tf:
                    name = Path(m.name).name
                    if m.isfile() and name in sentences:
                        yield name, tf.extractfile(m).read(), wav_dir / name.replace(".mp3", ".wav")

    kept, dropped, total = [], {}, 0.0
    limit = args.max_hours * 3600
    with ProcessPoolExecutor(args.workers) as ex:
        for res in ex.map(process, jobs(), chunksize=8):
            if "drop" in res:
                dropped[res["drop"]] = dropped.get(res["drop"], 0) + 1
                continue
            if total >= limit:
                (wav_dir / res["name"].replace(".mp3", ".wav")).unlink(missing_ok=True)
                dropped["max_hours"] = dropped.get("max_hours", 0) + 1
                continue
            res["other"] = res["name"] in other
            kept.append(res)
            total += res["dur"]

    if other:
        kept, bad = verify(kept, sentences, wav_dir, args.whisper_threads)
        dropped["whisper_cer"] = bad
        total = sum(r["dur"] for r in kept)

    kept.sort(key=lambda r: r["name"])
    with (out_dir / "metadata.csv").open("w", encoding="utf-8") as f:
        for r in kept:
            f.write(f"{r['name'].replace('.mp3', '.wav')}|{sentences[r['name']]}\n")

    stats = {
        "speaker": voice["speaker"],
        "status": "მზადაა",
        "hours": round(total / 3600, 2),
        "avg_sec": round(total / max(1, len(kept)), 1),
        "snr_median": round(float(np.median([r["snr"] for r in kept])), 1) if kept else None,
        "gain_db_median": round(float(np.median([r["gain_db"] for r in kept])), 1) if kept else None,
        "candidates": len(sentences),
        "other_used": sum(1 for r in kept if r.get("other")),
        "dropped": dropped,
    }
    (out_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(args.slot, len(kept), "clips", json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
