"""კანდიდატი დინამიკების ნიმუშები + ხმის გაზომვა.

გამოყენება:  python3 tools/cv_samples.py sources/cv17_ka [--min-hours 0.6] [--per 8]
შედეგი:      <dir>/samples/<id>/*.mp3  და  speakers.json-ში თითო კანდიდატს ემატება:
             samples[], f0_median (Hz), f0_spread (ნახევარტონი), snr_db, loudness_db

f0_median — ხმის სიმაღლე: ~85–155 Hz კაცი, ~165–255 Hz ქალი, 250+ ბავშვი/მაღალი ხმა.
f0_spread — კლიპებს შორის F0-ის გაფანტულობა. დიდი მნიშვნელობა (>3) შეიძლება ნიშნავდეს,
            რომ ერთ ანგარიშზე რამდენიმე ადამიანი იწერს — ასეთი დინამიკი მოსასმენია.
snr_db   — მეტყველება / ფონური ხმაური. 30+ კარგია, 20-ზე ნაკლები — ხმაურიანი ჩანაწერი.

გაშვება სისტემურ python3-ით (librosa, numpy იქ არის).
"""

import argparse
import json
import random
import tarfile
import tempfile
from pathlib import Path

import librosa
import numpy as np

from cv_speakers import rows


def measure(path: Path):
    y, sr = librosa.load(str(path), sr=16000, mono=True)
    if len(y) < sr // 2:
        return None
    rms = librosa.feature.rms(y=y, frame_length=512, hop_length=256)[0]
    rms = rms[rms > 1e-6]
    if len(rms) < 10:
        return None
    noise = np.percentile(rms, 10)
    speech = np.percentile(rms, 90)
    f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=500, sr=sr, frame_length=1024)
    f0 = f0[voiced & ~np.isnan(f0)]
    return {
        "snr_db": float(20 * np.log10(speech / max(noise, 1e-6))),
        "loudness_db": float(20 * np.log10(speech)),
        "f0": float(np.median(f0)) if len(f0) > 20 else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("--min-hours", type=float, default=0.6)
    ap.add_argument("--per", type=int, default=8)
    args = ap.parse_args()

    src = args.src
    data = json.loads((src / "speakers.json").read_text(encoding="utf-8"))
    cands = [s for s in data["speakers"] if s["hours"] >= args.min_hours and s["down_ratio"] < 0.2]
    by_cid = {s["client_id"]: s for s in cands}

    # თითო კანდიდატის კლიპები და წინადადებები
    clips, sentence = {}, {}
    for kind in ("validated", "other"):
        for r in rows(src / f"{kind}.tsv"):
            if r["client_id"] in by_cid:
                clips.setdefault(r["client_id"], []).append(r["path"])
                sentence[r["path"]] = r["sentence"]

    # რომელი კლიპები არის რეალურად აუდიო-არქივებში
    tars = sorted((src / "audio").glob("*.tar"))
    in_tar = {}
    for t in tars:
        with tarfile.open(t) as tf:
            for m in tf.getmembers():
                if m.isfile():
                    in_tar[Path(m.name).name] = (t, m.name)

    rng = random.Random(42)
    wanted = {}
    for cid, paths in clips.items():
        avail = [p for p in paths if p in in_tar]
        by_cid[cid]["audio_clips"] = len(avail)
        for p in rng.sample(avail, min(args.per, len(avail))):
            wanted[p] = cid

    out_root = src / "samples"
    per_tar = {}
    for p, cid in wanted.items():
        per_tar.setdefault(in_tar[p][0], []).append((in_tar[p][1], p, cid))
    for t, items in per_tar.items():
        with tarfile.open(t) as tf:
            for member, p, cid in items:
                d = out_root / by_cid[cid]["id"]
                d.mkdir(parents=True, exist_ok=True)
                (d / p).write_bytes(tf.extractfile(member).read())

    for s in cands:
        d = out_root / s["id"]
        files = sorted(d.glob("*.mp3")) if d.exists() else []
        ms = [(f, measure(f)) for f in files]
        ms = [(f, m) for f, m in ms if m]
        s["samples"] = [{"file": f"{s['id']}/{f.name}", "text": sentence.get(f.name, "")} for f, _ in ms]
        f0s = [m["f0"] for _, m in ms if m["f0"]]
        s["f0_median"] = round(float(np.median(f0s)), 1) if f0s else None
        s["f0_spread"] = round(float(np.std(12 * np.log2(np.array(f0s) / np.median(f0s)))), 2) if len(f0s) > 2 else None
        s["snr_db"] = round(float(np.median([m["snr_db"] for _, m in ms])), 1) if ms else None
        s["loudness_db"] = round(float(np.median([m["loudness_db"] for _, m in ms])), 1) if ms else None
        print(f"{s['id']} {s['hours']:6.2f}h audio={s.get('audio_clips', 0):5} f0={s['f0_median']} "
              f"spread={s['f0_spread']} snr={s['snr_db']} {s['gender']} {s['age']}")

    data["candidates_min_hours"] = args.min_hours
    (src / "speakers.json").write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
