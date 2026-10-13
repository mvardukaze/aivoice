"""Common Voice-ის ყველა დინამიკის ანალიზი საერთო (multi-speaker) საბაზო მოდელისთვის.

გამოყენება:  nice python3 tools/cv_corpus.py sources/cv27_ka [--min-min 5] [--per 12] [--workers 8]
შედეგი:      <dir>/corpus.json

ასაკი/სქესი არ გამოიყენება (Common Voice-ში ხშირად არასწორია). თითო დინამიკზე:
  audio_min / validated_min / other_min — რეალურად არქივებში არსებული აუდიო, წუთებში
  snr_db, speech_db, clip — ხარისხი (--per შემთხვევითი კლიპის მედიანა)
  f0_median, f0_spread — ხმის სიმაღლე; spread > 3 ნახევარტონი ⇒ ანგარიშზე შეიძლება რამდენიმე ადამიანია
  votes_down_ratio — საზოგადოების უარყოფითი ხმების წილი
"""

import argparse
import json
import random
import subprocess
import sys
import tarfile
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from cv_speakers import rows  # noqa: E402

_TARS = {}


def _member(tar_path, name):
    if tar_path is None:  # --targz რეჟიმი: ნიმუში უკვე ამოღებულია ფაილად
        return Path(name).read_bytes()
    tf = _TARS.get(tar_path)
    if tf is None:
        tf = _TARS[tar_path] = tarfile.open(tar_path)
        tf._idx = {Path(m.name).name: m for m in tf.getmembers() if m.isfile()}
    return tf.extractfile(tf._idx[name]).read()


def measure(job):
    """(speaker, [(tar, clip), ...]) -> ხარისხის მედიანები"""
    import librosa
    spk, clips = job
    snr, lvl, clip, f0s = [], [], [], []
    for tar, name in clips:
        try:
            pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", "-ar", "16000", "-ac", "1", "-f", "f32le", "pipe:1"],
                                 input=_member(tar, name), capture_output=True, check=True).stdout
        except Exception:
            continue
        y = np.frombuffer(pcm, dtype=np.float32)
        if len(y) < 8000:
            continue
        frame = 320
        n = len(y) // frame
        rms = np.sqrt(np.mean(y[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12)
        db = 20 * np.log10(rms)
        snr.append(np.percentile(db, 90) - np.percentile(db, 10))
        lvl.append(np.percentile(db, 90))
        clip.append(float(np.mean(np.abs(y) > 0.99)))
        f0, voiced, _ = librosa.pyin(y, fmin=60, fmax=450, sr=16000, frame_length=1024)
        f0 = f0[voiced & ~np.isnan(f0)]
        if len(f0) > 20:
            f0s.append(float(np.median(f0)))
    if not snr:
        return spk, None
    f0m = float(np.median(f0s)) if f0s else None
    return spk, {
        "snr_db": round(float(np.median(snr)), 1),
        "speech_db": round(float(np.median(lvl)), 1),
        "clip": round(float(np.max(clip)), 4),
        "f0_median": round(f0m, 1) if f0m else None,
        "f0_spread": round(float(np.std(12 * np.log2(np.array(f0s) / f0m))), 2) if len(f0s) > 3 else None,
        "measured": len(snr),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("--min-min", type=float, default=5, help="მინიმუმ რამდენი წუთი აუდიო, რომ გავზომოთ")
    ap.add_argument("--per", type=int, default=12)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    src = a.src

    dur = {r["clip"]: int(r["duration[ms]"]) for r in rows(src / "clip_durations.tsv")}
    where = {}
    targz = next(src.glob("*.tar.gz"), None)
    if targz:  # CV 23+ (Mozilla Data Collective): ერთი tar.gz, ყველა კლიპი შიგნით
        where = {c: None for c in dur}
    else:
        for t in sorted((src / "audio").glob("*.tar")):
            with tarfile.open(t) as tf:
                for m in tf:
                    if m.isfile():
                        where[Path(m.name).name] = str(t)
    print("აუდიო არქივებში:", len(where), "კლიპი", flush=True)

    spk = defaultdict(lambda: {"validated_ms": 0, "other_ms": 0, "clips": [], "up": 0, "down": 0, "n_val": 0, "n_oth": 0})
    for kind in ("validated", "other"):
        for r in rows(src / f"{kind}.tsv"):
            if r["path"] not in where:
                continue
            s = spk[r["client_id"]]
            s["validated_ms" if kind == "validated" else "other_ms"] += dur.get(r["path"], 0)
            s["n_val" if kind == "validated" else "n_oth"] += 1
            s["up"] += int(r["up_votes"] or 0)
            s["down"] += int(r["down_votes"] or 0)
            s["clips"].append((where[r["path"]], r["path"], kind))

    rng = random.Random(5)
    jobs = []
    for cid, s in spk.items():
        if (s["validated_ms"] + s["other_ms"]) / 60000 >= a.min_min:
            val = [c for c in s["clips"] if c[2] == "validated"] or s["clips"]
            jobs.append((cid, [(t, n) for t, n, _ in rng.sample(val, min(a.per, len(val)))]))
    print("გასაზომი დინამიკი:", len(jobs), flush=True)

    if targz:  # ერთი გავლით tar.gz-დან მხოლოდ ნიმუშების ამოღება
        want = {n for _, clips in jobs for _, n in clips}
        qdir = src / "qc_samples"
        qdir.mkdir(exist_ok=True)
        missing = {n for n in want if not (qdir / n).exists()}
        if missing:
            with tarfile.open(targz, "r|gz") as tf:
                for m in tf:
                    n = Path(m.name).name
                    if n in missing:
                        (qdir / n).write_bytes(tf.extractfile(m).read())
                        missing.discard(n)
                        if not missing:
                            break
        print("ნიმუშები ამოღებულია:", len(want), flush=True)
        jobs = [(cid, [(None, str(qdir / n)) for _, n in clips]) for cid, clips in jobs]

    quality = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (cid, q) in enumerate(ex.map(measure, jobs, chunksize=2), 1):
            quality[cid] = q
            if i % 25 == 0:
                print(f"  {i}/{len(jobs)}", flush=True)

    out = []
    for cid, s in spk.items():
        rec = {"id": cid[:12], "client_id": cid,
               "audio_min": round((s["validated_ms"] + s["other_ms"]) / 60000, 1),
               "validated_min": round(s["validated_ms"] / 60000, 1),
               "other_min": round(s["other_ms"] / 60000, 1),
               "clips_validated": s["n_val"], "clips_other": s["n_oth"],
               "votes_down_ratio": round(s["down"] / max(1, s["up"] + s["down"]), 3)}
        if quality.get(cid):
            rec.update(quality[cid])
        out.append(rec)
    out.sort(key=lambda r: -r["audio_min"])
    (src / "corpus.json").write_text(json.dumps({"speakers": out}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("შენახულია:", src / "corpus.json")


if __name__ == "__main__":
    main()
