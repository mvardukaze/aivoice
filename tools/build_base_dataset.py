"""საერთო (multi-speaker) dataset ქართული საბაზო მოდელისთვის — Common Voice 27-დან.

გამოყენება:  nice python3 tools/build_base_dataset.py [--name base_ka_v1] [--cap-hours 3] [--include-other]
შედეგი:      <paths.datasets>/<name>/
               audio/<clip>.flac     22050 Hz მონო, სიჩუმე მოჭრილი, ხმამაღლობა გათანაბრებული
               metadata.csv          <clip>.flac|<speaker>|<ტექსტი>   (Piper-ის multi-speaker ფორმატი)
               speakers.json         დინამიკი → წუთები, F0 (ხმების შერევისთვის)
               stats.json

დინამიკების შერჩევა corpus.json-იდან (tools/cv_corpus.py): ≥5 წთ, SNR ≥ 25, ერთ ანგარიშზე ერთი ადამიანი
(F0 spread ≤ 3), უარყოფითი ხმები < 20%, clipping-ის გარეშე. ნაგულისხმევად მხოლოდ validated კლიპები.
ერთ დინამიკზე მაქსიმუმ --cap-hours (ერთმა ხმამ მოდელი რომ არ „მიითვისოს“).
სიჩუმე აქ იჭრება, რადგან წვრთნა PIPER_AUDIO_FROM_SOURCE=1 რეჟიმშია (აუდიო პირდაპირ ფაილიდან).
"""

import argparse
import json
import random
import re
import subprocess
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.config import api_url, cfg, expand, path as cpath  # noqa: E402,F401
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
from cv_speakers import rows  # noqa: E402

SRC = cpath("sources") / "cv27_ka"
SR = 22050
TEXT_OK = re.compile(r"^[ა-ჰ .,!?:;\-—–„“\"'()]+$")


def good(s):
    return (s.get("snr_db", 0) >= 25 and s.get("clip", 1) < 0.001 and s["audio_min"] >= 5
            and (s.get("f0_spread") is None or s["f0_spread"] <= 3) and s["votes_down_ratio"] < 0.2)


def process(job):
    """(clip, mp3 bytes, out_path) → (clip, sec) ან (clip, None, მიზეზი)"""
    clip, data, out = job
    try:
        pcm = subprocess.run(["ffmpeg", "-v", "error", "-i", "pipe:0", "-ar", str(SR), "-ac", "1", "-f", "f32le", "pipe:1"],
                             input=data, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError:
        return clip, None, "decode"
    y = np.frombuffer(pcm, dtype=np.float32)
    frame = int(SR * 0.02)
    n = len(y) // frame
    if n < 20:
        return clip, None, "short"
    db = 20 * np.log10(np.sqrt(np.mean(y[: n * frame].reshape(n, frame) ** 2, axis=1) + 1e-12))
    speech, noise = np.percentile(db, 90), np.percentile(db, 10)
    if speech - noise < 20:
        return clip, None, "snr"
    if np.mean(np.abs(y) > 0.99) > 0.001:
        return clip, None, "clipping"
    voiced = np.where(db > max(noise + 12, speech - 35))[0]
    if not len(voiced):
        return clip, None, "silent"
    a = max(0, voiced[0] * frame - int(0.15 * SR))
    b = min(len(y), (voiced[-1] + 1) * frame + int(0.2 * SR))
    y = y[a:b]
    sec = len(y) / SR
    if not 1.0 <= sec <= 15:
        return clip, None, "duration"
    gain = 10 ** ((-20 - speech) / 20)
    peak = float(np.max(np.abs(y))) * gain
    if peak > 0.89:
        gain *= 0.89 / peak
    sf.write(out, np.clip(y * gain, -1, 1), SR, subtype="PCM_16", format="FLAC")
    return clip, sec, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="base_ka_v1")
    ap.add_argument("--cap-hours", type=float, default=3)
    ap.add_argument("--include-other", action="store_true")
    ap.add_argument("--workers", type=int, default=10)
    a = ap.parse_args()
    out = cpath("datasets") / a.name
    (out / "audio").mkdir(parents=True, exist_ok=True)
    rng = random.Random(27)

    corpus = {s["client_id"]: s for s in json.loads((SRC / "corpus.json").read_text(encoding="utf-8"))["speakers"]}
    chosen = {cid: s for cid, s in corpus.items() if good(s)}
    dur = {r["clip"]: int(r["duration[ms]"]) for r in rows(SRC / "clip_durations.tsv")}

    per_spk = {}
    for kind in ("validated",) + (("other",) if a.include_other else ()):
        for r in rows(SRC / f"{kind}.tsv"):
            cid, text = r["client_id"], " ".join((r["sentence"] or "").replace('"', "").split())  # " CSV-ს არღვევს
            if cid in chosen and TEXT_OK.match(text) and 1500 <= dur.get(r["path"], 0) <= 15500:
                per_spk.setdefault(cid, []).append((r["path"], text))
    sel, cap_ms = {}, a.cap_hours * 3600_000
    for cid, clips in per_spk.items():
        rng.shuffle(clips)
        tot = 0
        for c, t in clips:
            if tot >= cap_ms:
                break
            sel[c] = (cid, t)
            tot += dur[c]
    print(f"დინამიკი: {len(per_spk)}, კლიპი: {len(sel)}, ~{sum(dur[c] for c in sel) / 3.6e6:.1f} სთ (მოჭრამდე)", flush=True)

    spk_name = {cid: f"cv_{cid[:10]}" for cid in per_spk}
    done = {p.stem for p in (out / "audio").glob("*.flac")}  # გაგრძელება შეწყვეტის შემდეგ
    kept, dropped = {}, {}

    def jobs():
        with tarfile.open(next(SRC.glob("*.tar.gz")), "r|gz") as tf:
            for m in tf:
                c = Path(m.name).name
                if c in sel and m.isfile():
                    stem = c.rsplit(".", 1)[0]
                    if stem in done:
                        kept[c] = sf.info(out / "audio" / f"{stem}.flac").duration
                        continue
                    yield c, tf.extractfile(m).read(), out / "audio" / f"{stem}.flac"

    with ProcessPoolExecutor(a.workers) as ex:
        for i, (c, sec, why) in enumerate(ex.map(process, jobs(), chunksize=16), 1):
            if sec is None:
                dropped[why] = dropped.get(why, 0) + 1
            else:
                kept[c] = sec
            if i % 5000 == 0:
                print(f"  {i} დამუშავდა, {sum(kept.values()) / 3600:.1f} სთ", flush=True)

    lines, spk_sec = [], {}
    for c in sorted(kept):
        cid, text = sel[c]
        lines.append(f"{c.rsplit('.', 1)[0]}.flac|{spk_name[cid]}|{text}")
        spk_sec[cid] = spk_sec.get(cid, 0) + kept[c]
    (out / "metadata.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    speakers = {spk_name[cid]: {"client_id": cid, "minutes": round(sec / 60, 1),
                                "f0_median": corpus[cid].get("f0_median"), "snr_db": corpus[cid].get("snr_db")}
                for cid, sec in sorted(spk_sec.items(), key=lambda x: -x[1])}
    (out / "speakers.json").write_text(json.dumps(speakers, ensure_ascii=False, indent=1), encoding="utf-8")
    stats = {"source": "Mozilla Common Voice Scripted Speech 27.0 (ka), CC0", "speakers": len(speakers),
             "clips": len(lines), "hours": round(sum(kept.values()) / 3600, 2), "cap_hours": a.cap_hours,
             "include_other": a.include_other, "dropped": dropped, "status": "მზადაა"}
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
