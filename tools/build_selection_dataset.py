"""შერჩევიდან (tools/select_best.py) Piper-ის multi-speaker dataset.

    ./aivoice build-selection base_ka_v2 [--tar]
შედეგი: <paths.datasets>/<name>/  audio/*.flac (22050 Hz, სიჩუმე მოჭრილი, ხმამაღლობა გათანაბრებული),
        metadata.csv (<clip>.flac|<speaker>|<ტექსტი>), speakers.json, stats.json; --tar → <paths.gcp>/<name>.tar
გაგრძელებადია: უკვე გაკეთებული flac-ები გამოტოვდება; metadata ყოველ ჯერზე თავიდან იწერება — ასე თქვენ მიერ
ამოღებული ჩანაწერები/დინამიკები (rejected, rejected_speakers) ბოლო აწყობაზე აღარ შევა.
აუდიო მუშავდება build_base_dataset.process-ით (იგივე, რაც v1-ში).
"""

import argparse
import json
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from app.config import path as cpath  # noqa: E402
from build_base_dataset import process  # noqa: E402
from clip_quality import OUT as QDIR, bounded_map  # noqa: E402

import soundfile as sf  # noqa: E402


def main():
    ap = argparse.ArgumentParser(prog="./aivoice build-selection")
    ap.add_argument("name")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--tar", action="store_true", help="Google Cloud-ისთვის არქივი")
    a = ap.parse_args()
    sel = json.loads((ROOT / "data" / "selections" / f"{a.name}.json").read_text(encoding="utf-8"))
    rej, rej_spk = set(sel.get("rejected", [])), set(sel.get("rejected_speakers", []))
    info = {}
    for line in (QDIR / "clips.jsonl").read_text(encoding="utf-8").splitlines():
        r = json.loads(line)
        info[r["id"]] = r
    clips = [c for c in sel["clips"] if c not in rej and info[c]["spk"] not in rej_spk]
    out = cpath("datasets") / a.name
    (out / "audio").mkdir(parents=True, exist_ok=True)
    done = {p.stem for p in (out / "audio").glob("*.flac")}
    todo = [c for c in clips if c.rsplit(".", 1)[0] not in done]
    print(f"{a.name}: {len(clips)} ჩანაწერი (ამოღებული {len(sel['clips']) - len(clips)}), დასამუშავებელი {len(todo)}", flush=True)

    def jobs():
        for c in todo:
            yield c, (QDIR / "clips" / c).read_bytes(), out / "audio" / f"{c.rsplit('.', 1)[0]}.flac"

    dropped = {}
    with ProcessPoolExecutor(a.workers) as ex:
        for i, (c, sec, why) in enumerate(bounded_map(ex, process, jobs(), a.workers * 16), 1):
            if sec is None:
                dropped[why] = dropped.get(why, 0) + 1
            if i % 2000 == 0:
                print(f"  {i}/{len(todo)}", flush=True)

    # გაგრძელება: წინა მოდელის დინამიკები (base: <შერჩევა>) იმავე რიგით პირველები — Piper-ი ID-ებს ასე ანიჭებს
    base = sel.get("base")
    if base:
        order = json.loads((ROOT / "data" / "selections" / f"{base}.json").read_text(encoding="utf-8"))["speaker_order"]
        rank = {s: i for i, s in enumerate(order)}
        clips.sort(key=lambda c: (rank.get(info[c]["spk"], len(rank)), c))
    lines, spk_sec = [], {}
    for c in clips:
        stem = c.rsplit(".", 1)[0]
        f = out / "audio" / f"{stem}.flac"
        if not f.exists():
            continue
        r = info[c]
        lines.append(f"{stem}.flac|{r['spk']}|{r['text']}")
        spk_sec[r["spk"]] = spk_sec.get(r["spk"], 0) + sf.info(f).duration
    (out / "metadata.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    spk = sel["speakers"]
    speakers = {s: {"minutes": round(sec / 60, 1), "f0_median": spk.get(s, {}).get("f0")}
                for s, sec in sorted(spk_sec.items(), key=lambda x: -x[1])}
    (out / "speakers.json").write_text(json.dumps(speakers, ensure_ascii=False, indent=1), encoding="utf-8")
    stats = {"source": "Mozilla Common Voice Scripted Speech 27.0 (ka), CC0 — ხარისხით შერჩეული (tools/select_best.py)",
             "selection": a.name, "speakers": len(speakers), "clips": len(lines),
             "hours": round(sum(spk_sec.values()) / 3600, 2), "dropped": dropped,
             "rejected_by_ear": {"clips": len(rej), "speakers": sorted(rej_spk)}, "rules": sel["rules"], "status": "მზადაა"}
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({k: stats[k] for k in ("speakers", "clips", "hours", "dropped")}, ensure_ascii=False), flush=True)
    if a.tar:
        tar = cpath("gcp") / f"{a.name}.tar"
        tmp = tar.with_suffix(".part")
        with tarfile.open(tmp, "w") as t:
            for name in ("metadata.csv", "speakers.json", "stats.json"):
                t.add(out / name, arcname=f"{a.name}/{name}")
            for l in lines:
                t.add(out / "audio" / l.split("|", 1)[0], arcname=f"{a.name}/audio/{l.split('|', 1)[0]}")
        tmp.replace(tar)
        print(f"არქივი: {tar} ({tar.stat().st_size / 1e9:.1f} GB)", flush=True)


if __name__ == "__main__":
    main()
