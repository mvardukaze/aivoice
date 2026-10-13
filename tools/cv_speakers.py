"""Common Voice-ის დინამიკების ანალიზი: ვის რამდენი საათი ჩანაწერი აქვს.

გამოყენება:  python tools/cv_speakers.py sources/cv27_ka
შედეგი:      <dir>/speakers.json (პანელი ამას კითხულობს)

validated = საზოგადოების მიერ დადასტურებული კლიპები, other = ჯერ შეუმოწმებელი
(Piper-ისთვის ისიც გამოსადეგია — whisper-ით გადავამოწმებთ).
invalidated კლიპები არ ითვლება.
"""

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

csv.field_size_limit(10**9)


def rows(path: Path):
    with path.open(encoding="utf-8", newline="") as f:
        yield from csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)


def main(src: Path, min_hours: float = 0.3):
    durations = {r["clip"]: int(r["duration[ms]"]) for r in rows(src / "clip_durations.tsv")}

    # კლიპის მდებარეობა აუდიო-არქივებში (audio/ka/<split>/)
    split_of = {}
    for split in ("train", "dev", "test", "other", "invalidated"):
        p = src / f"{split}.tsv"
        if p.exists():
            for r in rows(p):
                split_of[r["path"]] = split

    spk = defaultdict(lambda: {"validated_ms": 0, "other_ms": 0, "validated": 0, "other": 0,
                               "down": 0, "up": 0, "gender": Counter(), "age": Counter(),
                               "splits": Counter(), "chars": 0})
    for kind in ("validated", "other"):
        for r in rows(src / f"{kind}.tsv"):
            s = spk[r["client_id"]]
            ms = durations.get(r["path"], 0)
            s[f"{kind}_ms"] += ms
            s[kind] += 1
            s["up"] += int(r["up_votes"] or 0)
            s["down"] += int(r["down_votes"] or 0)
            s["chars"] += len(r["sentence"])
            if r["gender"]:
                s["gender"][r["gender"]] += 1
            if r["age"]:
                s["age"][r["age"]] += 1
            s["splits"][split_of.get(r["path"], "?")] += 1

    out = []
    for cid, s in spk.items():
        total_ms = s["validated_ms"] + s["other_ms"]
        clips = s["validated"] + s["other"]
        out.append({
            "id": cid[:12],
            "client_id": cid,
            "hours": round(total_ms / 3.6e6, 2),
            "validated_hours": round(s["validated_ms"] / 3.6e6, 2),
            "clips": clips,
            "validated": s["validated"],
            "avg_sec": round(total_ms / clips / 1000, 1) if clips else 0,
            "gender": s["gender"].most_common(1)[0][0] if s["gender"] else "",
            "age": s["age"].most_common(1)[0][0] if s["age"] else "",
            "down_ratio": round(s["down"] / max(1, s["up"] + s["down"]), 3),
            "splits": dict(s["splits"]),
        })
    out.sort(key=lambda x: -x["hours"])

    all_ms = sum(durations.get(r["path"], 0) for k in ("validated", "other") for r in rows(src / f"{k}.tsv"))
    summary = {
        "source": "Mozilla Common Voice 17.0 — ka (mirror: fsicoli/common_voice_17_0), CC0",
        "speakers": len(out),
        "hours": round(all_ms / 3.6e6, 1),
        "speakers_over": {str(h): sum(1 for x in out if x["hours"] >= h) for h in (0.5, 1, 2, 5)},
    }
    result = {"summary": summary, "speakers": [x for x in out if x["hours"] >= min_hours]}
    (src / "speakers.json").write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    return result


if __name__ == "__main__":
    res = main(Path(sys.argv[1]))
    print(json.dumps(res["summary"], ensure_ascii=False, indent=1))
    print(f"{'id':13} {'hours':>6} {'valid':>6} {'clips':>6} {'avg':>5} {'gender':22} {'age':10} {'down':>5}  splits")
    for x in res["speakers"][:40]:
        print(f"{x['id']:13} {x['hours']:6.2f} {x['validated_hours']:6.2f} {x['clips']:6} {x['avg_sec']:5.1f} "
              f"{x['gender'][:22]:22} {x['age']:10} {x['down_ratio']:5.2f}  {x['splits']}")
