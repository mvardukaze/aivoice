"""საუკეთესო ჩანაწერების შერჩევა ახალი საბაზო dataset-ისთვის (Common Voice 27, ხარისხის შემოწმების შემდეგ).

    ./aivoice select [--hours 40] [--name base_ka_v2] [--exclude-from base_ka_v2]
შედეგი: data/selections/<name>.json — {params, stats, speakers: {spk: {minutes, clips}}, clips: [id…]}
(პანელი: მონაცემები → ხარისხი → „შერჩეული“ — შემთხვევითი მოსმენა და შემოწმება)

წესები (რიგით):
  1. დინამიკი: თქვენი „ამოვრიცხოთ“ / ავტომატური ამოგდება — არა.
  2. ჩანაწერი: MMS ტექსტს ემთხვევა (cer ≤ 0.10, ტექსტი ≥ 15 ასო — მოკლეზე MMS არასანდოა), არცერთი ცუდის ნიშანი,
     ხმაური SNR ≥ 30, შიშინი ≤ −15, ხმამაღლობა −24…−8 dBFS, სიჩქარე 8–16 ასო/წმ, 2–12 წამი.
  3. ქულა: MMS (მთავარი) + SNR + შიშინი — საუკეთესოები ჯერ.
  4. თითო დინამიკზე ზღვარი (თავად შეირჩევა, რომ ჯამი --hours იყოს) — ერთმა ხმამ dataset არ „მიითვისოს“;
     ≥ 3 წთ-ზე ნაკლები დინამიკი არ შედის (მისი „ანაბეჭდი“ კარგად ვერ ისწავლება).
--exclude-from: წინა შერჩევის ჩანაწერები გამოტოვდება; უკვე გაშვებული („launched“) შერჩევები — ყოველთვის.
"""

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.quality_api import load, read_json, DECISIONS  # noqa: E402

OUT = ROOT / "data" / "selections"
RULES = {"cer_max": 0.10, "min_letters": 15, "snr_min": 30, "hiss_max": -15, "db": (-24, -8), "cps": (8, 16),
         "dur": (2, 12), "min_speaker_min": 3}


def ok(c) -> bool:
    r = RULES
    return (not c.get("bad") and c.get("cer") is not None and c["cer"] <= r["cer_max"]
            and len(c.get("text", "")) >= r["min_letters"] and c.get("snr", 0) >= r["snr_min"]
            and c.get("hiss", 0) <= r["hiss_max"] and r["db"][0] <= c.get("speech_db", -99) <= r["db"][1]
            and r["cps"][0] <= c.get("cps", 0) <= r["cps"][1] and r["dur"][0] <= c.get("dur", 0) <= r["dur"][1])


def score(c) -> float:
    """ნაკლები = უკეთესი. MMS წონით მთავარია (1% შეცდომა ≈ 5 dB ხმაური)."""
    return c["cer"] * 500 - min(c["snr"], 60) + max(c["hiss"], -60) * 0.5


def main():
    ap = argparse.ArgumentParser(prog="./aivoice select")
    ap.add_argument("--hours", type=float, default=40)
    ap.add_argument("--name", default="base_ka_v2")
    ap.add_argument("--exclude-from", nargs="*", default=[])
    a = ap.parse_args()

    d = load()
    dec = read_json(DECISIONS, {}) or {}
    rows = {r["spk"]: r for r in __import__("app.quality_api", fromlist=["summary"]).summary()["speakers"]}
    skip = set()
    # უკვე გაშვებული შერჩევების ჩანაწერები ავტომატურად გამოტოვდება (შემდეგი 40 სთ — ახალი ჩანაწერებით)
    launched = [f.stem for f in OUT.glob("*.json") if f.stem != a.name and json.loads(f.read_text(encoding="utf-8")).get("launched")]
    for name in sorted(set(a.exclude_from) | set(launched)):
        skip |= set(json.loads((OUT / f"{name}.json").read_text(encoding="utf-8"))["clips"])
    per = {}
    for spk, clips in d["clips"].items():
        r = rows.get(spk, {})
        if r.get("final") != "include" or dec.get(spk, {}).get("decision") == "exclude":
            continue
        good = sorted((c for c in clips if c["id"] not in skip and ok(c)), key=score)
        if good:
            per[spk] = good

    def take(cap_min):
        sel = {}
        for spk, good in per.items():
            out, sec = [], 0.0
            for c in good:
                if sec >= cap_min * 60:
                    break
                out.append(c)
                sec += c["dur"]
            if sec >= RULES["min_speaker_min"] * 60:
                sel[spk] = out
        return sel, sum(c["dur"] for cs in sel.values() for c in cs) / 3600

    lo, hi = 3.0, 240.0  # დინამიკის ზღვარი (წთ) — ორობითი ძებნა, რომ ჯამი --hours იყოს
    for _ in range(30):
        mid = (lo + hi) / 2
        _, h = take(mid)
        lo, hi = (mid, hi) if h < a.hours else (lo, mid)
    sel, hours = take(hi)
    f0 = {r["spk"]: r.get("f0") for r in rows.values()}
    speakers = {spk: {"minutes": round(sum(c["dur"] for c in cs) / 60, 1), "clips": len(cs), "f0": f0.get(spk),
                      "current": rows[spk]["current"]} for spk, cs in sel.items()}
    low = [s for s, v in speakers.items() if (v["f0"] or 999) < 165]
    stats = {"hours": round(hours, 1), "clips": sum(v["clips"] for v in speakers.values()), "speakers": len(speakers),
             "low_speakers": len(low), "low_hours": round(sum(speakers[s]["minutes"] for s in low) / 60, 1),
             "new_speakers": sum(1 for v in speakers.values() if not v["current"]),
             "cap_minutes": round(hi, 1), "candidates": sum(len(v) for v in per.values()),
             "mean_cer": round(sum(c["cer"] for cs in sel.values() for c in cs) / max(1, sum(len(cs) for cs in sel.values())), 4)}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{a.name}.json").write_text(json.dumps({
        "name": a.name, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "base": sorted(launched)[-1] if launched else None,  # გაგრძელება: ამ მოდელის დინამიკების რიგი (build-selection) "rules": RULES, "params": vars(a), "stats": stats,
        "speakers": speakers, "clips": sorted(c["id"] for cs in sel.values() for c in cs)}, ensure_ascii=False, indent=1),
        encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
