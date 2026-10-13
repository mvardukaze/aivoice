"""ცალკე ხმის (დისტილაციის) ტექსტები: ~3000 მრავალფეროვანი ქართული წინადადება Common Voice 27-იდან (CC0).

    python3 tools/make_voice_texts.py [--n 3000] [--out data/prompts/voice_texts_v1.json]

კვოტები (ხმამ ეს ადგილები განსაკუთრებით კარგად უნდა ისწავლოს): „ღ“-იანი, „რ+თანხმოვანი“, 3+ თანხმოვნის
შეერთებები, კითხვები და ძახილი; დანარჩენი — შემთხვევითი. ლათინური/ციფრები გამორიცხულია (espeak მათ სხვანაირად კითხულობს).
"""

import argparse
import csv
import json
import random
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = Path("/mnt/d/aivoice/sources/cv27_ka/validated_sentences.tsv")
GEO = re.compile(r"^[ა-ჰ\s,.!?:;„“\"'—–\-()]+$")
CONS = "ბგდვზთკლმნპჟრსტფქღყშჩცძწჭხჯჰ"
CLUSTER3 = re.compile(f"[{CONS}]{{3,}}")
R_CLUSTER = re.compile(f"რ[{CONS}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--out", default=str(ROOT / "data" / "prompts" / "voice_texts_v1.json"))
    a = ap.parse_args()
    seen, pool = set(), []
    with SRC.open(encoding="utf-8") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            s = " ".join(row["sentence"].split())
            words = s.split()
            letters = len(re.findall("[ა-ჰ]", s))
            # CV-ში არის დაზიანებული სტრიქონები („????? ?????“) — GEO მათ ატარებს, ამიტომ ასოების წილიც მოწმდება
            if not (25 <= len(s) <= 140 and 4 <= len(words) <= 20) or not GEO.match(s) or s in seen \
                    or letters < 0.8 * len(s.replace(" ", "")):
                continue
            seen.add(s)
            pool.append(s)
    rng = random.Random(42)
    rng.shuffle(pool)
    quotas = [  # (სახელი, პირობა, რამდენი)
        ("ღ", lambda s: "ღ" in s, int(a.n * 0.20)),
        ("რ+თანხმოვანი", lambda s: bool(R_CLUSTER.search(s)), int(a.n * 0.15)),
        ("3+ თანხმოვანი", lambda s: bool(CLUSTER3.search(s)), int(a.n * 0.15)),
        ("კითხვა", lambda s: s.endswith("?"), int(a.n * 0.08)),
        ("ძახილი", lambda s: s.endswith("!"), int(a.n * 0.05)),
    ]
    chosen, stats = [], {}
    taken = set()
    for name, cond, k in quotas:
        got = [s for s in pool if s not in taken and cond(s)][:k]
        taken.update(got)
        chosen += got
        stats[name] = len(got)
    rest = [s for s in pool if s not in taken][: a.n - len(chosen)]
    chosen += rest
    stats["დანარჩენი"] = len(rest)
    rng.shuffle(chosen)
    Path(a.out).write_text(json.dumps({"source": "Common Voice 27 validated_sentences (CC0)", "stats": stats,
                                       "sentences": chosen}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(len(pool), "candidates →", len(chosen), stats)


if __name__ == "__main__":
    main()
