"""ჩასაწერი წინადადებების სია: ქართული ბგერების მაქსიმალური დაფარვით.

გამოყენება (.venv-ით — piper-ის espeak იქ არის):
    .venv/bin/python tools/make_prompts.py [--n 1200]
შედეგი: data/prompts/ka_v1.json

წყარო: Common Voice-ის validated_sentences.tsv (CC0).
მეთოდი: ხარბი შერჩევა — ყოველ ნაბიჯზე ის წინადადება, რომელიც ყველაზე მეტ ახალ ფონემა-წყვილს
(diphone) ამატებს, სიგრძეზე ნორმირებული. ~10% კითხვითი და ~8% ძახილის წინადადებაა,
რომ ინტონაციაც მრავალფეროვანი იყოს.
"""

import argparse
import csv
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

from piper.phonemize_espeak import EspeakPhonemizer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.config import path as cpath  # noqa: E402

SRC = cpath("sources") / "cv27_ka" / "validated_sentences.tsv"
OUT = ROOT / "data" / "prompts" / "ka_v1.json"
ALLOWED = re.compile(r"^[ა-ჰ .,!?:;\-—–„“\"'()]+$")

csv.field_size_limit(10**9)


def phones(p, text):
    seq = [ph for s in p.phonemize("ka", text) for ph in s if ph not in "ˈˌ"]
    return ["#" if ph in " ,.!?;:" else ph for ph in seq]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1200)
    args = ap.parse_args()
    rng = random.Random(7)

    seen, pool = set(), []
    with SRC.open(encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE):
            s = " ".join((r["sentence"] or "").split())
            words = s.split()
            if not (4 <= len(words) <= 16 and 25 <= len(s) <= 130) or not ALLOWED.match(s):
                continue
            key = re.sub(r"[^ა-ჰ ]", "", s)
            if key in seen or not s[-1] in ".!?":
                continue
            seen.add(key)
            pool.append(s)
    # ბუნებრიობა: სიტყვების ≥90% კორპუსში ხშირი (≥5-ჯერ) უნდა იყოს — უცხო სახელები/ტერმინები გამოირიცხება
    # (ხარბი შერჩევა სხვაგვარად სწორედ მათ ეტანება, რადგან იშვიათ ბგერათა წყვილებს შეიცავს)
    wf = Counter(w for s in pool for w in re.findall(r"[ა-ჰ]+", s))
    def natural(s):
        ws = re.findall(r"[ა-ჰ]+", s)
        return bool(ws) and sum(wf[w] >= 5 for w in ws) / len(ws) >= 0.9
    pool = [s for s in pool if natural(s) and s.count("„") == s.count("“")]
    rng.shuffle(pool)
    by_end = {k: [s for s in pool if s[-1] == k] for k in "?!"}
    print("ბუნებრივი:", len(pool), "| ?:", len(by_end["?"]), "| !:", len(by_end["!"]))
    pool = by_end["?"][:4000] + by_end["!"][:4000] + [s for s in pool if s[-1] == "."][:32000]
    p = EspeakPhonemizer()
    items = []
    for s in pool:
        ph = phones(p, s)
        items.append((s, set(zip(ph, ph[1:])), Counter(ph)))
    print("კანდიდატი:", len(items))

    quotas = {"?": int(args.n * 0.10), "!": int(args.n * 0.08)}
    covered, chosen, phone_count = Counter(), [], Counter()
    done_q = Counter()

    def score(i):
        s, di, _ = items[i]
        gain = sum(1 / (1 + covered[d]) for d in di)  # იშვიათ წყვილებს მეტი წონა
        sc = gain / (len(s) ** 0.5)
        return sc

    # „ზარმაცი“ ხარბი შერჩევა: ქულა მხოლოდ მცირდება, ამიტომ გადათვლა მხოლოდ რიგის სათავეს სჭირდება
    import heapq
    def run(indices, limit):
      heap = [(-score(i), i) for i in indices]
      heapq.heapify(heap)
      target = len(chosen) + limit
      while len(chosen) < target and heap:
        _, i = heapq.heappop(heap)
        fresh = score(i)
        if heap and fresh < -heap[0][0] - 1e-12:
            heapq.heappush(heap, (-fresh, i))
            continue
        s, di, pc = items[i]
        covered.update(di)
        phone_count.update(pc)
        done_q[s[-1]] += 1
        chosen.append(s)

    for end in "?!":
        run([i for i, it in enumerate(items) if it[0][-1] == end], quotas[end])
    run([i for i, it in enumerate(items) if it[0][-1] == "."], args.n - len(chosen))
    print("diphones:", len(covered), flush=True)

    all_di = set().union(*(di for _, di, _ in items))
    prompts = [{"id": f"ka{i + 1:04d}", "text": s} for i, s in enumerate(chosen)]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "name": "ka_v1", "source": "Mozilla Common Voice 17 validated_sentences (CC0)",
        "diphones_covered": len(covered), "diphones_in_pool": len(all_di),
        "phone_counts": dict(phone_count.most_common()), "prompts": prompts,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    words = sum(len(x["text"].split()) for x in prompts)
    print(f"{len(prompts)} წინადადება, {words} სიტყვა (~{words / 150 * 60 / 60:.0f} წთ კითხვა), "
          f"diphone {len(covered)}/{len(all_di)}; ყველაზე იშვიათი ფონემები:",
          phone_count.most_common()[-6:])


if __name__ == "__main__":
    main()
