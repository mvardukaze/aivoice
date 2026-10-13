"""ყველა დინამიკის ავტომატური შემოწმება MMS-ით (ქართული): რამდენად გასაგებად ამბობს და ისმის თუ არა „ღ“.

„ოდესღაც“-ის ტიპის „ს+ღ“ ორაზროვანია (ნათიაც „ოდესხაც“-ად ისმის), ამიტომ აქ „ღ“ ხმოვნებთან/მჟღერებთანაა.
    training/.venv/bin/python tools/gh_scan.py
შედეგი: data/ghtest/scan.json — {speaker: {cer, gh_hit, gh_total, texts}}; ხმის შემქმნელი რიგს ამით ალაგებს.
"""

import json
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "ghtest" / "scan.json"
API = "http://127.0.0.1:8200/api/designer"
PHRASES = ["დღეს ზღვაზე ტალღები აღარ ჩანს.", "მიღებული წერილი ღამით წავიკითხე."]
NORM = str.maketrans("", "", ".,!?;:„“\"'—-")


def synth(name, text, tries=20):
    for i in range(tries):  # პანელის გადატვირთვისას API წამიერად მიუწვდომელია
        try:
            return _synth(name, text)
        except OSError:
            if i == tries - 1:
                raise
            time.sleep(15)


def _synth(name, text):
    req = urllib.request.Request(f"{API}/synth", method="POST", headers={"Content-Type": "application/json"},
                                 data=json.dumps({"weights": {name: 1}, "text": text}).encode())
    r = json.loads(urllib.request.urlopen(req, timeout=180).read())
    return ROOT / "data" / "cache" / "designer" / r["url"].rsplit("/", 1)[-1]


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def gh_hits(ref: str, hyp: str) -> tuple[int, int]:
    """„ღ“-იანი სიტყვები: რამდენი იპოვა ჰიპოთეზაში „ღ“-თი (ყველაზე მსგავს სიტყვაში)."""
    hw = hyp.split()
    hit = tot = 0
    for w in ref.split():
        if "ღ" not in w:
            continue
        tot += 1
        best = min(hw, key=lambda x: lev(x, w), default="")
        hit += int("ღ" in best and lev(best, w) <= max(2, len(w) // 3))
    return hit, tot


def main():
    import librosa
    import torch
    from transformers import AutoProcessor, Wav2Vec2ForCTC
    torch.set_num_threads(8)
    for i in range(20):
        try:
            speakers = json.loads(urllib.request.urlopen(f"{API}/speakers").read())["speakers"]
            break
        except OSError:
            time.sleep(15)
    done = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    proc = AutoProcessor.from_pretrained("facebook/mms-1b-all", target_lang="kat")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", target_lang="kat", ignore_mismatched_sizes=True).eval()
    t0 = time.time()
    for k, s in enumerate(speakers, 1):
        n = s["name"]
        if n in done:
            continue
        texts, err, chars, hit, tot = [], 0, 0, 0, 0
        for ph in PHRASES:
            a, _ = librosa.load(str(synth(n, ph)), sr=16000)
            with torch.no_grad():
                ids = model(**proc(a, sampling_rate=16000, return_tensors="pt")).logits.argmax(-1)
            hyp = proc.batch_decode(ids)[0].translate(NORM).strip()
            ref = ph.translate(NORM).strip()
            err += lev(ref.replace(" ", ""), hyp.replace(" ", ""))
            chars += len(ref.replace(" ", ""))
            h, t = gh_hits(ref, hyp)
            hit, tot = hit + h, tot + t
            texts.append(hyp)
        done[n] = {"cer": round(err / chars, 3), "gh_hit": hit, "gh_total": tot, "texts": texts, "f0": s.get("f0")}
        if k % 10 == 0 or k == len(speakers):
            OUT.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"{k}/{len(speakers)} {time.time() - t0:.0f}s", flush=True)
    OUT.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
