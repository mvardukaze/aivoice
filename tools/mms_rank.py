"""ყველა დინამიკის (და შენახული ნარევის) რეიტინგი MMS-ით (ქართული): 5 წინადადება სხვადასხვა რთული ადგილით.
MMS „ღ“-ს მსაჯულად არ გამოდგა, მაგრამ საერთო სისუფთავეს (შიშინი, კანკალი, ბუნდოვანება, დაკარგული ბგერები)
მომხმარებლის ყურთან კარგად ასახავს (2026-10-12). ტექსტი გახმოვანებამდე ka_text()-ით მზადდება („ოდეს-ღაც“).

    training/.venv/bin/python tools/mms_rank.py
შედეგი: data/ghtest/rank.json — {key: {kind, name, f0, cer, per: [cer…], texts: […]}}; შეჩერების შემდეგ აგრძელებს.
"""

import json
import time
import urllib.request
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.config import api_url, cfg, expand, path as cpath  # noqa: E402,F401
OUT = ROOT / "data" / "ghtest" / "rank.json"
API = api_url("/api/designer")
SENTENCES = [
    "გამარჯობა! დღეს მშვენიერი ამინდია, არა?",
    "ოდესღაც, ძალიან დიდი ხნის წინ, ერთ პატარა სოფელში ცხოვრობდა მოხუცი მეთევზე.",
    "შენ აქ რას აკეთებ? ხომ გითხარი, არ მოსულიყავი!",
    "ზღვის ღრმა ტალღებმა ნავი ნაპირისკენ გააქანა.",
    "მწვრთნელმა ბავშვებს წყალში ცურვა ასწავლა.",
]
STRIP = str.maketrans("", "", ".,!?;:„“\"'—- ")


def get(url):
    for i in range(20):  # პანელის გადატვირთვისას API წამიერად მიუწვდომელია
        try:
            return json.loads(urllib.request.urlopen(url, timeout=60).read())
        except OSError:
            time.sleep(15)
    raise RuntimeError(url)


def synth(weights, text):
    for i in range(20):
        try:
            req = urllib.request.Request(f"{API}/synth", method="POST", headers={"Content-Type": "application/json"},
                                         data=json.dumps({"weights": weights, "text": text}).encode())
            url = json.loads(urllib.request.urlopen(req, timeout=180).read())["url"]
            return cpath("cache") / "designer" / url.rsplit("/", 1)[-1]
        except OSError:
            time.sleep(15)
    raise RuntimeError("synth")


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main():
    import librosa
    import torch
    from transformers import AutoProcessor, Wav2Vec2ForCTC
    torch.set_num_threads(8)
    items = [("speaker", s["name"], {s["name"]: 1}, s.get("f0")) for s in get(f"{API}/speakers")["speakers"]]
    items += [("mix", d["name"], d["weights"], d.get("f0_estimate")) for d in get(f"{API}/designs")]
    done = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    proc = AutoProcessor.from_pretrained("facebook/mms-1b-all", target_lang="kat")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", target_lang="kat", ignore_mismatched_sizes=True).eval()
    t0 = time.time()
    for k, (kind, name, weights, f0) in enumerate(items, 1):
        key = f"{kind}:{name}"
        if key in done:
            continue
        per, texts = [], []
        for s in SENTENCES:
            a, _ = librosa.load(str(synth(weights, s)), sr=16000)
            with torch.no_grad():
                hyp = proc.batch_decode(model(**proc(a, sampling_rate=16000, return_tensors="pt")).logits.argmax(-1))[0]
            ref = s.translate(STRIP)
            per.append(round(lev(ref, hyp.translate(STRIP)) / len(ref), 3))
            texts.append(hyp)
        done[key] = {"kind": kind, "name": name, "f0": f0, "cer": round(sum(per) / len(per), 3), "per": per, "texts": texts}
        if k % 10 == 0 or k == len(items):
            OUT.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")
            print(f"{k}/{len(items)} {time.time() - t0:.0f}s", flush=True)
    OUT.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
