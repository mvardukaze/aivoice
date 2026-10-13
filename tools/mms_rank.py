"""ყველა დინამიკის (და შენახული ნარევის) რეიტინგი MMS-ით (ქართული): 5 წინადადება სხვადასხვა რთული ადგილით.
MMS „ღ“-ს მსაჯულად არ გამოდგა, მაგრამ საერთო სისუფთავეს (შიშინი, კანკალი, ბუნდოვანება, დაკარგული ბგერები)
მომხმარებლის ყურთან კარგად ასახავს (2026-10-12). ტექსტი გახმოვანებამდე ka_text()-ით მზადდება („ოდეს-ღაც“).

    ./aivoice rank [--kind all|speaker|mix] [--fresh]   (ან პანელიდან: ხმის შემქმნელი → რეიტინგი)
შედეგი: data/mms_rank.json — {key: {kind, name, f0, cer, per: [cer…], texts: […], epoch}}; შეჩერების შემდეგ აგრძელებს.
პროგრესი: data/mms_rank.progress.json (პანელი აჩვენებს). --fresh — არჩეული ტიპის ძველი შედეგები იშლება.
"""

import json
import time
import urllib.request
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.config import api_url, cfg, path as cpath  # noqa: E402,F401
OUT = ROOT / "data" / "mms_rank.json"
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


PROGRESS = ROOT / "data" / "mms_rank.progress.json"


def save(done):
    tmp = OUT.with_suffix(".tmp")
    tmp.write_text(json.dumps(done, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(OUT)


def progress(**kw):
    PROGRESS.write_text(json.dumps({**kw, "updated": time.strftime("%Y-%m-%dT%H:%M:%S")}, ensure_ascii=False), encoding="utf-8")


def main():
    import argparse
    ap = argparse.ArgumentParser(prog="./aivoice rank")
    ap.add_argument("--kind", choices=("all", "speaker", "mix"), default="all")
    ap.add_argument("--fresh", action="store_true", help="არჩეული ტიპის ძველი შედეგები თავიდან")
    args = ap.parse_args()
    items = []
    if args.kind in ("all", "speaker"):
        items += [("speaker", s["name"], {s["name"]: 1}, s.get("f0")) for s in get(f"{API}/speakers")["speakers"]]
    designs = get(f"{API}/designs")
    if args.kind in ("all", "mix"):
        items += [("mix", d["name"], d["weights"], d.get("f0_estimate")) for d in designs]
    epoch = (get(f"{API}/status").get("meta") or {}).get("epoch")
    done = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    names = {d["name"] for d in designs}
    done = {k: v for k, v in done.items() if v["kind"] != "mix" or v["name"] in names}  # წაშლილი ნარევები
    if args.fresh:
        done = {k: v for k, v in done.items() if args.kind != "all" and v["kind"] != args.kind}
    todo = [it for it in items if f"{it[0]}:{it[1]}" not in done]
    save(done)
    progress(done=0, total=len(todo), kind=args.kind, started=time.strftime("%Y-%m-%dT%H:%M:%S"), stage="load")
    if not todo:
        progress(done=0, total=0, kind=args.kind, stage="done")
        print("ყველაფერი უკვე შეფასებულია")
        return
    import librosa
    import torch
    from transformers import AutoProcessor, Wav2Vec2ForCTC
    from transformers.utils import logging as tlog
    tlog.set_verbosity_error()
    torch.set_num_threads(max(1, min(8, int(cfg()["training"]["local_threads"]))))
    print(f"MMS-ის ჩატვირთვა… შესაფასებელია {len(todo)}", flush=True)
    proc = AutoProcessor.from_pretrained("facebook/mms-1b-all", target_lang="kat")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", target_lang="kat", ignore_mismatched_sizes=True).eval()
    t0 = time.time()
    for k, (kind, name, weights, f0) in enumerate(todo, 1):
        key = f"{kind}:{name}"
        progress(done=k - 1, total=len(todo), kind=args.kind, current=name, stage="run",
                 eta_min=round((time.time() - t0) / (k - 1) * (len(todo) - k + 1) / 60) if k > 1 else None)
        per, texts = [], []
        for s in SENTENCES:
            a, _ = librosa.load(str(synth(weights, s)), sr=16000)
            with torch.no_grad():
                hyp = proc.batch_decode(model(**proc(a, sampling_rate=16000, return_tensors="pt")).logits.argmax(-1))[0]
            ref = s.translate(STRIP)
            per.append(round(lev(ref, hyp.translate(STRIP)) / len(ref), 3))
            texts.append(hyp)
        done[key] = {"kind": kind, "name": name, "f0": f0, "cer": round(sum(per) / len(per), 3), "per": per, "texts": texts,
                     "epoch": epoch}
        save(done)
        print(f"{k}/{len(todo)} {name} {done[key]['cer']:.0%} ({time.time() - t0:.0f}s)", flush=True)
    progress(done=len(todo), total=len(todo), kind=args.kind, stage="done")


if __name__ == "__main__":
    main()
