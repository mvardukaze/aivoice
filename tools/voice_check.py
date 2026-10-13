"""გაწვრთნილი ხმის შემოწმება: სატესტო ფრაზები ხმით და მისი ნარევით (თუ არის) + MMS-ის შეფასება.

    ./aivoice check <voice_id> [--no-mms]
შედეგი: <runs>/<voice_id>/check/report.json + wav-ები; პანელი („ხმები“ → „შემოწმება“) ამას აჩვენებს.
საჭიროა ჩართული პანელი (გახმოვანება მისი API-თ ხდება). MMS — წვრთნის გარემოში (პირველად ~4 GB ჩამოიტვირთება).
CER — სიმბოლოების შეცდომის წილი: რაც ნაკლებია, მით სუფთად ისმის (MMS „ღ“-ს ვერ აფასებს — იხ. README).
"""

import argparse
import datetime as dt
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.config import api_url, cfg, path as cpath  # noqa: E402

PHRASES = [
    "გამარჯობა! დღეს მშვენიერი ამინდია, არა?",
    "ოდესღაც, ძალიან დიდი ხნის წინ, ერთ პატარა სოფელში ცხოვრობდა მოხუცი მეთევზე.",
    "შენ აქ რას აკეთებ? ხომ გითხარი, არ მოსულიყავი!",
    "ზღვის ღრმა ტალღებმა ნავი ნაპირისკენ გააქანა.",
    "მწვრთნელმა ბავშვებს წყალში ცურვა ასწავლა.",
    "ბოდიში... არ მინდოდა, რომ ასე მომხდარიყო.",
    "ჩვენ მოვიგეთ! წარმოგიდგენია? მოვიგეთ!",
    "თბილისში შემოდგომა განსაკუთრებით ლამაზია, როცა ხეები ყვითლდება.",
]
STRIP = str.maketrans("", "", ".,!?;:„“\"'—-… ")


def call(url, data=None, timeout=180):
    for _ in range(12):  # პანელის გადატვირთვისას API წამიერად მიუწვდომელია
        try:
            req = urllib.request.Request(url, method="POST" if data is not None else "GET",
                                         headers={"Content-Type": "application/json"},
                                         data=json.dumps(data).encode() if data is not None else None)
            return urllib.request.urlopen(req, timeout=timeout).read()
        except urllib.error.HTTPError as e:
            raise SystemExit(f"პანელის შეცდომა {e.code}: {e.read().decode(errors='replace')[:200]}")
        except OSError:
            time.sleep(10)
    raise SystemExit("პანელი არ პასუხობს — ჩართეთ: ./aivoice serve")


def save(url_path, dst: Path):
    dst.write_bytes(call(api_url(url_path)))


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def find_design(vid: str):
    job = ROOT / "data" / "kaggle_jobs" / f"{vid}.json"
    name = json.loads(job.read_text(encoding="utf-8")).get("design") if job.exists() else None
    if not name:
        return None
    for p in (ROOT / "data" / "designs").glob("*.json"):
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("name") == name:
            return d
    return None


def verdict(v, m):
    if v is None:
        return None
    if v <= 0.08 or (m is not None and v <= m + 0.02):
        return "good"
    if v <= 0.15 or (m is not None and v <= m + 0.06):
        return "fair"
    return "bad"


def main():
    ap = argparse.ArgumentParser(prog="./aivoice check")
    ap.add_argument("voice")
    ap.add_argument("--no-mms", action="store_true")
    a = ap.parse_args()

    out = cpath("runs") / a.voice / "check"
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    voices = json.loads(call(api_url("/api/voices")))
    v = next((x for x in voices if x["id"] == a.voice), None)
    if not v or not v.get("available"):
        raise SystemExit(f"ხმა {a.voice} ამ კომპიუტერზე არ არის")
    design = find_design(a.voice)
    rep = {"voice": a.voice, "name": v["name"], "design": design and design["name"], "params": v.get("defaults"),
           "started": dt.datetime.now().isoformat(timespec="seconds"), "mms": False, "rows": [], "stage": "synth"}
    wr = lambda: (out / "report.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")  # noqa: E731

    print(f"„{v['name']}“ — {len(PHRASES)} ფრაზა" + (f", ნარევი „{design['name']}“" if design else ", ნარევი ვერ მოიძებნა"), flush=True)
    for i, text in enumerate(PHRASES):
        r = json.loads(call(api_url("/api/synthesize"), {"voice": a.voice, "text": text}))
        save(r["url"], out / f"v{i}.wav")
        row = {"text": text, "voice": {"file": f"v{i}.wav", "seconds": r["seconds"]}}
        if design:
            m = json.loads(call(api_url("/api/designer/synth"), {
                "weights": design["weights"], "text": text, "length_scale": design.get("length_scale", 1.0),
                "noise_scale": design.get("noise_scale", 0.667), "noise_w_scale": design.get("noise_w_scale", 0.8)}))
            save(m["url"], out / f"m{i}.wav")
            row["mix"] = {"file": f"m{i}.wav"}
        rep["rows"].append(row)
        rep["progress"] = f"{i + 1}/{len(PHRASES)}"
        wr()
        print(f"  {i + 1}/{len(PHRASES)} გახმოვანდა", flush=True)

    if not a.no_mms:
        try:
            import librosa
            import torch
            from transformers import AutoProcessor, Wav2Vec2ForCTC
            from transformers.utils import logging as tlog
            tlog.set_verbosity_error()
        except ImportError:
            print("MMS გამოტოვებულია: საჭიროა წვრთნის გარემო (./install.sh --training)", flush=True)
        else:
            rep["stage"] = "mms"
            wr()
            print("MMS-ის ჩატვირთვა (პირველად ~4 GB ჩამოიტვირთება)…", flush=True)
            torch.set_num_threads(max(1, min(8, int(cfg()["training"]["local_threads"]))))
            proc = AutoProcessor.from_pretrained("facebook/mms-1b-all", target_lang="kat")
            model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", target_lang="kat", ignore_mismatched_sizes=True).eval()
            for row in rep["rows"]:
                ref = row["text"].translate(STRIP)
                for k in ("voice", "mix"):
                    if k not in row:
                        continue
                    audio, _ = librosa.load(str(out / row[k]["file"]), sr=16000)
                    with torch.no_grad():
                        hyp = proc.batch_decode(model(**proc(audio, sampling_rate=16000, return_tensors="pt")).logits.argmax(-1))[0]
                    row[k].update(cer=round(lev(ref, hyp.translate(STRIP)) / len(ref), 3), heard=hyp)
            rep["mms"] = True
            avg = lambda k: round(sum(r[k]["cer"] for r in rep["rows"] if k in r) / len(rep["rows"]), 3) if all(k in r for r in rep["rows"]) else None  # noqa: E731
            rep["summary"] = {"voice_cer": avg("voice"), "mix_cer": avg("mix")}
            rep["verdict"] = verdict(rep["summary"]["voice_cer"], rep["summary"]["mix_cer"])
            print(f"CER: ხმა {rep['summary']['voice_cer']} · ნარევი {rep['summary']['mix_cer']} → {rep['verdict']}", flush=True)
    rep["stage"] = "done"
    rep["finished"] = dt.datetime.now().isoformat(timespec="seconds")
    wr()
    print("✓ შემოწმება დასრულდა — „ხმები“ → „შემოწმება“", flush=True)


if __name__ == "__main__":
    main()
