"""„ღ“-ს ტესტი: 20 დინამიკი ამბობს „ოდესღაც, ძალიან დიდი ხნის წინ“; ორი ავტომატური „მსმენელი“ აფასებს,
ისმის თუ არა „ღ“. შედეგს მომხმარებელი ყურით ადარებს (ბრმა ტესტი, ხმის შემქმნელში).

    training/.venv/bin/python tools/gh_test.py

მოდელები (ერთხელ ჩამოიტვირთება, ~5 GB, ~/.cache/huggingface):
  - facebook/wav2vec2-xlsr-53-espeak-cv-ft — ბგერებს (espeak IPA) წერს; „ღ“ = ɣ
  - facebook/mms-1b-all + ქართული ადაპტერი (kat) — ქართულ ტექსტს წერს; ვამოწმებთ „ღ“-ს სიტყვაში „ოდესღაც“
შედეგი: data/ghtest/{clips/*.wav, clips.json (ბრმა — სახელების გარეშე), results.json}
"""

import json
import random
import shutil
import time
import urllib.request
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "ghtest"
TEXT = "ოდესღაც, ძალიან დიდი ხნის წინ"
API = "http://127.0.0.1:8200/api/designer"
# მომხმარებელმა ყურით: „ღ“ კარგად ისმის (2026-10-09)
USER_GOOD = ["cv_6023d1c991", "cv_3d7e36c318", "cv_b2fae02602", "cv_cf7c24d61c", "cv_75ef8f1d11", "cv_34be44c191"]
GH_LIKE = set("ɣʁɰ")  # ამომცნობი „ღ“-ს შეიძლება ახლო ბგერით ჩაწეროს — ცალკე ვაჩვენებთ ზუსტ ɣ-ს


def synth(name: str) -> Path:
    req = urllib.request.Request(f"{API}/synth", method="POST", headers={"Content-Type": "application/json"},
                                 data=json.dumps({"weights": {name: 1}, "text": TEXT}).encode())
    r = json.loads(urllib.request.urlopen(req, timeout=120).read())
    return ROOT / "data" / "cache" / "designer" / r["url"].rsplit("/", 1)[-1]


def pick_speakers() -> list[str]:
    job = json.loads((ROOT / "data" / "kaggle_jobs" / "base_ka.json").read_text())
    sp = json.loads((Path(job["dataset_dir"]) / "speakers.json").read_text())
    chosen = USER_GOOD + [s for s in job["preview_speakers"] if s not in USER_GOOD]
    rest = sorted([n for n, v in sp.items() if n not in chosen and (v.get("minutes") or 0) >= 10 and v.get("f0_median")],
                  key=lambda n: sp[n]["f0_median"])
    # 10 დანარჩენი — თანაბრად სიმაღლის მთელ დიაპაზონზე
    step = len(rest) / 10
    chosen += [rest[int(i * step + step / 2)] for i in range(10)]
    return chosen[:20], sp


def load16(path: Path):
    import librosa
    a, _ = librosa.load(str(path), sr=16000)
    return a.astype(np.float32)


def main():
    import torch
    from transformers import AutoProcessor, Wav2Vec2ForCTC
    torch.set_num_threads(8)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "clips").mkdir(exist_ok=True)
    names, sp = pick_speakers()
    order = list(range(len(names)))
    random.Random(7).shuffle(order)  # ბრმა: ნომრები სახელებთან არ არის დაკავშირებული
    rows = []
    for k, i in enumerate(order, 1):
        n = names[i]
        dst = OUT / "clips" / f"{k:02d}.wav"
        shutil.copy2(synth(n), dst)
        rows.append({"id": k, "speaker": n, "f0": sp[n].get("f0_median"), "minutes": sp[n].get("minutes"),
                     "user_good_before": n in USER_GOOD})
    (OUT / "clips.json").write_text(json.dumps([{"id": r["id"]} for r in rows]))
    print("synth OK", len(rows), flush=True)

    audio = [load16(OUT / "clips" / f"{r['id']:02d}.wav") for r in rows]

    t = time.time()
    from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2PhonemeCTCTokenizer
    W = "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
    fe = Wav2Vec2FeatureExtractor.from_pretrained(W)
    tok = Wav2Vec2PhonemeCTCTokenizer.from_pretrained(W, do_phonemize=False)  # მხოლოდ დეკოდირება — phonemizer არ სჭირდება
    model = Wav2Vec2ForCTC.from_pretrained(W).eval()
    for r, a in zip(rows, audio):
        with torch.no_grad():
            ids = model(fe(a, sampling_rate=16000, return_tensors="pt").input_values).logits.argmax(-1)
        ph = tok.batch_decode(ids)[0]
        r["w2v"] = ph
        r["w2v_gh"] = "ɣ" in ph
        r["w2v_gh_like"] = any(c in GH_LIKE for c in ph)
    del model
    print("wav2vec2 OK %.0fs" % (time.time() - t), flush=True)

    t = time.time()
    proc = AutoProcessor.from_pretrained("facebook/mms-1b-all", target_lang="kat")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", target_lang="kat", ignore_mismatched_sizes=True).eval()
    for r, a in zip(rows, audio):
        with torch.no_grad():
            ids = model(**proc(a, sampling_rate=16000, return_tensors="pt")).logits.argmax(-1)
        txt = proc.batch_decode(ids)[0]
        r["mms"] = txt
        first = txt.split()[0] if txt.split() else ""
        r["mms_first"] = first
        r["mms_gh"] = "ღ" in first or "ღ" in txt[:12]
    print("mms OK %.0fs" % (time.time() - t), flush=True)

    (OUT / "results.json").write_text(json.dumps({"text": TEXT, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                                  "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")
    for r in rows:
        print(r["id"], r["speaker"], r["f0"], "w2v:", r["w2v_gh"], r["w2v"][:40], "| mms:", r["mms_gh"], r["mms"][:40])


if __name__ == "__main__":
    main()
