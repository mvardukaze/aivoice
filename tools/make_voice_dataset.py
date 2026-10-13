"""ცალკე ხმის dataset შენახული ნარევიდან (დისტილაცია): საბაზო მოდელი + ნარევის ვექტორი → ჩანაწერები.

    training/.venv/bin/python tools/make_voice_dataset.py --design "ქალი 1" --slot f_mix_01

თითო წინადადება `--variants`-ჯერ (სინთეზი შემთხვევითია) → MMS (ქართული) ჩაწერს, რაც გაიგონა → რჩება ყველაზე
ნაკლები შეცდომის ვარიანტი; თუ საუკეთესოსაც > --max-cer შეცდომა აქვს, წინადადება ამოვარდება.
ტექსტი ka_text()-ით მზადდება („ოდეს-ღაც“) — metadata-შიც იგივე, რომ წვრთნა და გახმოვანება ერთნაირი იყოს.
შედეგი: <paths.datasets>/<slot>/{wav/*.wav, metadata.csv, stats.jsonl, design.json}; შეჩერების შემდეგ აგრძელებს.
"""

import argparse
import json
import re
import sys
import time
import wave
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.config import api_url, cfg, expand, path as cpath  # noqa: E402,F401
sys.path.insert(0, str(ROOT))
from app.textnorm import ka_text  # noqa: E402

DES = cpath("designer")
SR = 22050
STRIP = str.maketrans("", "", ".,!?;:„“\"'—–-() ")


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True, help="შენახული ნარევის სახელი (data/designs)")
    ap.add_argument("--slot", required=True)
    ap.add_argument("--texts", default=str(ROOT / "data" / "prompts" / "voice_texts_v1.json"))
    ap.add_argument("--variants", type=int, default=3)
    ap.add_argument("--max-cer", type=float, default=0.30)
    a = ap.parse_args()

    import librosa
    import torch
    from piper import PiperVoice
    from transformers import AutoProcessor, Wav2Vec2ForCTC
    torch.set_num_threads(8)

    design = next(d for p in (ROOT / "data" / "designs").glob("*.json")
                  if (d := json.loads(p.read_text(encoding="utf-8")))["name"] == a.design)
    out = cpath("datasets") / a.slot
    (out / "wav").mkdir(parents=True, exist_ok=True)
    meta_des = json.loads((DES / "meta.json").read_text())
    (out / "design.json").write_text(json.dumps({**design, "base_model": meta_des}, ensure_ascii=False, indent=1),
                                     encoding="utf-8")

    voice = PiperVoice.load(DES / "designer.onnx", DES / "config.json")
    emb = np.load(DES / "embeddings.npy")
    smap = json.loads((DES / "config.json").read_text())["speaker_id_map"]
    tot = sum(design["weights"].values())
    g = sum((w / tot) * emb[smap[n]] for n, w in design["weights"].items()).astype(np.float32)[None, :]
    scales = np.array([design.get("noise_scale", 0.667), design.get("length_scale", 1.0),
                       design.get("noise_w_scale", 0.8)], dtype=np.float32)

    proc = AutoProcessor.from_pretrained("facebook/mms-1b-all", target_lang="kat")
    asr = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", target_lang="kat", ignore_mismatched_sizes=True).eval()

    texts = json.loads(Path(a.texts).read_text(encoding="utf-8"))["sentences"]
    stats_p = out / "stats.jsonl"
    done = {json.loads(l)["i"] for l in stats_p.read_text(encoding="utf-8").splitlines()} if stats_p.exists() else set()
    t0, kept = time.time(), 0
    with stats_p.open("a", encoding="utf-8") as stats:
        for i, raw in enumerate(texts):
            if i in done:
                continue
            if len(re.findall("[ა-ჰ]", raw)) < 0.8 * len(raw.replace(" ", "")):  # დაზიანებული სტრიქონი („?????“)
                stats.write(json.dumps({"i": i, "text": raw, "cer": 1.0, "ok": False, "heard": "", "skip": "broken"},
                                       ensure_ascii=False) + "\n")
                continue
            text = ka_text(raw)
            ref = raw.translate(STRIP)
            best = None
            for v in range(a.variants):
                chunks = []
                for ph in voice.phonemize(text):
                    if not ph:
                        continue
                    ids = np.array([voice.phonemes_to_ids(ph)], dtype=np.int64)
                    chunks.append(voice.session.run(None, {"input": ids, "input_lengths": np.array([ids.shape[1]], dtype=np.int64),
                                                           "scales": scales, "g": g})[0].squeeze())
                    chunks.append(np.zeros(int(SR * 0.2), dtype=np.float32))
                audio = np.concatenate(chunks[:-1]) if chunks else np.zeros(1, np.float32)
                if len(audio) < SR * 0.3:  # ბგერებად ვერ გადაიქცა (ცარიელი/უცნაური ტექსტი) — გამოტოვება
                    best = best or (1.0, audio, "")
                    continue
                with torch.no_grad():
                    a16 = librosa.resample(audio, orig_sr=SR, target_sr=16000)
                    hyp = proc.batch_decode(asr(**proc(a16, sampling_rate=16000, return_tensors="pt")).logits.argmax(-1))[0]
                cer = lev(ref, hyp.translate(STRIP)) / max(1, len(ref))
                if best is None or cer < best[0]:
                    best = (cer, audio, hyp)
            cer, audio, hyp = best
            ok = cer <= a.max_cer
            if ok:
                audio = audio / (np.max(np.abs(audio)) + 1e-8) * 0.9
                with wave.open(str(out / "wav" / f"{i:05d}.wav"), "wb") as w:
                    w.setnchannels(1)
                    w.setsampwidth(2)
                    w.setframerate(SR)
                    w.writeframes((audio * 32767).astype(np.int16).tobytes())
                kept += 1
            stats.write(json.dumps({"i": i, "text": text, "cer": round(cer, 3), "ok": ok, "heard": hyp}, ensure_ascii=False) + "\n")
            stats.flush()
            if (i + 1) % 50 == 0:
                print(f"{i + 1}/{len(texts)} kept {kept} {time.time() - t0:.0f}s", flush=True)

    rows = [json.loads(l) for l in stats_p.read_text(encoding="utf-8").splitlines()]
    with (out / "metadata.csv").open("w", encoding="utf-8") as m:
        for r in sorted(rows, key=lambda r: r["i"]):
            if r["ok"]:
                m.write(f"{r['i']:05d}.wav|{r['text']}\n")
    print("DONE", sum(r["ok"] for r in rows), "/", len(rows))


if __name__ == "__main__":
    main()
