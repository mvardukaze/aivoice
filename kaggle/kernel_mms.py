# aivoice — Common Voice-ის ჩანაწერების ტექსტთან დამთხვევა (MMS-1b-all, ქართული) Kaggle-ის 2×T4-ზე.
# შემავალი: dataset aivoice-cv27-clips (list.jsonl + clips.tar → *.mp3). გამომავალი: /kaggle/working/mms.jsonl
# იგივე ფორმატი, რასაც tools/clip_quality.py (mms ეტაპი) წერს: {"id", "cer", "heard"}.
import glob
import json
import os
import subprocess
import sys
import time

STRIP = str.maketrans("", "", ".,!?;:„“\"'—–-()… ")
OUT = "/kaggle/working"


def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def worker(part, nparts):
    import librosa
    import numpy as np
    import torch
    from transformers import AutoProcessor, Wav2Vec2ForCTC
    from transformers.utils import logging
    logging.set_verbosity_error()
    lst = next(iter(glob.glob("/kaggle/input/**/list.jsonl", recursive=True)))
    files = {os.path.basename(p): p for p in glob.glob("/kaggle/input/**/*.mp3", recursive=True)}
    if not files:  # Kaggle-მა tar არ გახსნა — ვხსნით თავად (მთავარი პროცესი ამას უკვე აკეთებს)
        files = {os.path.basename(p): p for p in glob.glob("/kaggle/temp/clips/**/*.mp3", recursive=True)}
    rows = [json.loads(l) for l in open(lst, encoding="utf-8")]
    rows = sorted(rows, key=lambda r: r.get("dur", 0))[part::nparts]  # სიგრძით — padding ნაკლებია
    out_f = f"{OUT}/mms.part{part}.jsonl"
    done = set()
    if os.path.exists(out_f):
        done = {json.loads(l)["id"] for l in open(out_f, encoding="utf-8") if l.strip()}
    rows = [r for r in rows if r["id"] not in done and r["id"] in files]
    proc = AutoProcessor.from_pretrained("facebook/mms-1b-all", target_lang="kat")
    model = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", target_lang="kat", ignore_mismatched_sizes=True)
    model = model.eval().half().cuda()
    print(f"[{part}] {len(rows)} ჩანაწერი, {len(files)} mp3", flush=True)
    t0, B = time.time(), 16
    with open(out_f, "a", encoding="utf-8") as fout:
        for i in range(0, len(rows), B):
            batch = rows[i:i + B]
            audio = []
            for r in batch:
                try:
                    y, _ = librosa.load(files[r["id"]], sr=16000)
                    a = max(0, int((r.get("lead", 0) - 0.1) * 16000))
                    b = len(y) - max(0, int((r.get("trail", 0) - 0.1) * 16000))
                    audio.append(y[a:b] if b - a > 1600 else y)
                except Exception:  # noqa: BLE001
                    audio.append(np.zeros(16000, dtype=np.float32))
            f = proc(audio, sampling_rate=16000, return_tensors="pt", padding=True)
            with torch.no_grad():
                logits = model(f.input_values.half().cuda(),
                               attention_mask=f.attention_mask.cuda() if "attention_mask" in f else None).logits
            for r, hyp in zip(batch, proc.batch_decode(logits.argmax(-1).cpu())):
                ref = r["text"].translate(STRIP)
                fout.write(json.dumps({"id": r["id"], "cer": round(lev(ref, hyp.translate(STRIP)) / max(1, len(ref)), 3),
                                       "heard": hyp}, ensure_ascii=False) + "\n")
            if (i // B) % 200 == 0:
                fout.flush()
                print(f"[{part}] {i + len(batch)}/{len(rows)} · {(i + len(batch)) / (time.time() - t0):.1f}/წმ", flush=True)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        worker(int(sys.argv[1]), int(sys.argv[2]))
        sys.exit(0)
    import torch
    if not glob.glob("/kaggle/input/**/*.mp3", recursive=True):
        for t in glob.glob("/kaggle/input/**/*.tar", recursive=True):
            os.makedirs("/kaggle/temp/clips", exist_ok=True)
            subprocess.run(["tar", "-xf", t, "-C", "/kaggle/temp/clips"], check=True)
    n = max(1, torch.cuda.device_count())
    procs = [subprocess.Popen([sys.executable, __file__, str(i), str(n)], env={**os.environ, "CUDA_VISIBLE_DEVICES": str(i)})
             for i in range(n)]
    codes = [p.wait() for p in procs]
    with open(f"{OUT}/mms.jsonl", "w", encoding="utf-8") as fo:
        for i in range(n):
            p = f"{OUT}/mms.part{i}.jsonl"
            if os.path.exists(p):
                fo.write(open(p, encoding="utf-8").read())
    json.dump({"codes": codes, "lines": sum(1 for _ in open(f"{OUT}/mms.jsonl", encoding="utf-8"))},
              open(f"{OUT}/status.json", "w"))
    print("done", codes, flush=True)
