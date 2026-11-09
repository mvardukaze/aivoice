# OmniVoice-ის სიჩქარის ტესტი Kaggle-ზე (ქართული, voice design). tools/kaggle_omni_bench.py ავსებს __CONFIG__-ს.
# შედეგი /kaggle/working-ში: results.json, bench.log, samples/*.wav
#
# CONFIG: sentences [{id,text}], budget_s (ტესტების ლიმიტი წამებში, ჩატვირთვის შემდეგ), quick (მხოლოდ რამდენიმე
#         კონფიგურაცია — მაგ. P100-ის შესამოწმებლად), pip (omnivoice-ის წყარო)

import json
import os
import subprocess
import sys
import time
from pathlib import Path

CFG = json.loads('''__CONFIG__''')
OUT = Path("/kaggle/working")
T0 = time.time()
RESULTS = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "configs": [], "errors": [], "env": {}}


def log(*a):
    msg = f"[{time.time() - T0:7.1f}s] " + " ".join(str(x) for x in a)
    print(msg, flush=True)
    with open(OUT / "bench.log", "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def save():
    (OUT / "results.json").write_text(json.dumps(RESULTS, ensure_ascii=False, indent=1), encoding="utf-8")


# ყველა GPU-სამუშაო ერთ მოდულშია: მთავარი პროცესი მას იმპორტს უკეთებს, 2 GPU-ზე კი თითო პროცესი თითო GPU-ზე.
WORKER = r'''
import json, sys, time, gc
import numpy as np
import torch
import soundfile as sf

_model = None

def load():
    global _model
    from omnivoice import OmniVoice
    t = time.time()
    _model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)
    return time.time() - t

def sr():
    return _model.sampling_rate

def gen(texts, instruct, num_step):
    out = _model.generate(text=texts, language="ka", instruct=[instruct] * len(texts), num_step=num_step)
    return [np.asarray(a, dtype=np.float32).reshape(-1) for a in out]

def run(sentences, instruct, num_step, bs, sample_ids=(), sample_prefix="", sample_dir=None):
    """sentences — სიგრძით დალაგებული (batch-ში padding-ის შესამცირებლად). აბრუნებს გაზომვას."""
    torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
    items, t = [], time.time()
    for i in range(0, len(sentences), bs):
        chunk = sentences[i:i + bs]
        tb = time.time()
        audios = gen([s["text"] for s in chunk], instruct, num_step)
        torch.cuda.synchronize()
        dt = time.time() - tb
        for s, a in zip(chunk, audios):
            d = len(a) / sr()
            fr = a[: len(a) // 480 * 480].reshape(-1, 480)  # 20 ms
            rms = np.sqrt((fr ** 2).mean(1) + 1e-12)
            items.append({"id": s["id"], "chars": len(s["text"]), "dur": round(d, 3),
                          "cps": round(len(s["text"]) / max(d, 1e-3), 2),
                          "silence_frac": round(float((rms < 0.01).mean()), 3),
                          "peak": round(float(np.abs(a).max()), 3), "batch_s": round(dt, 3)})
            if sample_dir and s["id"] in sample_ids:
                sf.write(f"{sample_dir}/{sample_prefix}_{s['id']}.wav", a, sr())
    wall = time.time() - t
    audio = sum(x["dur"] for x in items)
    return {"wall_s": round(wall, 2), "audio_s": round(audio, 2), "rtf": round(wall / audio, 4),
            "audio_h_per_gpu_h": round(audio / wall, 2),
            "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2), "items": items}

def free():
    global _model
    _model = None
    gc.collect(); torch.cuda.empty_cache()

if __name__ == "__main__":
    # worker: python omni_worker.py <job.json> <out.json> — GPU არჩეულია CUDA_VISIBLE_DEVICES-ით
    job = json.load(open(sys.argv[1]))
    from pathlib import Path
    ready, go = Path(job["ready"]), Path(job["go"])
    load()
    gen([job["sentences"][0]["text"]], job["instruct"], job["num_step"])  # warmup
    ready.write_text("1")
    while not go.exists():
        time.sleep(0.05)
    r = run(job["sentences"], job["instruct"], job["num_step"], job["bs"])
    r["gpu"] = torch.cuda.get_device_name(0)
    r["end_t"] = time.time()
    json.dump(r, open(sys.argv[2], "w"))
'''


def main():
    log("pip install omnivoice")
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", CFG["pip"], "soundfile"],
                       capture_output=True, text=True)
    log("pip rc", r.returncode, (r.stdout + r.stderr)[-2000:])
    RESULTS["env"]["pip_rc"] = r.returncode
    RESULTS["env"]["pip_s"] = round(time.time() - T0, 1)
    import torch
    n = torch.cuda.device_count()
    gpus = [torch.cuda.get_device_name(i) for i in range(n)]
    cap = [torch.cuda.get_device_capability(i) for i in range(n)]
    RESULTS["env"].update(gpus=gpus, capability=cap, torch=torch.__version__, cuda=torch.version.cuda,
                          arch_list=torch.cuda.get_arch_list())
    try:
        import transformers
        RESULTS["env"]["transformers"] = transformers.__version__
    except Exception as e:  # noqa: BLE001
        RESULTS["env"]["transformers"] = repr(e)
    log("GPUs", gpus, cap, torch.__version__, torch.cuda.get_arch_list())
    save()

    Path("/tmp/omni_worker.py").write_text(WORKER)
    sys.path.insert(0, "/tmp")
    import omni_worker as W

    try:
        RESULTS["env"]["load_s"] = round(W.load(), 1)
        RESULTS["env"]["sample_rate"] = W.sr()
        t = time.time()
        W.gen(["გამარჯობა, ეს სატესტო წინადადებაა."], CFG["voices"]["female"], 32)
        RESULTS["env"]["warmup_s"] = round(time.time() - t, 1)
    except Exception as e:  # noqa: BLE001
        import traceback
        RESULTS["errors"].append({"stage": "load/warmup", "error": repr(e), "tb": traceback.format_exc()[-3000:]})
        log("FATAL", repr(e))
        save()
        return
    log("model loaded", RESULTS["env"])
    save()

    sents = sorted(CFG["sentences"], key=lambda s: len(s["text"]))
    samples = Path(OUT / "samples")
    samples.mkdir(exist_ok=True)
    short_id, long_id = sents[2]["id"], sents[-1]["id"]
    bench_t0 = time.time()

    def over():
        return time.time() - bench_t0 > CFG["budget_s"]

    plan = []
    for voice in ("female", "male"):
        for step in (32, 16):
            # ნიმუშები: ქალი/კაცი × 32/16; 32-ზე მოკლეც და გრძელიც (სულ 6)
            ids = (short_id, long_id) if step == 32 else (long_id,)
            plan.append(dict(voice=voice, num_step=step, bs=1, sample_ids=ids))
    for step in (32, 16):
        for bs in (4, 8):
            plan.append(dict(voice="female", num_step=step, bs=bs, sample_ids=()))
    plan.append(dict(voice="female", num_step=16, bs=16, sample_ids=()))
    if CFG.get("quick"):
        plan = [p for p in plan if (p["bs"], p["voice"]) in ((1, "female"), (8, "female"))]

    for p in plan:
        if over():
            RESULTS["errors"].append({"stage": "skip", "error": f"budget exceeded, skipped {p}"})
            continue
        name = f"1gpu_{p['voice']}_s{p['num_step']}_bs{p['bs']}"
        log("run", name)
        try:
            r = W.run(sents, CFG["voices"][p["voice"]], p["num_step"], p["bs"], p["sample_ids"],
                      f"{p['voice']}_s{p['num_step']}", str(samples))
            r.update(name=name, gpus=1, gpu=gpus[0], **{k: p[k] for k in ("voice", "num_step", "bs")})
            RESULTS["configs"].append(r)
            log(name, {k: r[k] for k in ("wall_s", "audio_s", "rtf", "audio_h_per_gpu_h", "peak_vram_gb")})
        except Exception as e:  # noqa: BLE001
            import traceback
            RESULTS["errors"].append({"stage": name, "error": repr(e), "tb": traceback.format_exc()[-2000:]})
            log("ERROR", name, repr(e))
            W.torch.cuda.empty_cache()
        save()

    if n >= 2 and not CFG.get("quick"):
        W.free()
        for step in (32, 16):
            if over():
                RESULTS["errors"].append({"stage": "skip", "error": f"budget exceeded, skipped 2gpu s{step}"})
                continue
            name = f"2gpu_female_s{step}_bs8"
            log("run", name)
            try:
                RESULTS["configs"].append(dual(sents, step, 8, n, gpus, name))
                log(name, {k: RESULTS["configs"][-1][k] for k in ("wall_s", "audio_s", "rtf", "audio_h_per_gpu_h")})
            except Exception as e:  # noqa: BLE001
                RESULTS["errors"].append({"stage": name, "error": repr(e)})
                log("ERROR", name, repr(e))
            save()
    RESULTS["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    RESULTS["total_s"] = round(time.time() - T0, 1)
    save()
    log("done")


def dual(sents, step, bs, n, gpus, name):
    """ორი პროცესი, თითო GPU-ზე, თითოეულს სრული 40 წინადადება; კედლის დრო — ორივე GPU-ს საერთო."""
    go = Path("/tmp/omni_go")
    go.unlink(missing_ok=True)
    procs, outs = [], []
    for g in range(2):
        job = {"sentences": sents, "instruct": CFG["voices"]["female"], "num_step": step, "bs": bs,
               "ready": f"/tmp/omni_ready{g}", "go": str(go)}
        Path(job["ready"]).unlink(missing_ok=True)
        jp, op = f"/tmp/omni_job{g}.json", f"/tmp/omni_out{g}_{step}.json"
        Path(jp).write_text(json.dumps(job, ensure_ascii=False))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g))
        procs.append(subprocess.Popen([sys.executable, "/tmp/omni_worker.py", jp, op], env=env))
        outs.append((Path(job["ready"]), Path(op)))
    t_wait = time.time()
    while not all(r.exists() for r, _ in outs):
        if any(p.poll() is not None for p in procs) or time.time() - t_wait > 900:
            for p in procs:
                p.kill()
            raise RuntimeError("worker failed before ready")
        time.sleep(0.5)
    t = time.time()
    go.write_text("1")
    for p in procs:
        p.wait(timeout=1800)
    res = [json.loads(o.read_text()) for _, o in outs]
    wall = max(r["end_t"] for r in res) - t
    audio = sum(r["audio_s"] for r in res)
    return {"name": name, "gpus": 2, "gpu": " + ".join(r["gpu"] for r in res), "voice": "female",
            "num_step": step, "bs": bs, "wall_s": round(wall, 2), "audio_s": round(audio, 2),
            "rtf": round(wall / audio, 4), "audio_h_per_gpu_h": round(audio / wall, 2),
            "note": "audio_h_per_gpu_h here = per Kaggle session-hour (both GPUs together)",
            "per_gpu": [{k: r[k] for k in ("wall_s", "audio_s", "rtf", "peak_vram_gb")} for r in res]}


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # noqa: BLE001
        import traceback
        RESULTS["errors"].append({"stage": "main", "error": repr(e), "tb": traceback.format_exc()[-3000:]})
        save()
        raise
