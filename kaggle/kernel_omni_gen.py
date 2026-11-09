# OmniVoice-ით სინთეზური ქართული ჩანაწერები საბაზო Piper მოდელისთვის (base_ka_v4_omni_nc) — Kaggle-ის 2×T4-ზე.
# tools/omni_dataset.py ავსებს __CONFIG__-ს. ⚠ OmniVoice-ის წონები CC-BY-NC — შედეგი მხოლოდ არაკომერციულია.
#
# mode "refs": თითო ხმაზე რამდენიმე კანდიდატი (instruct + seed, voice design) → MMS, ხანგრძლივობა, F0,
#              clone-prompt-ით სატესტო წინადადებები (სიჩქარე, CER). შედეგი: refs/*.wav, refs.json
# mode "gen":  თითო ხმაზე ფიქსირებული reference (clone prompt) → shard-ის წინადადებები; ყოველი ჩანაწერი
#              MMS-ით მოწმდება, ცუდი — ერთხელ თავიდან (სხვა seed), რჩება უკეთესი. 22050 Hz FLAC, სიჩუმე მოჭრილი,
#              ხმამაღლობა გათანაბრებული. ყოველ ~250 ჩანაწერზე tars/<voice>.c<N>.tar (flac + rows.jsonl) —
#              გაჩერებისას დაკარგვა მცირეა; შემდეგი გაშვება უკვე გაკეთებულს (CONFIG done) გამოტოვებს.
#              დროის ლიმიტი: budget_s (ჩატვირთვიდან), მერე ყველაფერი იფუთება და kernel სრულდება.
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

CFG = json.loads('''__CONFIG__''')
OUT = Path(os.environ.get("OMNI_OUT", "/kaggle/working"))  # OMNI_OUT/OMNI_TMP — ლოკალური შემოწმებისთვის
TMP = Path(os.environ.get("OMNI_TMP", "/kaggle/temp"))
T0 = time.time()

WORKER = r'''
import io, json, os, re, sys, tarfile, time
from pathlib import Path
import numpy as np
import torch, torchaudio
import soundfile as sf

job = json.load(open(sys.argv[1]))
CFG = job["cfg"]
OUT, TMP = Path(os.environ.get("OMNI_OUT", "/kaggle/working")), Path(os.environ.get("OMNI_TMP", "/kaggle/temp"))
GPU = job["gpu"]
STRIP = str.maketrans("", "", ".,!?;:„“\"'—–-()… ")
LOG = open(OUT / f"worker{GPU}.log", "a", encoding="utf-8")
T_START = time.time()

def log(*a):
    m = f"[{time.strftime('%H:%M:%S')} gpu{GPU}] " + " ".join(str(x) for x in a)
    print(m, flush=True); LOG.write(m + "\n"); LOG.flush()

from omnivoice import OmniVoice
model = OmniVoice.from_pretrained("k2-fsa/OmniVoice", device_map="cuda:0", dtype=torch.float16)
SR = model.sampling_rate
from transformers import AutoProcessor, Wav2Vec2ForCTC
from transformers.utils import logging as tlog
tlog.set_verbosity_error()
proc = AutoProcessor.from_pretrained("facebook/mms-1b-all", target_lang="kat")
mms = Wav2Vec2ForCTC.from_pretrained("facebook/mms-1b-all", target_lang="kat", ignore_mismatched_sizes=True).eval().half().cuda()
log("models loaded", round(time.time() - T_START, 1), "s")

def lev(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]

def resample(y, sr_from, sr_to):
    return torchaudio.functional.resample(torch.from_numpy(np.ascontiguousarray(y, dtype=np.float32)), sr_from, sr_to).numpy()

def heard(y24):
    y = resample(y24, SR, 16000)
    f = proc(y, sampling_rate=16000, return_tensors="pt")
    with torch.no_grad():
        lg = mms(f.input_values.half().cuda()).logits
    return proc.batch_decode(lg.argmax(-1).cpu())[0]

def cer(text, hyp):
    ref = text.translate(STRIP)
    return round(lev(ref, hyp.translate(STRIP)) / max(1, len(ref)), 3), len(ref)

def gen(text, seed, **kw):
    torch.manual_seed(int(seed) % (2**31 - 1))
    a = model.generate(text=text, language="ka", num_step=CFG["num_step"], **kw)[0]
    return np.asarray(a, dtype=np.float32).reshape(-1)

def f0_median(y, sr):
    try:
        import librosa
        f0, vf, _ = librosa.pyin(y, fmin=60, fmax=450, sr=sr, frame_length=2048)
        f0 = f0[vf & np.isfinite(f0)]
        return round(float(np.median(f0)), 1) if len(f0) else None
    except Exception:
        return None

def metrics(y, sr):
    """build_base_dataset.process-ის მსგავსი გაზომვა (20 ms ჩარჩოები)."""
    fr = int(sr * 0.02); n = len(y) // fr
    if n < 10:
        return {"dur": round(len(y) / sr, 3), "bad": "short"}
    db = 20 * np.log10(np.sqrt(np.mean(y[: n * fr].reshape(n, fr) ** 2, axis=1) + 1e-12))
    speech, noise = float(np.percentile(db, 90)), float(np.percentile(db, 10))
    voiced = np.where(db > max(noise + 12, speech - 35))[0]
    # შიდა პაუზა: ყველაზე გრძელი ჩუმი მონაკვეთი პირველ და ბოლო ხმოვან ჩარჩოს შორის
    gap = 0
    if len(voiced) > 1:
        gap = int(np.max(np.diff(voiced))) - 1
    return {"dur": round(len(y) / sr, 3), "speech_db": round(speech, 1), "snr": round(speech - noise, 1),
            "peak": round(float(np.max(np.abs(y))), 3), "clip": round(float(np.mean(np.abs(y) > 0.99)), 5),
            "max_gap_s": round(gap * 0.02, 2), "voiced": voiced}

def trim_norm(y, sr, m):
    """სიჩუმე მოჭრილი (0.15 / 0.2 წმ ზღვარი), ხმამაღლობა −20 dB-მდე, პიკი ≤ 0.89 (build_base_dataset.process)."""
    fr = int(sr * 0.02); v = m["voiced"]
    a = max(0, v[0] * fr - int(0.15 * sr)); b = min(len(y), (v[-1] + 1) * fr + int(0.2 * sr))
    y = y[a:b]
    gain = 10 ** ((-20 - m["speech_db"]) / 20)
    peak = float(np.max(np.abs(y))) * gain
    if peak > 0.89:
        gain *= 0.89 / peak
    return np.clip(y * gain, -1, 1)

def wav16(y, sr):
    b = io.BytesIO(); sf.write(b, y, sr, subtype="PCM_16", format="WAV"); return b.getvalue()

# ---------------- refs ----------------
def run_refs():
    d = OUT / "refs"; d.mkdir(exist_ok=True)
    res = []
    for v in job["voices"]:
        for seed in v["seeds"]:
            r = {"name": v["name"], "instruct": v["instruct"], "seed": seed}
            try:
                t = time.time()
                y = gen(CFG["ref_text"], seed, instruct=v["instruct"], speed=CFG["ref_speed"])
                r["design_s"] = round(time.time() - t, 2)
                m = metrics(y, SR); m.pop("voiced", None); r.update(m)
                h = heard(y); r["cer"], letters = cer(CFG["ref_text"], h); r["heard"] = h
                r["cps"] = round(letters / max(r["dur"], 1e-3), 2)
                r["f0"] = f0_median(resample(y, SR, 16000), 16000)
                fn = f"{v['name']}_s{seed}.wav"; (d / fn).write_bytes(wav16(y, SR)); r["file"] = fn
                prompt = model.create_voice_clone_prompt(ref_audio=(torch.from_numpy(y), SR), ref_text=CFG["ref_text"])
                tests = []
                for k, txt in enumerate(CFG["tests"]):
                    for sp in CFG["test_speeds"]:
                        t = time.time(); yt = gen(txt, seed * 10 + k, voice_clone_prompt=prompt, speed=sp); wall = time.time() - t
                        mt = metrics(yt, SR); mt.pop("voiced", None)
                        c, L = cer(txt, heard(yt))
                        tests.append({"k": k, "speed": sp, "wall": round(wall, 2), "dur": mt["dur"], "cer": c,
                                      "cps": round(L / max(mt["dur"], 1e-3), 2), "f0": f0_median(resample(yt, SR, 16000), 16000)})
                        if k == 0:
                            (d / f"{v['name']}_s{seed}_t{k}_x{sp}.wav").write_bytes(wav16(yt, SR))
                r["tests"] = tests
                log(v["name"], seed, "dur", r["dur"], "cer", r["cer"], "f0", r["f0"], "cps", r["cps"],
                    "tests", [(t_["speed"], t_["cer"], t_["cps"], round(t_["wall"] / max(t_["dur"], .1), 3)) for t_ in tests])
            except Exception as e:
                import traceback
                r["error"] = repr(e); r["tb"] = traceback.format_exc()[-1500:]
                log("ERROR", v["name"], seed, repr(e))
            res.append(r)
            (OUT / f"refs.part{GPU}.json").write_text(json.dumps(res, ensure_ascii=False, indent=0))

# ---------------- gen ----------------
def run_gen():
    deadline = job["deadline"]
    tars = OUT / "tars"; tars.mkdir(exist_ok=True)
    inp = Path(job["input"])
    for v in job["voices"]:
        if time.time() > deadline:
            log("deadline — skip", v["name"]); continue
        try:
            gen_voice(v, deadline, tars, inp)
        except Exception as e:  # ერთი ხმის შეცდომამ დანარჩენები არ უნდა გააჩეროს
            import traceback
            log("ERROR voice", v["name"], repr(e), traceback.format_exc()[-1500:])

def gen_voice(v, deadline, tars, inp):
    if True:
        ref = inp / "refs" / v["ref"]
        y_ref, sr_ref = sf.read(str(ref), dtype="float32")
        prompt = model.create_voice_clone_prompt(ref_audio=(torch.from_numpy(y_ref), sr_ref), ref_text=v["ref_text"])
        done = set(v.get("done", []))
        rows = [json.loads(l) for l in open(inp / "shards" / v["shard"], encoding="utf-8")]
        rows = [r for r in rows if r["id"] not in done]
        log(v["name"], "start", len(rows), "to do (", len(done), "done before )")
        chunk_i = len(list(tars.glob(f"{v['name']}.r{CFG['run']}.c*.tar")))
        pend, t_voice, aud, n = [], time.time(), 0.0, 0
        cdir = TMP / "clips" / v["name"]; cdir.mkdir(parents=True, exist_ok=True)

        def flush():
            nonlocal pend, chunk_i
            if not pend:
                return
            p = tars / f"{v['name']}.r{CFG['run']}.c{chunk_i:03d}.tar"
            with tarfile.open(str(p) + ".part", "w") as tf:
                data = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in pend).encode()
                ti = tarfile.TarInfo(f"{v['name']}/rows.c{CFG['run']}_{chunk_i:03d}.jsonl"); ti.size = len(data)
                tf.addfile(ti, io.BytesIO(data))
                for r in pend:
                    if r.get("file"):
                        tf.add(str(cdir / r["file"]), arcname=f"{v['name']}/{r['file']}")
            os.replace(str(p) + ".part", p)
            for r in pend:
                if r.get("file"):
                    (cdir / r["file"]).unlink(missing_ok=True)
            chunk_i += 1; pend = []

        for i, s in enumerate(rows):
            if time.time() > deadline:
                log("deadline reached in", v["name"], "after", n); break
            txt = s.get("omni_text") or s["text"]
            best = None
            for att in range(CFG["attempts"]):
                seed = v["seed"] * 1000003 + int(s["id"].lstrip("oc") or 0) * 7 + att * 104729
                t = time.time()
                try:
                    y = gen(txt, seed, voice_clone_prompt=prompt, speed=v["speed"])
                    if len(y) < SR // 5:  # ცარიელი/ძალიან მოკლე (postprocess-მა ყველაფერი მოჭრა) — 2026-11-09 m04 ამაზე ჩავარდა
                        raise ValueError(f"empty audio ({len(y)} samples)")
                    wall = time.time() - t
                    h = heard(y); c, L = cer(s["text"], h)
                    m = metrics(y, SR)
                except Exception as e:
                    log("ERROR gen", s["id"], att, repr(e)); torch.cuda.empty_cache(); continue
                cps = L / max(m["dur"], 1e-3)
                bad = (c > (0.10 if L >= 15 else 0.05)) or not (7 <= cps <= 24) or m.get("bad") or m.get("clip", 0) > 0.001
                cand = {"y": y, "m": m, "cer": c, "heard": h, "letters": L, "att": att + 1, "seed": seed, "wall": wall, "bad": bool(bad)}
                if best is None or (cand["bad"], cand["cer"]) < (best["bad"], best["cer"]):
                    best = cand
                if not bad:
                    break
            if best is None:
                pend.append({"id": s["id"], "voice": v["name"], "error": "gen"}); continue
            m = best["m"]
            row = {"id": s["id"], "voice": v["name"], "text": s["text"], "cer": best["cer"], "heard": best["heard"],
                   "letters": best["letters"], "attempts": best["att"], "seed": best["seed"], "bad": best["bad"],
                   "raw_dur": m["dur"], "snr": m.get("snr"), "peak": m.get("peak"), "clip": m.get("clip"),
                   "max_gap_s": m.get("max_gap_s"), "gen_s": round(best["wall"], 2)}
            if "voiced" in m and len(m["voiced"]):
                y22 = resample(best["y"], SR, 22050)
                m22 = metrics(y22, 22050)
                if "voiced" in m22 and len(m22["voiced"]):
                    yt = trim_norm(y22, 22050, m22)
                    fn = f"{s['id']}.flac"
                    sf.write(str(cdir / fn), yt, 22050, subtype="PCM_16", format="FLAC")
                    row.update(file=fn, dur=round(len(yt) / 22050, 3), cps=round(best["letters"] / max(len(yt) / 22050, 1e-3), 2))
                    aud += row["dur"]
            pend.append(row); n += 1
            if len(pend) >= CFG["chunk"]:
                flush()
            if n % 100 == 0:
                el = time.time() - t_voice
                log(v["name"], f"{n}/{len(rows)}", f"audio {aud/60:.1f} min", f"RTF {el/max(aud,1):.3f}",
                    f"left {max(0, deadline - time.time())/3600:.2f} h")
        flush()
        el = time.time() - t_voice
        log(v["name"], "done", n, f"audio {aud/3600:.2f} h, wall {el/3600:.2f} h, RTF {el/max(aud,1):.3f}")
        st = OUT / f"voice_{v['name']}.r{CFG['run']}.json"
        st.write_text(json.dumps({"voice": v["name"], "clips": n, "todo": len(rows), "audio_s": round(aud, 1),
                                  "wall_s": round(el, 1), "complete": n >= len(rows)}))

if CFG["mode"] == "refs":
    run_refs()
else:
    run_gen()
log("worker finished")
'''


def log(*a):
    msg = f"[{time.time() - T0:8.1f}s] " + " ".join(str(x) for x in a)
    print(msg, flush=True)
    with open(OUT / "main.log", "a", encoding="utf-8") as f:
        f.write(msg + "\n")


def main():
    r = subprocess.run([sys.executable, "-m", "pip", "install", "-q", CFG["pip"], "soundfile"], capture_output=True, text=True)
    log("pip rc", r.returncode, (r.stdout + r.stderr)[-1500:])
    import torch
    n = max(1, torch.cuda.device_count())
    log("GPUs", [torch.cuda.get_device_name(i) for i in range(n)])
    Path("/tmp/omni_gen_worker.py").write_text(WORKER)
    inp = None
    if CFG["mode"] == "gen":
        inp = str(Path(next(iter(glob.glob(os.environ.get("OMNI_IN", "/kaggle/input") + "/**/voices.json", recursive=True)))).parent)
    deadline = T0 + CFG.get("budget_s", 36000)
    # ხმები GPU-ებზე: დარჩენილი წინადადებების რაოდენობით დაბალანსებული
    voices = CFG["voices"]
    loads = [0] * n
    parts = [[] for _ in range(n)]
    for v in sorted(voices, key=lambda v: -(v.get("todo") or len(v.get("seeds", [1])))):
        g = loads.index(min(loads))
        parts[g].append(v)
        loads[g] += v.get("todo") or len(v.get("seeds", [1]))
    procs = []
    for g in range(n):
        if not parts[g]:
            continue
        parts[g].sort(key=lambda v: v["name"])
        jp = f"/tmp/omni_job{g}.json"
        Path(jp).write_text(json.dumps({"cfg": {k: v for k, v in CFG.items() if k != "voices"}, "voices": parts[g],
                                        "gpu": g, "deadline": deadline, "input": inp}, ensure_ascii=False))
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(g), PYTHONUNBUFFERED="1")
        procs.append(subprocess.Popen([sys.executable, "/tmp/omni_gen_worker.py", jp], env=env))
        log("gpu", g, [v["name"] for v in parts[g]])
    codes = [p.wait() for p in procs]
    st = {"codes": codes, "wall_s": round(time.time() - T0, 1), "mode": CFG["mode"], "run": CFG.get("run")}
    if CFG["mode"] == "refs":
        res = []
        for p in sorted(OUT.glob("refs.part*.json")):
            res += json.loads(p.read_text(encoding="utf-8"))
        (OUT / "refs.json").write_text(json.dumps(res, ensure_ascii=False, indent=1), encoding="utf-8")
    else:
        st["voices"] = [json.loads(p.read_text()) for p in sorted(OUT.glob("voice_*.json"))]
    (OUT / "status.json").write_text(json.dumps(st, ensure_ascii=False, indent=1))
    log("done", st)


if __name__ == "__main__":
    main()
