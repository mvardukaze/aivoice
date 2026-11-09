"""OmniVoice-ის (k2-fsa/OmniVoice) ქართული გენერაციის სიჩქარის ტესტი Kaggle-ის უფასო GPU-ზე.

გამოყენება (პროექტის .venv-ით):
    .venv/bin/python tools/kaggle_omni_bench.py sentences          # 40 წინადადება (ka_text) → work/omni_bench/sentences.json
    .venv/bin/python tools/kaggle_omni_bench.py push [--gpu t4|p100] [--quick]
    .venv/bin/python tools/kaggle_omni_bench.py status [--gpu ...]
    .venv/bin/python tools/kaggle_omni_bench.py fetch [--gpu ...]  # → /mnt/d/aivoice/omnivoice/bench/<gpu>/
    .venv/bin/python tools/kaggle_omni_bench.py report [--gpu ...]

Kernel-ი პირადია, GPU + ინტერნეტით (ისევე, როგორც tools/kaggle_train.py). ავტორიზაცია: ~/.kaggle/access_token.
დროის ლიმიტი: kernel-ის -t timeout + სკრიპტის შიდა budget_s (Kaggle-ის 30 სთ/კვირა წვრთნასთან საერთოა).
"""

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from kaggle_train import WORK, kaggle, user  # noqa: E402

TEMPLATE = ROOT / "kaggle" / "kernel_omni_bench.py"
PIP = "omnivoice @ git+https://github.com/k2-fsa/OmniVoice.git@08be0b4ccbac3e13e374e86fbfead4b4cac343e2"
VOICES = {"female": "female, young adult, moderate pitch", "male": "male, middle-aged, low pitch"}
SENTS = WORK / "omni_bench" / "sentences.json"
OUT = Path("/mnt/d/aivoice/omnivoice/bench")
SHAPES = {"t4": "NvidiaTeslaT4", "p100": "NvidiaTeslaP100"}
# (kernel timeout, შიდა budget) წამებში
LIMITS = {False: (1980, 1200), True: (720, 240)}


def cmd_sentences(a):
    from app.textnorm import ka_text
    s = json.loads((ROOT / "data" / "prompts" / "voice_texts_v1.json").read_text(encoding="utf-8"))["sentences"]
    random.seed(7)
    random.shuffle(s)
    short = [x for x in s if 32 <= len(x) <= 48][:16]
    med = [x for x in s if 85 <= len(x) <= 105][:12]
    lng = [x for x in s if 135 <= len(x) <= 165][:12]
    pool = [x for x in s if 60 <= len(x) <= 80]
    while len(lng) < 12:  # CV-ში 150-სიმბოლოიანი იშვიათია — ორი წინადადება ერთად
        lng.append(pool.pop() + " " + pool.pop())
    out = [{"id": f"b{i:02d}", "text": ka_text(t)} for i, t in enumerate(short + med + lng)]
    SENTS.parent.mkdir(parents=True, exist_ok=True)
    SENTS.write_text(json.dumps(out, ensure_ascii=False, indent=0), encoding="utf-8")
    print(SENTS, len(out), "sentences,", sum(len(o["text"]) for o in out), "chars")


def kid(gpu):
    return f"{user()}/aivoice-omni-bench-{gpu}"


def cmd_push(a):
    if not SENTS.exists():
        cmd_sentences(a)
    timeout, budget = LIMITS[a.quick]
    cfg = {"sentences": json.loads(SENTS.read_text(encoding="utf-8")), "voices": VOICES, "pip": PIP,
           "budget_s": budget, "quick": a.quick}
    k = WORK / f"kernel-omni-bench-{a.gpu}"
    k.mkdir(parents=True, exist_ok=True)
    (k / "bench.py").write_text(TEMPLATE.read_text(encoding="utf-8").replace(
        "__CONFIG__", json.dumps(cfg, ensure_ascii=False).replace("\\", "\\\\")), encoding="utf-8")
    (k / "kernel-metadata.json").write_text(json.dumps({
        "id": kid(a.gpu), "title": f"aivoice omni bench {a.gpu}", "code_file": "bench.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_internet": True, "machine_shape": SHAPES[a.gpu],
        "dataset_sources": [],
    }))
    kaggle("kernels", "push", "-p", str(k), "-t", str(timeout))


def cmd_status(a):
    print(kaggle("kernels", "status", kid(a.gpu), check=False).stdout.strip())


def cmd_fetch(a):
    d = OUT / a.gpu
    d.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", kid(a.gpu), "-p", str(d), "-o")
    cmd_report(a)


def cmd_report(a):
    r = json.loads((OUT / a.gpu / "results.json").read_text(encoding="utf-8"))
    print(json.dumps(r["env"], ensure_ascii=False))
    print(f"{'config':32} {'gpu':24} {'audio_s':>8} {'wall_s':>8} {'RTF':>7} {'aud h/GPU h':>11} {'VRAM GB':>7}")
    for c in r["configs"]:
        print(f"{c['name']:32} {c['gpu'][:24]:24} {c['audio_s']:8.1f} {c['wall_s']:8.1f} {c['rtf']:7.3f} "
              f"{c['audio_h_per_gpu_h']:11.1f} {c.get('peak_vram_gb', '-')!s:>7}")
    for e in r["errors"]:
        print("ERROR", e.get("stage"), e.get("error"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("cmd", choices=["sentences", "push", "status", "fetch", "report"])
    p.add_argument("--gpu", choices=list(SHAPES), default="t4")
    p.add_argument("--quick", action="store_true")
    a = p.parse_args()
    globals()[f"cmd_{a.cmd}"](a)


if __name__ == "__main__":
    main()
