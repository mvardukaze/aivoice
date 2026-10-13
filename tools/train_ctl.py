"""ხმების წვრთნის მართვა — ერთი ადგილი (ბრძანება და პანელის „წვრთნის“ ჩანართი ამას იყენებს).

    ./aivoice train list
    ./aivoice train new f_mix_02 --design "ქალი 2" [--name "ქალი 2"] [--epochs 150]
    ./aivoice train prepare f_mix_02            # ჩანაწერები (ამ კომპიუტერზე) + საწყისი checkpoint
    ./aivoice train start f_mix_02 --on kaggle  # ან --on local ; --wait — დაელოდოს დასრულებას
    ./aivoice train status [f_mix_02]
    ./aivoice train wait f_mix_02               # ლოდინი + შედეგის ჩამოტვირთვა და ბიბლიოთეკაში ჩასმა
    ./aivoice train stop f_mix_02
    ./aivoice train fetch f_mix_02

მდგომარეობა: <paths.runs>/<slot>/status.json (+ ctl.json — ეტაპი, სად, როდის). Job: data/kaggle_jobs/<slot>.json.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from app.config import cfg, path as cpath  # noqa: E402

JOBS = ROOT / "data" / "kaggle_jobs"
DESIGNS = ROOT / "data" / "designs"
TPY = ROOT / "training" / ".venv" / "bin" / "python"
LOCAL_CTRL = ROOT / "data" / "local_train.json"
POLL_S = 600


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def read_json(p: Path, default=None):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write_json(p: Path, d):
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(d, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(p)


def run_dir(slot) -> Path:
    return cpath("runs") / slot


def ctl(slot, **kw) -> dict:
    """ctl.json — რა ეტაპზეა (prepare / training / done / failed / stopped), სად (kaggle / local), შენიშვნა."""
    p = run_dir(slot) / "ctl.json"
    d = read_json(p, {}) or {}
    if kw:
        d.update(kw, updated=now())
        write_json(p, d)
    return d


def job(slot) -> dict:
    j = read_json(JOBS / f"{slot}.json")
    if not j:
        raise SystemExit(f"ასეთი ხმა არ არის: {slot} (ჯერ: ./aivoice train new {slot} --design ...)")
    return j


def design(name) -> dict:
    for p in DESIGNS.glob("*.json"):
        d = read_json(p, {})
        if d.get("name") == name:
            return d
    raise SystemExit(f"შენახული ნარევი ვერ ვიპოვე: {name}")


def watcher_running() -> bool:
    """ამ კომპიუტერზე Kaggle-ის მეთვალყურე სერვისი (systemd) თავად ჩამოტვირთავს შედეგს."""
    if not shutil.which("systemctl"):
        return False
    r = subprocess.run(["systemctl", "is-active", "aivoice-kaggle"], capture_output=True, text=True)
    return r.stdout.strip() == "active"


# ---------------- ბრძანებები ----------------
def cmd_list(a):
    for p in sorted(JOBS.glob("*.json")):
        slot = p.stem
        st = read_json(run_dir(slot) / "status.json", {}) or {}
        c = ctl(slot)
        print(f"{slot:12} {c.get('stage') or st.get('status') or '—':10} {c.get('target') or st.get('device') or '':7} "
              f"ეპოქა {(st.get('epoch') or 0) + 1}/{read_json(p, {}).get('max_epochs')}  {c.get('note') or ''}")


def cmd_new(a):
    if not re.fullmatch(r"[a-z][a-z0-9_]{1,30}", a.slot):
        raise SystemExit("სახელი: ლათინური პატარა ასოები, ციფრები, _ (მაგ. f_mix_02)")
    if (JOBS / f"{a.slot}.json").exists() and not a.force:
        raise SystemExit(f"{a.slot} უკვე არსებობს (--force — თავიდან)")
    d = design(a.design)
    j = {"_note": f"„{a.design}“ — შენახული ნარევი → ცალკე ხმა (train_ctl.py)", "design": a.design,
         "source_dir": str(cpath("datasets") / a.slot), "init_dir": str(cpath("work") / f"init-{a.slot}"),
         "num_speakers": 1, "batch": 24, "devices": 2, "max_epochs": a.epochs, "warmstart": "init.ckpt",
         "audio_from_source": True, "trim_silence": False, "preview_speakers": [],
         "inference": {"noise_scale": d.get("noise_scale", 0.667), "length_scale": d.get("length_scale", 1.0),
                       "noise_w": d.get("noise_w_scale", 0.8)}}
    write_json(JOBS / f"{a.slot}.json", j)
    cat_p = ROOT / "voices" / "catalog.json"
    cat = read_json(cat_p, {"voices": []})
    if not any(v["id"] == a.slot for v in cat["voices"]):
        cat["voices"].append({"id": a.slot, "name": a.name or a.design, "gender": a.gender or "", "age": "", "character": "",
                              "status": "planned", "source": f"საკუთარი სინთეზური ხმა: ნარევი „{a.design}“", "license": "own",
                              "base": "base_ka"})
        cat_p.write_text(json.dumps(cat, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    ctl(a.slot, stage="new", target=None, note="შექმნილია — შემდეგი: prepare")
    print(f"✓ {a.slot} შექმნილია. შემდეგი ნაბიჯი: ./aivoice train prepare {a.slot}")


def cmd_prepare(a):
    """ჩანაწერების გენერაცია (ნარევით, MMS-ით შემოწმებული) + საწყისი checkpoint (ნარევი „ჩაცხობილი“)."""
    if not TPY.exists():
        raise SystemExit("საჭიროა წვრთნის გარემო: ./install.sh --training")
    j = job(a.slot)
    texts = ROOT / "data" / "prompts" / "voice_texts_v1.json"
    log = (run_dir(a.slot)).joinpath("ctl.log")
    log.parent.mkdir(parents=True, exist_ok=True)
    steps = []
    if not texts.exists():
        steps.append(("წინადადებები", [sys.executable, "tools/make_voice_texts.py"]))
    steps.append(("ჩანაწერები", [str(TPY), "tools/make_voice_dataset.py", "--design", j["design"], "--slot", a.slot]))
    init = Path(j["init_dir"]) / j["warmstart"]
    if not init.exists() or a.force:
        steps.append(("საწყისი checkpoint", [str(TPY), "tools/fold_speaker.py", "--design", j["design"], "--out", str(init)]))
    for name, cmd in steps:
        ctl(a.slot, stage="prepare", note=f"მზადდება: {name}", error=None)
        print(f"==> {name}", flush=True)
        with log.open("a", encoding="utf-8") as lf:
            lf.write(f"\n==== {now()} {name}: {' '.join(cmd)}\n")
            lf.flush()
            r = subprocess.run(cmd, cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT, env={**__import__("os").environ, "CUDA_VISIBLE_DEVICES": ""})
        if r.returncode != 0:
            ctl(a.slot, stage="failed", note=f"„{name}“ ჩავარდა — იხ. {log}", error=f"exit {r.returncode}")
            raise SystemExit(f"✗ {name} ჩავარდა — იხ. {log}")
    n = sum(1 for _ in (Path(j["source_dir"]) / "metadata.csv").open(encoding="utf-8"))
    ctl(a.slot, stage="prepared", note=f"მზადაა: {n} ჩანაწერი — შემდეგი: start")
    print(f"✓ მზადაა ({n} ჩანაწერი). შემდეგი: ./aivoice train start {a.slot} --on kaggle")


def cmd_start(a):
    j = job(a.slot)
    if a.epochs:
        j["max_epochs"] = a.epochs
        write_json(JOBS / f"{a.slot}.json", j)
    if not (Path(j.get("source_dir") or j.get("dataset_dir", "")) / "metadata.csv").exists():
        raise SystemExit(f"ჩანაწერები ჯერ არ არის — ./aivoice train prepare {a.slot}")
    if a.on == "kaggle":
        import kaggle_train as kt
        if not kt.quota_ok():
            print(f"! Kaggle: ბოლო 7 დღეში {kt.kaggle_hours_7d()} სთ — შეიძლება კვოტა არ ეყოს (kaggle.com-ზე შეამოწმეთ)")
        ctl(a.slot, stage="uploading", target="kaggle", note="Kaggle-ზე ატვირთვა…", error=None, started=now())
        kt.cmd_push(argparse.Namespace(slot=a.slot))
        ctl(a.slot, stage="training", target="kaggle", note="Kaggle-ზე იწვრთნება")
        print(f"✓ Kaggle-ზე გაეშვა: https://www.kaggle.com/code/{kt.user()}/aivoice-train-{kt.slug(a.slot)}")
    elif a.on == "local":
        write_json(LOCAL_CTRL, {"slot": a.slot, "run": True, "reason": "ხელით გაშვება (train_ctl)", "at": now()})
        ctl(a.slot, stage="training", target="local", note="ამ კომპიუტერზე იწვრთნება (CPU — ნელია)", error=None, started=now())
        if not local_service_running():
            print("ლოკალური წვრთნა ამ ფანჯარაში (Ctrl+C — გაჩერება; შემდეგ ჯერზე იქიდან გააგრძელებს)")
            subprocess.run([str(TPY), "tools/local_train.py"], cwd=ROOT)
            return
        print("✓ ლოკალური წვრთნის სერვისი იწყებს (aivoice-local)")
    else:
        raise SystemExit("Google Cloud ჯერ მხოლოდ საბაზო მოდელისთვისაა (./aivoice gcp ...) — ხმებისთვის: kaggle ან local")
    if a.wait:
        cmd_wait(a)


def local_service_running() -> bool:
    if not shutil.which("systemctl"):
        return False
    r = subprocess.run(["systemctl", "is-active", "aivoice-local"], capture_output=True, text=True)
    return r.stdout.strip() == "active"


def cmd_status(a):
    slots = [a.slot] if a.slot else [p.stem for p in sorted(JOBS.glob("*.json"))]
    for s in slots:
        st = read_json(run_dir(s) / "status.json", {}) or {}
        c = ctl(s)
        loc = read_json(run_dir(s) / "local" / "state.json", {}) or {}
        print(f"■ {s}: {c.get('stage') or st.get('status') or '—'} · {c.get('target') or st.get('device') or ''}"
              f" · ეპოქა {(st.get('epoch') or loc.get('epoch') or 0) + 1}/{job(s).get('max_epochs')}"
              f"{' · ' + c['note'] if c.get('note') else ''}{' · შეცდომა: ' + str(st.get('error') or c.get('error')) if (st.get('error') or c.get('error')) else ''}")


def cmd_wait(a):
    """ელოდება დასრულებას; თუ ამ კომპიუტერზე მეთვალყურე სერვისი არ არის — თავად ჩამოტვირთავს და აქვეყნებს."""
    import kaggle_train as kt
    print("ველოდები… (Ctrl+C — ლოდინის შეწყვეტა; წვრთნა გაგრძელდება)", flush=True)
    while True:
        c, st = ctl(a.slot), read_json(run_dir(a.slot) / "status.json", {}) or {}
        target = c.get("target") or st.get("device")
        if target == "kaggle":
            if not watcher_running() and st.get("status") == "training":
                s = kt.cmd_status(argparse.Namespace(slot=a.slot))
                if "complete" in s or "error" in s or "cancel" in s:
                    kt.cmd_fetch(argparse.Namespace(slot=a.slot))
                    st = read_json(run_dir(a.slot) / "status.json", {}) or {}
            if st.get("status") in ("done", "partial", "failed"):
                return finish(a.slot, st)
        elif target == "local":
            loc = read_json(run_dir(a.slot) / "local" / "state.json", {}) or {}
            if loc.get("done") or (not loc.get("running") and loc.get("note", "").startswith("დასრულდა")):
                return finish(a.slot, {"status": "done", "epoch": loc.get("epoch")})
            print(f"  {now()} ეპოქა {(loc.get('epoch') or 0) + 1} · {loc.get('step_in_epoch') or 0}/{loc.get('steps_per_epoch') or '?'}", flush=True)
        print(f"  {now()} {st.get('kaggle_state') or st.get('status') or ''}", flush=True)
        time.sleep(POLL_S)


def finish(slot, st):
    j = job(slot)
    if st.get("status") == "done":
        v = ROOT / "voices" / slot / f"{slot}.onnx.json"
        if v.exists() and j.get("inference"):  # ნარევის ინტონაცია/ტემპი — ხმის ნაგულისხმევად
            c = json.loads(v.read_text(encoding="utf-8"))
            c["inference"] = j["inference"]
            v.write_text(json.dumps(c, ensure_ascii=False, indent=2), encoding="utf-8")
        ctl(slot, stage="done", note="მზადაა — „ხმების“ ჩანართში", finished=now())
        print(f"✓ {slot} მზადაა — „ხმების“ ჩანართში")
    else:
        ctl(slot, stage=st.get("status"), note=str(st.get("error") or "იხ. სტატუსი"))
        print(f"! {slot}: {st.get('status')} {st.get('error') or ''}")


def cmd_stop(a):
    c = ctl(a.slot)
    if c.get("target") == "local":
        write_json(LOCAL_CTRL, {"slot": a.slot, "run": False, "reason": "ხელით გაჩერება", "at": now()})
        ctl(a.slot, stage="stopped", note="გაჩერებულია (შემდეგ ჯერზე იქიდან გააგრძელებს)")
        print("✓ ლოკალური წვრთნა ჩერდება")
    elif c.get("target") == "kaggle":
        import kaggle_train as kt
        ctl(a.slot, stage="stopped", note="Kaggle-ზე გააჩერეთ ვებგვერდიდან („Cancel run“)")
        print(f"Kaggle-ის CLI-ს გაჩერება არ შეუძლია — გააჩერეთ აქ: https://www.kaggle.com/code/{kt.user()}/aivoice-train-{kt.slug(a.slot)}")
    else:
        print("ამ ხმას ახლა არაფერი ეშვება")


def cmd_fetch(a):
    import kaggle_train as kt
    kt.cmd_fetch(argparse.Namespace(slot=a.slot))
    finish(a.slot, read_json(run_dir(a.slot) / "status.json", {}) or {})


def main():
    ap = argparse.ArgumentParser(prog="./aivoice train")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    p = sub.add_parser("new")
    p.add_argument("slot")
    p.add_argument("--design", required=True)
    p.add_argument("--name")
    p.add_argument("--gender", default="")
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("prepare")
    p.add_argument("slot")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("start")
    p.add_argument("slot")
    p.add_argument("--on", choices=["kaggle", "local", "gcp"], default="kaggle")
    p.add_argument("--epochs", type=int)
    p.add_argument("--wait", action="store_true")
    for c in ("status",):
        sub.add_parser(c).add_argument("slot", nargs="?")
    for c in ("wait", "stop", "fetch"):
        sub.add_parser(c).add_argument("slot")
    a = ap.parse_args()
    {"list": cmd_list, "new": cmd_new, "prepare": cmd_prepare, "start": cmd_start, "status": cmd_status,
     "wait": cmd_wait, "stop": cmd_stop, "fetch": cmd_fetch}[a.cmd](a)


if __name__ == "__main__":
    main()
