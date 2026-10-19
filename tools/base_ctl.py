"""საბაზო (მრავალხმიანი) მოდელის შექმნა ნულიდან ან არსებულის გაგრძელება — ერთი ადგილი (ბრძანება და პანელი).

    ./aivoice base options                                   რა შეიძლება (მოდელები, შერჩევები, GPU, Google Cloud)
    ./aivoice base new base_ka_v4 --mode continue --from base_ka_v3 --hours 40 --short --punct --clusters --epochs 90
    ./aivoice base new base_ka_v4 --mode scratch --hours 40 --epochs 160 [--selection <არსებული შერჩევა>]
    ./aivoice base prepare base_ka_v4                        dataset (+ არქივი Google Cloud-ისთვის)
    ./aivoice base start base_ka_v4 --on gcp|local           Google Cloud L4 (on-demand) ან ამ კომპიუტერის GPU (≥ 12 GB)
    ./aivoice base status [base_ka_v4] | stop base_ka_v4 | finalize base_ka_v4
    ყველგან: --dry-run — მხოლოდ აჩვენებს, რა მოხდებოდა (არაფერს ქმნის/უშვებს)

გაგრძელება: ახალი შერჩევა (<slot>_add) — მხოლოდ ჯერ გამოუყენებელი ჩანაწერებით; ერწყმის წინა მოდელის გაშვებულ
შერჩევას (base = მისი სახელი) — build_selection_dataset.py წინა მოდელის დინამიკებს იმავე რიგით პირველებად წერს
(Piper-ი ID-ებს metadata.csv-ში გამოჩენით ანიჭებს). წვრთნა გრძელდება წინა მოდელის final.ckpt-იდან.
მდგომარეობა: <runs>/<slot>/ctl.json (ეტაპი), status.json (ეპოქა), gpu/state.json (ლოკალური GPU).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))
from app.config import cfg, expand, path as cpath  # noqa: E402
from train_ctl import ctl, now, read_json, run_dir, write_json  # noqa: E402

JOBS = ROOT / "data" / "kaggle_jobs"
SEL = ROOT / "data" / "selections"
PY = sys.executable
TPY = ROOT / "training" / ".venv" / "bin" / "python"
LJSPEECH = ROOT / "training" / "checkpoints" / "en_US-ljspeech-medium" / "ljspeech-medium.clean.ckpt"
SLOT_RE = re.compile(r"base_ka_v\d{1,3}")  # ხმის შემქმნელი designer_base_ka_v*-ს პოულობს (app/designer.py)
NUM_SPEAKERS = 600  # წინასწარ — მომავალი გაგრძელებების ახალი დინამიკებისთვის
GCP_BATCH_PER_S, GCP_USD_PER_H, GCP_BATCH = 0.92, 0.9, 32  # L4 on-demand (base_ka_v2/v3-ის გამოცდილებით)
CLIPS_PER_HOUR = 750  # შერჩევის საშუალო (v2: 782, v3_add: 721)
MIN_VRAM_GB = 12
GCP_ACTIVE = ("created", "setup", "setup_failed", "training", "stopped")
PROGRESS = re.compile(r"Epoch (\d+):\s+(\d+)%\|[^|]*\|\s*(\d+)/(\d+) \[([^<]+)<([^,]+),\s*([\d.]+)(s/it|it/s)")


# ---------------- ინფორმაცია ----------------
def selection(name: str) -> dict | None:
    return read_json(SEL / f"{name}.json") if name else None


def launched_selection(model: str) -> dict | None:
    """მოდელის გაშვებული შერჩევა (speaker_order-ით) — გაგრძელებისთვის აუცილებელია."""
    for p in sorted(SEL.glob("*.json")):
        s = read_json(p, {}) or {}
        if s.get("launched") and (s.get("model") or p.stem) == model and s.get("speaker_order"):
            return s
    return None


def models() -> list[dict]:
    """დასრულებული საბაზო მოდელები, რომელთა გაგრძელებაც შეიძლება (final.ckpt + შერჩევა speaker_order-ით)."""
    out = []
    for p in sorted(JOBS.glob("*.json")):
        j = read_json(p, {}) or {}
        slot = p.stem
        if (j.get("num_speakers") or 1) <= 1:
            continue
        final = run_dir(slot) / "kaggle_out" / "final.ckpt"
        st = read_json(run_dir(slot) / "kaggle_out" / "status.json", {}) or read_json(run_dir(slot) / "status.json", {}) or {}
        sel = launched_selection(slot)
        if not (final.exists() and sel and st.get("epoch") is not None and st.get("status") == "done"):
            continue
        out.append({"slot": slot, "epoch": int(st["epoch"]) + 1, "selection": sel["name"], "clips": sel["stats"]["clips"],
                    "hours": sel["stats"]["hours"], "speakers": len(sel["speaker_order"]), "checkpoint": str(final)})
    return out


def selections() -> list[dict]:
    """ჯერ გამოუყენებელი შერჩევები (გაშვებულები და სხვის ნაწილად შესულები — არა)."""
    all_ = {p.stem: read_json(p, {}) or {} for p in sorted(SEL.glob("*.json"))}
    parts = {x for s in all_.values() for x in s.get("parts", [])}
    jobs = {p.stem for p in JOBS.glob("*.json")}
    return [{"name": n, "hours": s.get("stats", {}).get("hours"), "clips": s.get("stats", {}).get("clips"),
             "speakers": s.get("stats", {}).get("speakers"), "created": s.get("created")}
            for n, s in all_.items() if not s.get("launched") and n not in parts and n not in jobs]


def quality_ready() -> bool:
    q = cpath("data_root") / "analysis" / "cv27"
    return (q / "clips.jsonl").exists() and ((q / "mms.jsonl").exists() or (q / "mms.cpu.jsonl").exists())


def gpu_info() -> dict | None:
    """ამ კომპიუტერის NVIDIA GPU (nvidia-smi) — None, თუ არ არის."""
    if not shutil.which("nvidia-smi"):
        return None
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20)
    except (OSError, subprocess.TimeoutExpired):
        return None
    gpus = []
    for line in r.stdout.strip().splitlines():
        name, _, mem = line.rpartition(",")
        try:
            gpus.append((float(mem) / 1024, name.strip()))
        except ValueError:
            pass
    if r.returncode != 0 or not gpus:
        return None
    vram, name = max(gpus)
    ok = vram >= MIN_VRAM_GB - 0.5  # „12 GB“ ბარათები ~11.7-ს აჩვენებს
    return {"name": name, "vram_gb": round(vram, 1), "ok": ok, "batch": 32 if vram >= 20 else 16 if ok else None,
            "reason": None if ok else f"GPU-ს {vram:.0f} GB მეხსიერება აქვს — საბაზო მოდელს სჭირდება ≥ {MIN_VRAM_GB} GB"}


def gcp_state() -> dict:
    g = cfg().get("gcp", {}).get("gcloud", "gcloud")
    gbin = str(expand(g)) if "/" in g else shutil.which(g)
    st = read_json(cpath("gcp") / "state.json", {}) or {}
    return {"configured": bool(gbin and Path(gbin).exists()), "slot": st.get("slot"), "phase": st.get("phase"),
            "busy": st.get("phase") in GCP_ACTIVE, "progress": st.get("progress"), "error": st.get("error")}


def next_slot() -> str:
    n = [int(m.group(1)) for p in list(JOBS.glob("base_ka_v*.json")) + list(SEL.glob("base_ka_v*.json"))
         if (m := re.fullmatch(r"base_ka_v(\d+)", p.stem))]
    return f"base_ka_v{max(n, default=1) + 1}"


def estimate(clips: int, epochs: int, batch: int = GCP_BATCH) -> dict:
    per = max(1, round(clips * 0.99 / batch))  # 1% — ვალიდაცია
    h = epochs * per / GCP_BATCH_PER_S / 3600
    return {"batches_per_epoch": per, "hours": round(h, 1), "usd": round(h * GCP_USD_PER_H)}


def options() -> dict:
    return {"models": models(), "selections": selections(), "quality": quality_ready(), "gcp": gcp_state(),
            "gpu": gpu_info(), "next_slot": next_slot(), "training_env": TPY.exists(),
            "rates": {"batch_per_s": GCP_BATCH_PER_S, "usd_per_h": GCP_USD_PER_H, "batch": GCP_BATCH,
                      "clips_per_hour": CLIPS_PER_HOUR, "scratch_epochs": 160, "continue_epochs": 90}}


def job(slot) -> dict:
    j = read_json(JOBS / f"{slot}.json")
    if not j or (j.get("num_speakers") or 1) <= 1:
        raise SystemExit(f"ასეთი საბაზო მოდელი არ არის: {slot} (ჯერ: ./aivoice base new {slot} ...)")
    return j


def log_path(slot) -> Path:
    p = run_dir(slot) / "ctl.log"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def run_step(slot, name, cmd, env=None) -> int:
    """ბრძანება ლოგში (runs/<slot>/ctl.log)."""
    print(f"==> {name}", flush=True)
    with log_path(slot).open("a", encoding="utf-8") as lf:
        lf.write(f"\n==== {now()} {name}: {' '.join(map(str, cmd))}\n")
        lf.flush()
        return subprocess.run(list(map(str, cmd)), cwd=ROOT, stdout=lf, stderr=subprocess.STDOUT,
                              env={**os.environ, **(env or {})}).returncode


# ---------------- new ----------------
def select_flags(a) -> list[str]:
    f = []
    if a.short:  # მოკლე ფრაზები („კი“, „არა, მადლობა“) და ოდნავ სწრაფი/ნელი მეტყველება
        f += ["--min-letters", "6", "--dur-min", "1", "--cps", "6", "18"]
    if a.punct:
        f += ["--punct-bonus", "100"]
    if a.clusters:
        f += ["--cluster-bonus", "8"]
    return f


def merge(base: dict, add: dict, name: str) -> dict:
    """base_ka_v3.json-ის მსგავსად: წინა მოდელის შერჩევა + ახალი ნაწილი; base = წინა (მისი speaker_order)."""
    have = set(base["clips"])
    extra = [c for c in add["clips"] if c not in have]
    speakers = {s: dict(v) for s, v in base["speakers"].items()}
    for s, v in add["speakers"].items():
        if s in speakers:
            speakers[s]["minutes"] = round(speakers[s]["minutes"] + v["minutes"], 1)
            speakers[s]["clips"] += v["clips"]
        else:
            speakers[s] = dict(v)
    low = [s for s, v in speakers.items() if (v.get("f0") or 999) < 165]
    nb, na = base["stats"]["clips"], len(extra)
    rules = base["rules"] if all(isinstance(v, dict) for v in base["rules"].values()) else {base["name"]: base["rules"]}
    stats = {"hours": round(sum(v["minutes"] for v in speakers.values()) / 60, 1), "clips": nb + na,
             "speakers": len(speakers), "low_speakers": len(low),
             "new_speakers": sum(1 for v in speakers.values() if not v.get("current")),
             "mean_cer": round((base["stats"].get("mean_cer", 0) * nb + add["stats"].get("mean_cer", 0) * na) / max(1, nb + na), 4),
             "low_hours": round(sum(speakers[s]["minutes"] for s in low) / 60, 1)}
    return {"name": name, "created": now(), "base": base["name"], "parts": [base["name"], add["name"]],
            "rules": {**rules, add["name"]: add["rules"]}, "stats": stats, "speakers": speakers,
            "clips": base["clips"] + extra, "rejected": sorted(set(base.get("rejected", [])) | set(add.get("rejected", []))),
            "rejected_speakers": sorted(set(base.get("rejected_speakers", [])) | set(add.get("rejected_speakers", [])))}


def preview_speakers(sel: dict) -> list[str]:
    """2 მაღალი + 2 დაბალი ხმა (ყველაზე მეტი წუთით) — ნიმუშებისთვის."""
    sp = sorted(sel["speakers"].items(), key=lambda x: -x[1]["minutes"])
    hi = [s for s, v in sp if (v.get("f0") or 0) >= 165][:2]
    lo = [s for s, v in sp if 0 < (v.get("f0") or 0) < 165][:2]
    return hi + lo


def check_new(a) -> dict:
    """ვალიდაცია (პანელიც იძახებს, სანამ ფონურ პროცესს გაუშვებს). აბრუნებს გეგმას."""
    if not SLOT_RE.fullmatch(a.slot or ""):
        raise SystemExit("სახელი: base_ka_v<ნომერი> (მაგ. base_ka_v4) — ხმის შემქმნელი ასე პოულობს")
    if ((JOBS / f"{a.slot}.json").exists() or (SEL / f"{a.slot}.json").exists()) and not a.force:
        raise SystemExit(f"{a.slot} უკვე არსებობს — სხვა სახელი აირჩიეთ")
    if not 1 <= a.epochs <= 2000:
        raise SystemExit("ეპოქები: 1–2000")
    frm = None
    if a.mode == "continue":
        frm = next((m for m in models() if m["slot"] == a.from_), None)
        if not frm:
            raise SystemExit(f"გასაგრძელებელი მოდელი ვერ ვიპოვე: {a.from_} (საჭიროა დასრულებული წვრთნა და გაშვებული შერჩევა)")
    if a.selection:
        s = selection(a.selection)
        if not s:
            raise SystemExit(f"შერჩევა ვერ ვიპოვე: {a.selection}")
        if s.get("launched"):
            raise SystemExit(f"„{a.selection}“ უკვე გამოყენებულია სხვა მოდელში")
    elif not quality_ready():
        raise SystemExit("ახალი შერჩევისთვის ჯერ საჭიროა ხარისხის შემოწმება („მონაცემები → ხარისხი“, ./aivoice quality)")
    elif not 1 <= a.hours <= 400:
        raise SystemExit("საათები: 1–400")
    max_epochs = (frm["epoch"] + a.epochs) if frm else a.epochs
    clips = (selection(a.selection)["stats"]["clips"] if a.selection else int(a.hours * CLIPS_PER_HOUR))
    clips += frm["clips"] if frm else 0
    return {"from": frm, "max_epochs": max_epochs, "clips_estimate": clips, "estimate": estimate(clips, a.epochs)}


def cmd_new(a):
    plan = check_new(a)
    frm = plan["from"]
    add_name = a.selection or (f"{a.slot}_add" if frm else a.slot)
    sel_cmd = None if a.selection else [PY, "tools/select_best.py", "--hours", str(a.hours), "--name", add_name, *select_flags(a)]
    if sel_cmd and not frm:
        sel_cmd.append("--include-launched")  # ნულიდან — საუკეთესო ჩანაწერები, უკვე გამოყენებულების ჩათვლით
    print(f"{a.slot}: {'გაგრძელება ' + frm['slot'] + ' (ეპოქა ' + str(frm['epoch']) + ')' if frm else 'ნულიდან (LJSpeech)'}"
          f" → max_epochs {plan['max_epochs']}")
    print(f"  შერჩევა: {' '.join(sel_cmd) if sel_cmd else 'არსებული ' + a.selection}"
          f"{' + შერწყმა ' + frm['selection'] + '-თან (base, speaker_order)' if frm else ''}")
    e = plan["estimate"]
    print(f"  ≈ {plan['clips_estimate']} ჩანაწერი · {e['batches_per_epoch']} ბატჩი/ეპოქა · Google Cloud L4: ≈ {e['hours']} სთ, ≈ ${e['usd']}")
    if a.dry_run:
        print("(--dry-run: არაფერი შექმნილა)")
        return
    prev_job = read_json(JOBS / f"{frm['slot']}.json", {}) if frm else {}
    j = {"_note": f"საბაზო მოდელი {a.slot}: " + (f"{frm['slot']}-ის გაგრძელება (ეპოქა {frm['epoch']} →) + ახალი ჩანაწერები"
                                                 if frm else "ნულიდან (LJSpeech → Common Voice)") + " — base_ctl.py",
         "dataset_dir": str(cpath("datasets") / a.slot), "num_speakers": NUM_SPEAKERS, "batch": GCP_BATCH, "devices": 1,
         "max_epochs": plan["max_epochs"]}
    j.update({"resume_from": frm["checkpoint"]} if frm else {"warmstart": LJSPEECH.name})
    j.update({"audio_from_source": True, "trim_silence": False, "preview_speakers": prev_job.get("preview_speakers", []),
              "base_meta": {"mode": a.mode, "from": frm and frm["slot"], "from_epoch": frm and frm["epoch"], "epochs": a.epochs,
                            "hours": None if a.selection else a.hours, "selection": add_name, "target": a.on,
                            "flags": select_flags(a), "created": now()}})
    write_json(JOBS / f"{a.slot}.json", j)
    ctl(a.slot, stage="selecting", target=None, note="ჩანაწერების შერჩევა…", error=None)
    if sel_cmd and run_step(a.slot, "შერჩევა", sel_cmd) != 0:
        ctl(a.slot, stage="failed", note="შერჩევა ჩავარდა — იხ. ლოგი", error="select_best")
        raise SystemExit("✗ შერჩევა ჩავარდა")
    add = selection(add_name)
    if frm:
        sel = merge(launched_selection(frm["slot"]), add, a.slot)
    else:
        sel = {**add, "name": a.slot, "base": None}
        if add_name != a.slot:
            sel["source_selection"] = add_name
    write_json(SEL / f"{a.slot}.json", sel)
    if not j["preview_speakers"]:
        j["preview_speakers"] = preview_speakers(sel)
        write_json(JOBS / f"{a.slot}.json", j)
    s = sel["stats"]
    short = "" if a.selection or add["stats"]["hours"] >= a.hours * 0.9 else \
        f" (ახალი მხოლოდ {add['stats']['hours']} სთ — მეტი შესაფერისი ჩანაწერი აღარ არის)"
    ctl(a.slot, stage="new", note=f"შერჩეულია: {s['hours']} სთ, {s['clips']} ჩანაწერი, {s['speakers']} ხმა{short} — შემდეგი: მომზადება")
    print(f"✓ {a.slot}: {json.dumps(s, ensure_ascii=False)}\n  შემდეგი: ./aivoice base prepare {a.slot}")


# ---------------- prepare ----------------
def build_cmd(slot, tar: bool) -> list:
    return ["nice", "-n", "10", PY, "tools/build_selection_dataset.py", slot, *(["--tar"] if tar else [])]


def cmd_prepare(a):
    j = job(a.slot)
    if not selection(a.slot):
        raise SystemExit(f"შერჩევა ჯერ არ არის (data/selections/{a.slot}.json)")
    tar = a.tar or (j.get("base_meta") or {}).get("target", "gcp") == "gcp"
    cmd = build_cmd(a.slot, tar)
    print(f"{a.slot}: dataset → {j['dataset_dir']}{' + ' + str(cpath('gcp') / (a.slot + '.tar')) if tar else ''}\n  {' '.join(map(str, cmd))}")
    if a.dry_run:
        print("(--dry-run)")
        return
    ctl(a.slot, stage="prepare", note="dataset მზადდება (აუდიოს დამუშავება — 1–3 სთ)", error=None)
    if run_step(a.slot, "dataset", cmd) != 0:
        ctl(a.slot, stage="failed", note="dataset-ის აწყობა ჩავარდა — იხ. ლოგი", error="build_selection_dataset")
        raise SystemExit("✗ dataset ჩავარდა")
    n = sum(1 for _ in (Path(j["dataset_dir"]) / "metadata.csv").open(encoding="utf-8"))
    ctl(a.slot, stage="prepared", note=f"მზადაა: {n} ჩანაწერი — შემდეგი: გაშვება")
    print(f"✓ {n} ჩანაწერი. შემდეგი: ./aivoice base start {a.slot} --on gcp")


# ---------------- start ----------------
def speaker_order(ds: Path) -> list[str]:
    order, seen = [], set()
    for line in (ds / "metadata.csv").open(encoding="utf-8"):
        s = line.split("|")[1] if line.count("|") >= 2 else None
        if s and s not in seen:
            seen.add(s)
            order.append(s)
    return order


def check_order(slot, j) -> list[str]:
    """გაგრძელებისას წინა მოდელის დინამიკები იმავე რიგით პირველები უნდა იყვნენ — თორემ ხმები აირევა."""
    ds = Path(j["dataset_dir"])
    if not (ds / "metadata.csv").exists():
        raise SystemExit(f"dataset ჯერ არ არის — ./aivoice base prepare {slot}")
    order = speaker_order(ds)
    frm = (j.get("base_meta") or {}).get("from")
    if frm:
        prev = launched_selection(frm)["speaker_order"]
        if order[:len(prev)] != prev:
            bad = next(i for i, y in enumerate(prev) if i >= len(order) or order[i] != y)
            raise SystemExit(f"დინამიკების რიგი არ ემთხვევა {frm}-ს (პოზიცია {bad}: {prev[bad]}) — "
                             "ალბათ წინა მოდელის ხმა ამოღებულია; გაგრძელება შეუძლებელია")
    if len(order) > j["num_speakers"]:
        raise SystemExit(f"{len(order)} დინამიკი > num_speakers {j['num_speakers']}")
    return order


def mark_launched(slot, j, order, where):
    sel = selection(slot)
    if sel.get("launched"):
        return
    sel.update(launched=now(), model=slot, dataset=j["dataset_dir"], speaker_order=order,
               _note=f"გაშვებული ({where}{', ' + sel['base'] + '-ის გაგრძელება' if sel.get('base') else ', ნულიდან'}). "
                     "speaker_order = Piper-ის ID-ების რიგი; შემდეგი შერჩევა ამ ჩანაწერებს გამოტოვებს.")
    write_json(SEL / f"{slot}.json", sel)


def check_start(slot, on) -> dict:
    """გაშვებამდე შემოწმება (პანელიც იძახებს). აბრუნებს {batch, order}."""
    j = job(slot)
    order = check_order(slot, j)
    if on == "gcp":
        g = gcp_state()
        if not g["configured"]:
            raise SystemExit("Google Cloud არ არის მორგებული (gcloud) — „პარამეტრები“ / README")
        if g["busy"] and g["slot"] != slot:
            raise SystemExit(f"Google Cloud დაკავებულია: {g['slot']} ({g['phase']}) — ერთდროულად მხოლოდ ერთი წვრთნა")
        return {"batch": GCP_BATCH, "order": order}
    g = gpu_info()
    if not g:
        raise SystemExit("ამ კომპიუტერზე NVIDIA GPU ვერ ვიპოვე — გამოიყენეთ Google Cloud")
    if not g["ok"]:
        raise SystemExit(g["reason"])
    if not TPY.exists():
        raise SystemExit("საჭიროა წვრთნის გარემო: ./install.sh --training")
    return {"batch": g["batch"], "order": order, "gpu": g}


def gcp(*args) -> subprocess.CompletedProcess:
    print("  gcp", *args, flush=True)
    return subprocess.run([PY, "tools/gcp_train.py", *args], cwd=ROOT, capture_output=True, text=True)


def restart_gcp_watcher(slot):
    """aivoice-gcp სერვისი (თუ არის) — თავიდან, რომ ახალი slot წაიკითხოს; თუ არა — ცალკე პროცესი."""
    r = subprocess.run(["sudo", "-n", "systemctl", "restart", "aivoice-gcp"], capture_output=True, text=True)
    if r.returncode == 0:
        return "aivoice-gcp"
    log = (run_dir(slot) / "gcp_watch.log").open("a", encoding="utf-8")
    p = subprocess.Popen([PY, "tools/gcp_train.py", "watch"], cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                         stdin=subprocess.DEVNULL, start_new_session=True, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    ctl(slot, watcher_pid=p.pid)
    return f"pid {p.pid}"


def start_gcp(slot, j, order):
    """launch_v3.sh-ის მსგავსად: შექმნა (on-demand) → ატვირთვა (ატვირთული გამოტოვდება) → მეთვალყურე. 6 ცდა."""
    import kaggle_train as kt
    mark_launched(slot, j, order, "Google Cloud")
    ctl(slot, stage="uploading", target="gcp", note="Google Cloud: მანქანის შექმნა და ატვირთვა…", error=None, started=now())
    for attempt in range(1, 7):
        print(f"== {now()} ცდა {attempt}", flush=True)
        g = gcp_state()
        mine = g["slot"] == slot and g["phase"] in GCP_ACTIVE
        st = gcp("status").stdout.splitlines()[:1] if mine else ["არ არსებობს"]
        st = st[0] if st else ""
        print("  ", st, flush=True)
        if mine and re.search(r"TERMINATED|STOPPED|SUSPENDED", st):
            if gcp("start").returncode != 0:
                gcp("delete")
                if gcp("create", slot).returncode != 0:
                    time.sleep(300)
                    continue
        elif not mine or "არ არსებობს" in st:
            r = gcp("create", slot)
            print(r.stdout[-500:], r.stderr[-500:], flush=True)
            if r.returncode != 0:
                ctl(slot, note=f"L4 ახლა არ არის — ვცდი თავიდან ({attempt}/6)")
                time.sleep(300)
                continue
        ctl(slot, note="Google Cloud: ატვირთვა და მომზადება (~30–60 წთ)…")
        r = gcp("push")
        print(r.stdout[-1500:], r.stderr[-800:], flush=True)
        if r.returncode == 0:
            w = restart_gcp_watcher(slot)
            kt.save_status(slot, status="training", device="gcp", epoch=(j.get("base_meta") or {}).get("from_epoch"),
                           kaggle_state="Google Cloud L4 · იწყება")
            ctl(slot, stage="training", note=f"Google Cloud L4-ზე იწვრთნება (მეთვალყურე: {w})")
            print("✓ გაეშვა", flush=True)
            return
        time.sleep(60)
    ctl(slot, stage="failed", note="Google Cloud-ზე ვერ გაეშვა (6 ცდა) — იხ. ლოგი", error="gcp")
    raise SystemExit("✗ ვერ გაეშვა")


def cmd_start(a):
    j = job(a.slot)
    if a.max_epochs:
        j["max_epochs"] = a.max_epochs
        if not a.dry_run:
            write_json(JOBS / f"{a.slot}.json", j)
    c = check_start(a.slot, a.on)
    tar = cpath("gcp") / f"{Path(j['dataset_dir']).name}.tar"
    if a.on == "gcp":
        print(f"{a.slot} → Google Cloud L4 (on-demand): {'' if tar.exists() else 'არქივი (build --tar) → '}"
              f"create {a.slot} → push → aivoice-gcp; max_epochs {j['max_epochs']}, "
              f"{'გაგრძელება ' + j['resume_from'] if j.get('resume_from') else 'ნულიდან (LJSpeech)'}; {len(c['order'])} დინამიკი")
    else:
        print(f"{a.slot} → ეს კომპიუტერი: {c['gpu']['name']} ({c['gpu']['vram_gb']} GB), batch {c['batch']}, max_epochs {j['max_epochs']}")
    if a.dry_run:
        print("(--dry-run: მანქანა/წვრთნა არ გაშვებულა, შერჩევა „launched“ არ მონიშნულა)")
        return
    if a.on == "gcp":
        if not tar.exists():
            ctl(a.slot, stage="prepare", note="Google Cloud-ისთვის არქივი მზადდება…", error=None)
            if run_step(a.slot, "არქივი", build_cmd(a.slot, True)) != 0:
                ctl(a.slot, stage="failed", note="არქივი ვერ მომზადდა — იხ. ლოგი", error="tar")
                raise SystemExit("✗ არქივი")
        start_gcp(a.slot, j, c["order"])
    else:
        mark_launched(a.slot, j, c["order"], "ეს კომპიუტერი, " + c["gpu"]["name"])
        run_local(a.slot, j, c["batch"])


# ---------------- ლოკალური GPU ----------------
def gpu_dir(slot) -> Path:
    return run_dir(slot) / "gpu"


def best_ckpt(paths: list[Path]) -> Path | None:
    """მეტი global_step = უახლესი; დაზიანებული გამოტოვდება (torch — წვრთნის გარემოში)."""
    paths = [str(p) for p in paths if p.exists()]
    if not paths:
        return None
    code = ("import sys, torch\nb = None\nfor p in sys.argv[1:]:\n    try:\n"
            "        s = torch.load(p, map_location='cpu', weights_only=False, mmap=True)['global_step']\n"
            "    except Exception:\n        continue\n    b = max(b or (s, p), (s, p))\nprint(b[1] if b else '')")
    r = subprocess.run([str(TPY), "-c", code, *paths], capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""})
    out = r.stdout.strip().splitlines()
    return Path(out[-1]) if out and out[-1] else None


def run_local(slot, j, batch):
    """წვრთნა ამ კომპიუტერის GPU-ზე (gcp/train_vm.sh-ის მსგავსად) — გრძელდება უახლესი checkpoint-იდან; ბოლოს finalize."""
    import kaggle_train as kt
    d = gpu_dir(slot)
    d.mkdir(parents=True, exist_ok=True)
    state_p = d / "state.json"
    state = read_json(state_p, {}) or {}

    def save(**kw):
        state.update(kw, slot=slot, updated=now())
        write_json(state_p, state)

    meta = d / "metadata.csv"
    meta.write_text((Path(j["dataset_dir"]) / "metadata.csv").read_text(encoding="utf-8").replace('"', ""), encoding="utf-8")
    ckpt = best_ckpt(list(d.glob("lightning_logs/version_*/checkpoints/*.ckpt")))
    if ckpt:
        start = ["--ckpt_path", str(ckpt)]
    elif j.get("resume_from"):
        start = ["--ckpt_path", j["resume_from"]]
    else:
        start = ["--model.warmstart_ckpt", str(LJSPEECH)]
    workers = max(1, min(8, (os.cpu_count() or 4) - 2))
    cache = Path.home() / ".cache" / "aivoice" / f"{slot}_gpu"  # ბევრი პატარა ფაილი — Linux-ის დისკზე
    cmd = [str(TPY), "-m", "piper.train", "fit", "--data.voice_name", slot, "--data.csv_path", str(meta),
           "--data.audio_dir", str(Path(j["dataset_dir"]) / "audio"), "--model.sample_rate", "22050",
           "--data.espeak_voice", "ka", "--data.cache_dir", str(cache), "--data.config_path", str(d / "config.json"),
           "--data.batch_size", str(batch), "--data.num_workers", str(workers), "--data.validation_split", "0.01",
           "--data.num_test_examples", "2", "--data.trim_silence", "false", "--model.mos_metric", "none",
           "--model.num_speakers", str(j["num_speakers"]), "--trainer.accelerator", "gpu", "--trainer.devices", "1",
           "--trainer.max_epochs", str(j["max_epochs"]), "--trainer.default_root_dir", str(d), *start]
    env = {**os.environ, "PIPER_AUDIO_FROM_SOURCE": "1", "PIPER_CKPT_EVERY_STEPS": "2000", "PYTHONUNBUFFERED": "1"}
    log = (d / "train.log").open("a", encoding="utf-8")
    log.write(f"\n==== {now()} {' '.join(start)} (batch {batch})\n")
    log.flush()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, cwd=ROOT / "training")
    stopping = []

    def on_term(*_):
        stopping.append(1)
        try:
            proc.terminate()
        except OSError:
            pass
    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)
    save(running=True, runner_pid=os.getpid(), pid=proc.pid, started=now(), resumed_from=start[-1], batch=batch,
         error=None, done=False)
    ctl(slot, stage="training", target="gpu", note=f"ამ კომპიუტერის GPU-ზე იწვრთნება (batch {batch})", error=None, started=now())
    last, buf = 0.0, b""
    while True:
        chunk = proc.stdout.read1(65536) if hasattr(proc.stdout, "read1") else proc.stdout.read(65536)
        if not chunk:
            break
        buf += chunk
        *lines, buf = re.split(rb"[\r\n]", buf)
        for raw in lines:
            s = raw.decode("utf-8", "replace")
            m = PROGRESS.search(s)
            if m:
                ep, _, i, n, _, _, rate, unit = m.groups()
                if time.time() - last > 30:
                    sec = float(rate) if unit == "s/it" else 1 / max(float(rate), 1e-9)
                    save(epoch=int(ep), step_in_epoch=int(i), steps_per_epoch=int(n), sec_per_step=round(sec, 3),
                         epoch_hours=round(int(n) * sec / 3600, 2))
                    kt.save_status(slot, status="training", device="gpu", epoch=int(ep),
                                   kaggle_state=f"ეს კომპიუტერი (GPU) · ეპოქა {int(ep) + 1}, ბატჩი {i}/{n}")
                    last = time.time()
            elif s.strip():
                log.write(s + "\n")
                log.flush()
    rc = proc.wait()
    log.write(f"==== {now()} exit {rc}\n")
    log.close()
    if stopping:
        save(running=False, pid=None, note="გაჩერდა (შემდეგ ჯერზე იქიდან გააგრძელებს)")
        ctl(slot, stage="stopped", note="გაჩერებულია — „გაშვება“ უახლესი checkpoint-იდან გააგრძელებს")
        return
    if rc != 0:
        save(running=False, pid=None, error=f"exit {rc}")
        ctl(slot, stage="failed", note=f"წვრთნა შეწყდა (exit {rc}) — იხ. gpu/train.log", error=f"exit {rc}")
        raise SystemExit(f"✗ exit {rc}")
    save(running=False, pid=None, done=True, note="დასრულდა (max_epochs)")
    finalize(slot)


def finalize(slot, dry=False):
    """gcp_train.finalize-ის მსგავსად (ლოკალური GPU): final.ckpt, ONNX, ხმის შემქმნელი, ნიმუშები, სქესი/ასაკი, რეიტინგი."""
    import kaggle_train as kt
    d, out = gpu_dir(slot), run_dir(slot) / "kaggle_out"
    steps = ["final.ckpt + config.json → kaggle_out/", f"onnx → kaggle_out/{slot}.onnx",
             f"designer → {cpath('designer').parent / ('designer_' + slot)}", "traits", "previews", "mms_rank (ფონზე)"]
    if dry:
        print("finalize:", " · ".join(steps))
        return
    ctl(slot, stage="training", note="სრულდება: ONNX, ხმის შემქმნელი, ნიმუშები…")
    ckpt = best_ckpt(list(d.glob("lightning_logs/version_*/checkpoints/*.ckpt")))
    if not ckpt:
        raise SystemExit("checkpoint ვერ ვიპოვე")
    out.mkdir(parents=True, exist_ok=True)
    if (out / "final.ckpt").exists():
        (out / "final.ckpt").replace(out / "final.prev.ckpt")
    shutil.copy2(ckpt, out / "final.ckpt")
    shutil.copy2(d / "config.json", out / "config.json")
    shutil.copy2(out / "config.json", out / f"{slot}.onnx.json")
    epoch = read_json(d / "state.json", {}).get("epoch")
    st = read_json(out / "status.json", {}) or {}
    st.update(status="done", epoch=epoch, source="local", gpu=(gpu_info() or {}).get("name"))
    write_json(out / "status.json", st)
    env = {"CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4"}
    run_step(slot, "onnx", [TPY, "-m", "piper.train.export_onnx", "--checkpoint", out / "final.ckpt",
                            "--output-file", out / f"{slot}.onnx"], env)
    run_step(slot, "ხმის შემქმნელი", [TPY, "tools/export_designer.py", out / "final.ckpt", "--out",
                                       cpath("designer").parent / f"designer_{slot}", "--config", out / "config.json",
                                       "--source", "local"], env)
    run_step(slot, "სქესი/ასაკი", [TPY, "tools/voice_traits.py", "--model", slot], env)
    run_step(slot, "ნიმუშები", [TPY, "tools/make_previews.py", out / "final.ckpt", "--label", "საბოლოო", "--slot", slot], env)
    rank = (cpath("runs") / "_rank.log").open("a", encoding="utf-8")
    subprocess.Popen(["nice", "-n", "10", str(TPY), "tools/mms_rank.py", "--model", slot], cwd=ROOT, stdout=rank,
                     stderr=subprocess.STDOUT, start_new_session=True, env={**os.environ, **env, "OMP_NUM_THREADS": "8"})
    kt.save_status(slot, status="done", epoch=epoch, device="gpu", finished=now())
    ctl(slot, stage="done", note="დასრულდა — „ხმის შემქმნელში“ ახალი მოდელია (რეიტინგი ფონზე ითვლება)", finished=now())
    print("✓ დასრულდა", flush=True)


# ---------------- status / stop ----------------
def cmd_status(a):
    slots = [a.slot] if a.slot else [p.stem for p in sorted(JOBS.glob("*.json")) if (read_json(p, {}).get("num_speakers") or 1) > 1]
    g = gcp_state()
    for s in slots:
        c = ctl(s)
        st = read_json(run_dir(s) / "status.json", {}) or {}
        loc = read_json(gpu_dir(s) / "state.json", {}) or {}
        ep = loc.get("epoch") if c.get("target") == "gpu" else st.get("epoch")
        print(f"■ {s}: {c.get('stage') or st.get('status') or '—'} · {c.get('target') or ''} · ეპოქა "
              f"{(ep or 0) + 1}/{read_json(JOBS / f'{s}.json', {}).get('max_epochs')}"
              f"{' · ' + c['note'] if c.get('note') else ''}{' · შეცდომა: ' + str(c['error']) if c.get('error') else ''}")
        if g["slot"] == s:
            print(f"  Google Cloud: {g['phase']}{' · ' + str(g['error']) if g.get('error') else ''}")


def kill_group(pid) -> bool:
    if not pid or int(pid) in (os.getpid(), os.getpgid(0)):
        return False
    try:
        os.killpg(int(pid), signal.SIGTERM)
        return True
    except (OSError, ValueError):
        try:
            os.kill(int(pid), signal.SIGTERM)
            return True
        except (OSError, ValueError):
            return False


def cmd_stop(a):
    job(a.slot)
    c = ctl(a.slot)
    g = gcp_state()
    if c.get("target") == "gcp" or (g["slot"] == a.slot and g["busy"]):
        print("Google Cloud: მანქანის გაჩერება (დისკი ინახება; „გაშვება“ — გაგრძელება)")
        if a.dry_run:
            return
        if g["slot"] == a.slot and g["phase"] in GCP_ACTIVE:
            r = gcp("stop")
            print(r.stdout[-300:], r.stderr[-300:])
        ctl(a.slot, stage="stopped", note="Google Cloud-ის მანქანა გაჩერებულია (დისკი ინახება — „გაშვება“ გააგრძელებს)")
    elif c.get("target") == "gpu":
        loc = read_json(gpu_dir(a.slot) / "state.json", {}) or {}
        print("ლოკალური GPU წვრთნის გაჩერება", loc.get("runner_pid"))
        if a.dry_run:
            return
        kill_group(loc.get("runner_pid")) or kill_group(loc.get("pid"))
        ctl(a.slot, stage="stopped", note="გაჩერებულია — „გაშვება“ უახლესი checkpoint-იდან გააგრძელებს")
    else:
        print("ამ მოდელს ახლა არაფერი ეშვება")


def cmd_finalize(a):
    job(a.slot)
    finalize(a.slot, a.dry_run)


def cmd_options(_):
    print(json.dumps(options(), ensure_ascii=False, indent=1))


def parser():
    ap = argparse.ArgumentParser(prog="./aivoice base")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("options")
    p = sub.add_parser("new")
    p.add_argument("slot")
    p.add_argument("--mode", choices=["scratch", "continue"], required=True)
    p.add_argument("--from", dest="from_", help="გასაგრძელებელი მოდელი (მაგ. base_ka_v3)")
    p.add_argument("--hours", type=float, default=40)
    p.add_argument("--short", action="store_true", help="მოკლე ფრაზებიც (≥ 6 ასო, ≥ 1 წმ, 6–18 ასო/წმ)")
    p.add_argument("--punct", action="store_true", help="კითხვა/ძახილი — უპირატესობა")
    p.add_argument("--clusters", action="store_true", help="რ ღ ბ ფ პ + თანხმოვანი — უპირატესობა")
    p.add_argument("--selection", help="არსებული (გამოუყენებელი) შერჩევა ახალის ნაცვლად")
    p.add_argument("--epochs", type=int, help="ნულიდან: სულ (160); გაგრძელება: დამატებით (90)")
    p.add_argument("--on", choices=["gcp", "local"], default="gcp", help="სად იწვრთნება (prepare-ს არქივისთვის)")
    p.add_argument("--force", action="store_true")
    p = sub.add_parser("prepare")
    p.add_argument("slot")
    p.add_argument("--tar", action="store_true")
    p = sub.add_parser("start")
    p.add_argument("slot")
    p.add_argument("--on", choices=["gcp", "local"], default="gcp")
    p.add_argument("--max-epochs", type=int)
    sub.add_parser("status").add_argument("slot", nargs="?")
    for c in ("stop", "finalize"):
        sub.add_parser(c).add_argument("slot")
    for name, sp in sub.choices.items():
        if name not in ("options", "status"):
            sp.add_argument("--dry-run", action="store_true", help="მხოლოდ აჩვენოს, რა მოხდებოდა")
    return ap


def main():
    a = parser().parse_args()
    if a.cmd == "new" and a.epochs is None:
        a.epochs = 160 if a.mode == "scratch" else 90
    {"options": cmd_options, "new": cmd_new, "prepare": cmd_prepare, "start": cmd_start, "status": cmd_status,
     "stop": cmd_stop, "finalize": cmd_finalize}[a.cmd](a)


if __name__ == "__main__":
    main()
