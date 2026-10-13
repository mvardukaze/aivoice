"""წვრთნა Kaggle-ის უფასო GPU-ზე (T4 ×2), ამ კომპიუტერიდან მართვით.

გამოყენება (პროექტის .venv-ით — kaggle CLI იქ არის):
    .venv/bin/python tools/kaggle_train.py base            # wheel + core.pyx + საწყისი checkpoint-ები → aivoice-base
    .venv/bin/python tools/kaggle_train.py push <slot>     # dataset (ერთხელ) + resume (თუ არის) + წვრთნის გაშვება
    .venv/bin/python tools/kaggle_train.py status <slot>
    .venv/bin/python tools/kaggle_train.py fetch <slot>    # შედეგი → training/runs/<slot>/, ნიმუშები პანელზე
    .venv/bin/python tools/kaggle_train.py watch           # სერვისი (aivoice-kaggle): ყოველ 10 წთ-ში სტატუსი,
                                                           # დასრულებისას fetch, 12 სთ-ის ლიმიტზე — შემდეგი სესია

წვრთნის პარამეტრები: data/kaggle_jobs/<slot>.json (თუ არ არის — ერთი ხმა datasets/<slot>-იდან, ნათიადან).
ყველაფერი პირადია (is_private). ავტორიზაცია: ~/.kaggle/access_token.

კვოტა (30 GPU-სთ/კვირა): ახალი სესია მხოლოდ მაშინ იწყება, თუ ბოლო 7 დღის სესიები + 11 სთ + მარაგი
ლიმიტში ეტევა (მოძრავი ფანჯარა — Kaggle-ის აღდგენის დღე ზუსტად არ ვიცით, ასე ნებისმიერ შემთხვევაში
სწორია). სანამ კვოტა არ ჰყოფნის, წვრთნა ლოკალურად (CPU) გრძელდება — tools/local_train.py.
"""

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.config import api_url, cfg, expand, path as cpath  # noqa: E402,F401
KAGGLE = ROOT / ".venv" / "bin" / "kaggle"
WORK = cpath("work")  # ატვირთვის დროებითი ფაილები
BASE_DIR = WORK / "base"
# wheel + core.pyx + LJSpeech checkpoint. v2 — ნათიას checkpoint-ის გარეშე (ძველი „aivoice-base“ წაიშალა 2026-10-13)
BASE_SLUG = cfg()["kaggle"]["base_dataset"]
TEMPLATE = ROOT / "kaggle" / "kernel_train.py"
JOBS = ROOT / "data" / "kaggle_jobs"

sys.path.insert(0, str(ROOT / "tools"))
from common import MAX_EPOCHS, MIN_EPOCHS, PHRASES, TARGET_SAMPLES  # noqa: E402

_user = None
KAGGLE_WEEK_H, SESSION_H, QUOTA_MARGIN_H = 30, 11.5, 3  # მარაგი: ჩავარდნილი გაშვებებიც ხარჯავს, მაგრამ არ ითვლება
LOCAL_CTRL = ROOT / "data" / "local_train.json"
GCP_STATE = cpath("gcp") / "state.json"
TRAIN_PY = ROOT / "training" / ".venv" / "bin" / "python"


def user() -> str:
    global _user
    if _user is None:
        out = subprocess.run([str(KAGGLE), "config", "view"], capture_output=True, text=True).stdout
        _user = next(l.split(":", 1)[1].strip() for l in out.splitlines() if "username" in l)
    return _user


def kaggle(*args, check=True):
    print("+ kaggle", " ".join(args), flush=True)
    r = subprocess.run([str(KAGGLE), *args], capture_output=True, text=True)
    tail = "\n".join(l for l in (r.stdout + r.stderr).splitlines() if "%|" not in l)[-1500:]
    print(tail, flush=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"kaggle {' '.join(args[:2])}: {tail[-300:]}")
    return r


def upload_dataset(path: Path, slug: str, title: str):
    u = user()
    (path / "dataset-metadata.json").write_text(json.dumps(
        {"title": title, "id": f"{u}/{slug}", "licenses": [{"name": "other"}]}, ensure_ascii=False))
    # "datasets status"/"files" ახალ dataset-ზე 403-ს აბრუნებს — არსებობას სიით ვამოწმებთ
    exists = slug in kaggle("datasets", "list", "--mine", "-s", slug, check=False).stdout
    if exists:
        kaggle("datasets", "version", "-p", str(path), "-m", time.strftime("%Y-%m-%d %H:%M"), "-r", "zip")
    else:
        kaggle("datasets", "create", "-p", str(path), "-r", "zip")  # ნაგულისხმევად პირადია
    return f"{u}/{slug}"


def wait_dataset_ready(ref: str, expect: str | None = None, timeout=3600):
    """ახალი dataset/ვერსია Kaggle-ზე ჯერ მუშავდება (ამ დროს "files" 403-ს აბრუნებს) —
    kernel-ს მხოლოდ მზა dataset-ს ვაბამთ. expect: ფაილი, რომელიც სიაში უნდა ჩანდეს."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = subprocess.run([str(KAGGLE), "datasets", "files", ref], capture_output=True, text=True)
        if r.returncode == 0:  # სია გვერდებადაა დაყოფილი — expect შეიძლება პირველ გვერდზე არ ჩანდეს
            return True
        time.sleep(30)
    raise RuntimeError(f"{ref}: dataset {timeout // 60} წუთში არ დამუშავდა")


def slug(slot):
    return slot.replace("_", "-")


def run_dir(slot):
    d = cpath("runs") / slot
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_status(slot, **kw):
    p = run_dir(slot) / "status.json"
    st = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    st.update(kw, voice=slot, updated=time.strftime("%Y-%m-%dT%H:%M:%S"))
    p.write_text(json.dumps(st, ensure_ascii=False, indent=1), encoding="utf-8")


def job(slot) -> dict:
    p = JOBS / f"{slot}.json"
    j = json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}
    j.setdefault("dataset_dir", None)  # None ⇒ datasets/<slot> (wav → flac კონვერტაცია)
    j.setdefault("num_speakers", 1)
    j.setdefault("batch", 24)
    j.setdefault("devices", 1)
    j.setdefault("warmstart", "ljspeech-medium.clean.ckpt")
    j.setdefault("audio_from_source", False)
    j.setdefault("trim_silence", True)
    j.setdefault("preview_speakers", [])
    return j


def cmd_base(_):
    """Kaggle-ის საბაზო dataset: piper_overlay.tar (ჩვენი შესწორებები) + LJSpeech checkpoint (საწყისი წერტილი)."""
    import tarfile
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    for old in BASE_DIR.glob("piper_tts-*.whl"):
        old.unlink()
    with tarfile.open(BASE_DIR / "piper_overlay.tar", "w") as t:
        t.add(ROOT / "piper_overlay" / "piper", arcname="piper")
    lj = cpath("runs").parent / "checkpoints" / "ljspeech-medium.clean.ckpt"
    for c in [ROOT / "training" / "checkpoints" / "en_US-ljspeech-medium" / "ljspeech-medium.clean.ckpt", lj]:
        if c.exists() and not (BASE_DIR / c.name).exists():
            shutil.copy2(c, BASE_DIR / c.name)
    if not (BASE_DIR / "ljspeech-medium.clean.ckpt").exists():
        raise SystemExit("ljspeech-medium.clean.ckpt ვერ ვიპოვე — ჯერ: ./aivoice models pull")
    upload_dataset(BASE_DIR, BASE_SLUG, "aivoice base")


def flac(args):
    src, dst = args
    if not dst.exists():
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-c:a", "flac", str(dst)], check=True)


def prepare_voice_dataset(slot, j) -> tuple[Path, int]:
    """ატვირთვისთვის მზა საქაღალდე: metadata.csv + audio/*.flac"""
    if j["dataset_dir"]:
        d = Path(j["dataset_dir"])
        return d, sum(1 for _ in (d / "metadata.csv").open(encoding="utf-8"))
    ds = Path(j.get("source_dir") or ROOT / "datasets" / slot)  # wav/ + metadata.csv (მაგ. make_voice_dataset.py)
    d = WORK / f"voice-{slot}"
    (d / "audio").mkdir(parents=True, exist_ok=True)
    lines = (ds / "metadata.csv").read_text(encoding="utf-8").splitlines()
    jobs = [(ds / "wav" / l.split("|", 1)[0], d / "audio" / l.split("|", 1)[0].replace(".wav", ".flac")) for l in lines]
    with ThreadPoolExecutor(12) as ex:
        list(ex.map(flac, jobs))
    (d / "metadata.csv").write_text("\n".join(l.replace(".wav|", ".flac|", 1) for l in lines) + "\n", encoding="utf-8")
    return d, len(lines)


def cmd_push(a):
    slot = a.slot
    j = job(slot)
    u = user()
    d, clips = prepare_voice_dataset(slot, j)

    # დიდი dataset-ი მხოლოდ მაშინ იტვირთება, როცა metadata შეიცვალა
    digest = hashlib.sha1((d / "metadata.csv").read_bytes()).hexdigest()
    marker = run_dir(slot) / "kaggle_dataset.sha1"
    vds = f"{u}/aivoice-{slug(slot)}"
    if not marker.exists() or marker.read_text().strip() != digest:
        upload_dataset(d, f"aivoice-{slug(slot)}", f"aivoice {slot}")
        marker.write_text(digest)
        wait_dataset_ready(vds, "metadata.csv")

    sources = [f"{u}/{BASE_SLUG}", vds]
    if j.get("init_dir"):  # საწყისი checkpoint ცალკე dataset-ად (fold_speaker.py) — ერთხელ იტვირთება
        islug = f"aivoice-init-{slug(slot)}"
        if kaggle("datasets", "files", f"{u}/{islug}", check=False).returncode != 0:
            upload_dataset(Path(j["init_dir"]), islug, f"aivoice init {slot}")
        wait_dataset_ready(f"{u}/{islug}")
        sources.append(f"{u}/{islug}")
    resume = run_dir(slot) / "kaggle_out" / "final.ckpt"
    if resume.exists():
        # წინა სესიის checkpoint — ყოველ ჯერზე ახალი dataset (checkpoint-ის ეპოქით). ვერსიის განახლება არ გამოდგა:
        # დაუმუშავებელი ვერსია kernel-ს უხმოდ აგდებდა (ERROR, ლოგის გარეშე, 2026-10-07). ახალი dataset-ი
        # დამუშავებამდე 403-ს აბრუნებს, ასე რომ მზადყოფნის შემოწმება სანდოა.
        ep = (json.loads((run_dir(slot) / "kaggle_out" / "status.json").read_text()).get("epoch")
              if (run_dir(slot) / "kaggle_out" / "status.json").exists() else None)
        rslug = f"aivoice-resume-{slug(slot)}-e{ep}-{int(resume.stat().st_mtime) % 10**6}"
        rd = WORK / rslug
        rd.mkdir(parents=True, exist_ok=True)
        if not (rd / "resume.ckpt").exists():
            shutil.copy2(resume, rd / "resume.ckpt")
        ref = f"{u}/{rslug}"
        if kaggle("datasets", "files", ref, check=False).returncode != 0:
            upload_dataset(rd, rslug, f"aivoice resume {slot} e{ep}")
        wait_dataset_ready(ref)
        sources.append(ref)

    max_epochs = j.get("max_epochs") or max(MIN_EPOCHS, min(MAX_EPOCHS, math.ceil(TARGET_SAMPLES / clips)))
    prev = json.loads((run_dir(slot) / "status.json").read_text()) if (run_dir(slot) / "status.json").exists() else {}
    session = 1 + sum(1 for x in prev.get("sessions", []) if x.get("status") in ("partial", "done")) if resume.exists() else 1
    cfg = {"slot": slot, "session": session, "max_epochs": max_epochs, "batch": j["batch"], "devices": j["devices"],
           "num_speakers": j["num_speakers"], "warmstart": j["warmstart"],
           "audio_from_source": j["audio_from_source"], "trim_silence": j["trim_silence"],
           "preview_speakers": j["preview_speakers"], "phrases": PHRASES}
    k = WORK / f"kernel-{slot}"
    k.mkdir(parents=True, exist_ok=True)
    (k / "train.py").write_text(TEMPLATE.read_text(encoding="utf-8").replace(
        "__CONFIG__", json.dumps(cfg, ensure_ascii=False).replace("\\", "\\\\")), encoding="utf-8")
    kid = f"{u}/aivoice-train-{slug(slot)}"
    (k / "kernel-metadata.json").write_text(json.dumps({
        "id": kid, "title": f"aivoice train {slug(slot)}", "code_file": "train.py",
        "language": "python", "kernel_type": "script", "is_private": True,
        "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": sources,
    }, ensure_ascii=False))
    kaggle("kernels", "push", "-p", str(k))
    save_status(slot, status="training", device="kaggle", batch=j["batch"], devices=j["devices"], clips=clips,
                max_epochs=max_epochs, num_speakers=j["num_speakers"], kernel=kid,
                session_started=time.strftime("%Y-%m-%dT%H:%M:%S"), error=None)


def kernel_id(slot):
    return f"{user()}/aivoice-train-{slug(slot)}"


def cmd_status(a):
    return kaggle("kernels", "status", kernel_id(a.slot), check=False).stdout.lower()


def cmd_fetch(a):
    slot = a.slot
    out = run_dir(slot) / "kaggle_out"
    tmp = run_dir(slot) / "kaggle_out.new"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    kaggle("kernels", "output", kernel_id(slot), "-p", str(tmp))
    st = json.loads((tmp / "status.json").read_text()) if (tmp / "status.json").exists() else {}
    if st.get("status") in ("done", "partial") and (tmp / "final.ckpt").exists():
        shutil.rmtree(out, ignore_errors=True)
        tmp.rename(out)  # წინა final.ckpt მხოლოდ წარმატებული ახლით ჩანაცვლდება
    else:
        out = tmp
    onnx = out / f"{slot}.onnx"
    if onnx.exists():
        prev = run_dir(slot) / "previews"
        n = len(list(prev.glob("k*"))) + 1 if prev.exists() else 1
        pdir = prev / f"k{n:02d}"
        pdir.mkdir(parents=True, exist_ok=True)
        files = []
        for w in sorted((out / "previews").glob("*.wav")):
            shutil.copy2(w, pdir / w.name)
            stem = w.stem.rsplit("_", 1)
            i = int(stem[-1]) - 1
            files.append({"file": f"{pdir.name}/{w.name}", "text": PHRASES[i] if i < len(PHRASES) else "",
                          "speaker": stem[0] if len(stem) == 2 else None})
        (pdir / "meta.json").write_text(json.dumps(
            {"label": f"Kaggle #{n}", "epoch": st.get("epoch"), "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "files": files}, ensure_ascii=False, indent=1), encoding="utf-8")
    if job(slot)["num_speakers"] > 1 and st.get("status") in ("done", "partial") and (out / "final.ckpt").exists():
        # ხმის შემქმნელი ახალ (უკეთეს) მოდელზე გადადის — ექსპორტი training venv-ით (torch)
        r = subprocess.run([str(ROOT / "training" / ".venv" / "bin" / "python"), str(ROOT / "tools" / "export_designer.py"),
                            str(out / "final.ckpt")], capture_output=True, text=True, env={"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin"})
        print("designer export:", r.stdout.strip()[-200:] or r.stderr.strip()[-300:], flush=True)
    if st.get("status") == "done" and job(slot)["num_speakers"] == 1:
        vdir = ROOT / "voices" / slot
        vdir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(onnx, vdir / onnx.name)
        shutil.copy2(str(onnx) + ".json", vdir / (onnx.name + ".json"))
        cat_p = ROOT / "voices" / "catalog.json"
        cat = json.loads(cat_p.read_text(encoding="utf-8"))
        for v in cat["voices"]:
            if v["id"] == slot:
                v.update(status="testing", model=f"{slot}/{onnx.name}")  # „სატესტო“ — შემოწმების შემდეგ „მზადაა“
        cat_p.write_text(json.dumps(cat, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    prev_st = json.loads((run_dir(slot) / "status.json").read_text())
    log = out / "train.log"
    if prev_st.get("sessions") and log.exists() and "[warmstart]" in log.read_text(encoding="utf-8", errors="replace") \
            and st.get("epoch") is not None and prev_st["sessions"][-1].get("epoch") is not None:
        # resume-ის შემდეგ warmstart-მა ნავარჯიშები წონები LJSpeech-ით გადაწერა (ძველი kernel, 2026-10-07):
        # ეს სესია თითქმის თავიდან დაწყებას უდრის — ეპოქების ლიმიტს იმდენივეს ვუმატებთ
        # (წინა ეპოქა სესიების სიიდან — status.json-ის "epoch" push-ის შემდეგ შეიძლება ცარიელი იყოს)
        lost = st["epoch"] - prev_st["sessions"][-1]["epoch"]
        jp = JOBS / f"{slot}.json"
        if lost > 0 and jp.exists():
            jj = json.loads(jp.read_text(encoding="utf-8"))
            jj["max_epochs"] = jj["max_epochs"] + lost
            jj["_warmstart_reset"] = jj.get("_warmstart_reset", []) + [{"session_epoch": st["epoch"], "added": lost}]
            jp.write_text(json.dumps(jj, ensure_ascii=False, indent=1), encoding="utf-8")
            if st.get("status") == "done":
                st["status"] = "partial"
            print("warmstart reset detected: max_epochs +", lost, flush=True)
    hours = (prev_st.get("kaggle_hours") or 0) + (st.get("hours") or 0)
    sessions = prev_st.get("sessions", []) + [{"ended": time.strftime("%Y-%m-%dT%H:%M:%S"), "hours": st.get("hours"),
                                                "epoch": st.get("epoch"), "status": st.get("status")}]
    save_status(slot, status=st.get("status", "failed"), epoch=st.get("epoch"), device="kaggle",
                kaggle_hours=round(hours, 2), error=st.get("error"), sessions=sessions)
    print("fetched:", st)


def kaggle_hours_7d() -> float:
    """ბოლო 7 დღის Kaggle-ის GPU-საათები (სესია ითვლება, თუ ფანჯარაში მოხვდა მისი ნებისმიერი ნაწილი)."""
    cutoff = time.time() - 7 * 86400
    total = 0.0
    for p in (cpath("runs")).glob("*/status.json"):
        for s in (json.loads(p.read_text(encoding="utf-8")).get("sessions") or []):
            try:
                end = time.mktime(time.strptime(s["ended"], "%Y-%m-%dT%H:%M:%S"))
            except (KeyError, ValueError):
                continue
            if end >= cutoff:
                total += s.get("hours") or 0
    return round(total, 2)


def quota_ok() -> bool:
    return kaggle_hours_7d() + SESSION_H + QUOTA_MARGIN_H <= KAGGLE_WEEK_H


def local_ctrl(slot, run: bool, reason: str):
    cur = json.loads(LOCAL_CTRL.read_text(encoding="utf-8")) if LOCAL_CTRL.exists() else {}
    if cur.get("slot") == slot and cur.get("run") == run:
        return
    LOCAL_CTRL.write_text(json.dumps({"slot": slot, "run": run, "reason": reason, "at": time.strftime("%Y-%m-%dT%H:%M:%S")},
                                     ensure_ascii=False, indent=1), encoding="utf-8")
    print("local:", slot, run, reason, flush=True)


def local_state(slot) -> dict:
    p = run_dir(slot) / "local" / "state.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def take_local_ckpt(slot) -> bool:
    """Kaggle-ზე დაბრუნებამდე: თუ ლოკალურ checkpoint-ს მეტი ნაბიჯი აქვს, ის ხდება kaggle_out/final.ckpt.
    ლოკალური წვრთნა ჯერ უნდა გაჩერდეს (False — ჯერ კიდევ მუშაობს, შემდეგ ციკლზე ვცდით)."""
    local_ctrl(slot, False, "Kaggle-ის კვოტა აღდგა — წვრთნა Kaggle-ზე გადადის")
    for _ in range(30):
        if not local_state(slot).get("running"):
            break
        time.sleep(10)
    else:
        return False
    r = subprocess.run([str(TRAIN_PY), "-c", "import sys, json; sys.path.insert(0, 'tools'); import local_train as l; "
                        f"b = l.best_ckpt({slot!r}); print(json.dumps([b[0], b[1], str(b[2])] if b else None))"],
                       capture_output=True, text=True, cwd=ROOT, env={"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin"})
    best = json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 and r.stdout.strip() else None
    out = run_dir(slot) / "kaggle_out"
    if best and Path(best[2]) != out / "final.ckpt":
        step, epoch, src = best
        new = out / "final.ckpt.new"
        new.unlink(missing_ok=True)
        shutil.copy2(src, new)
        (out / "final.ckpt").replace(out / "final.prev.ckpt")
        new.replace(out / "final.ckpt")
        stp = out / "status.json"
        st = json.loads(stp.read_text()) if stp.exists() else {}
        st.update(epoch=epoch, step=step, source="local")
        stp.write_text(json.dumps(st, ensure_ascii=False, indent=1))
        save_status(slot, epoch=epoch)
        print("local checkpoint → kaggle:", src, epoch, step, flush=True)
    return True


def cmd_watch(_):
    while True:
        for p in sorted((cpath("runs")).glob("*/status.json")):
            st = json.loads(p.read_text(encoding="utf-8"))
            if st.get("device") != "kaggle" or st.get("status") not in ("training", "partial"):
                continue
            slot, ns = st["voice"], argparse.Namespace(slot=st["voice"])
            try:
                if st["status"] == "partial":  # წინა სესიის გაგრძელება — Kaggle-ზე, ან კვოტის მოლოდინში ლოკალურად
                    gcp = json.loads(GCP_STATE.read_text()) if GCP_STATE.exists() else {}
                    if gcp.get("phase") in ("setup", "training"):  # Google Cloud-ზე მიდის (tools/gcp_train.py)
                        local_ctrl(slot, False, "წვრთნა Google Cloud-ზე მიდის")
                        save_status(slot, local=False, kaggle_state="Google Cloud L4")
                        continue
                    if quota_ok():
                        if take_local_ckpt(slot):
                            cmd_push(ns)
                            save_status(slot, local=False)
                    else:
                        local_ctrl(slot, True, "Kaggle-ის კვოტა არ ჰყოფნის — ლოკალური წვრთნა")
                        save_status(slot, local=True, kaggle_state=f"კვოტის მოლოდინი ({kaggle_hours_7d()} სთ ბოლო 7 დღეში)")
                    continue
                local_ctrl(slot, False, "Kaggle-ზე მიდის წვრთნა")
                s = cmd_status(ns)
                if "complete" in s:
                    cmd_fetch(ns)  # partial → შემდეგ ციკლზე: Kaggle (თუ კვოტა ჰყოფნის) ან ლოკალურად
                    continue
                elif "error" in s or "cancel" in s:
                    cmd_fetch(ns)  # ლოგი/სტატუსი გამოგვადგება
                    save_status(slot, status="failed", error="Kaggle: " + s.strip()[-200:])
                else:
                    save_status(slot, kaggle_state=s.strip()[-120:])
            except Exception as e:  # მაგ. კვირის კვოტა ამოიწურა — შემდეგ ციკლზე თავიდან ვცდით
                save_status(slot, error=str(e)[-300:])
        time.sleep(600)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("base")
    sub.add_parser("watch")
    for c in ("push", "status", "fetch"):
        sub.add_parser(c).add_argument("slot")
    a = ap.parse_args()
    {"base": cmd_base, "push": cmd_push, "status": cmd_status, "fetch": cmd_fetch, "watch": cmd_watch}[a.cmd](a)


if __name__ == "__main__":
    main()
