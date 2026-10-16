"""წვრთნა Google Cloud-ის L4-ზე ($300 საცდელი კრედიტი): base_ka-ს გაგრძელება ან ახალი მოდელი (create base_ka_v2).

გამოყენება (gcloud: aivoice.toml [gcp].gcloud, ანგარიში აქტიური, impersonation: aivoice-trainer SA):
    python3 tools/gcp_train.py create     # spot მანქანა g2-standard-8 (1× L4), us-central1 (a→b→c)
    python3 tools/gcp_train.py push       # dataset + piper + checkpoint → მანქანა, setup, წვრთნის ჩართვა
    python3 tools/gcp_train.py status     # მანქანის მდგომარეობა + წვრთნის ბოლო სტრიქონები
    python3 tools/gcp_train.py start      # spot-ის შეწყვეტის შემდეგ თავიდან ჩართვა
    python3 tools/gcp_train.py fetch      # უახლესი checkpoint → training/runs/base_ka/gcp/
    python3 tools/gcp_train.py stop|delete

მანქანაზე: /data (dataset, run/, train.log), სერვისი aivoice-train (gcp/train_vm.sh) აგრძელებს უახლესი
checkpoint-იდან ყოველი ჩართვისას. SSH: ცალკე გასაღები ~/.ssh/aivoice_gcp, მომხმარებელი aivoice.
"""

import json
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.config import cfg, expand, path as cpath  # noqa: E402,F401
G = str(expand(cfg()["gcp"]["gcloud"])) if "/" in cfg()["gcp"]["gcloud"] else cfg()["gcp"]["gcloud"]
NAME = "aivoice-l4"
ZONES = ["us-central1-a", "us-central1-b", "us-central1-c"]
KEY = expand(cfg()["gcp"]["ssh_key"])
STAGE = cpath("gcp")
STATE = STAGE / "state.json"
# რომელ მოდელს ვწვრთნით — state.json-ის "slot" (create <slot>); ძველი გაშვებები: base_ka
SLOT = (json.loads(STATE.read_text()).get("slot") if STATE.exists() else None) or "base_ka"
RUN = cpath("runs") / SLOT
JOB = ROOT / "data" / "kaggle_jobs" / f"{SLOT}.json"
LJSPEECH = ROOT / "training" / "checkpoints" / "en_US-ljspeech-medium" / "ljspeech-medium.clean.ckpt"
# L4 24 GB: batch 32 (Kaggle-ზე T4 16 GB — 24); 8 vCPU → 6 worker (აუდიო/სპექტროგრამა პროცესორზე)
VM_JOB = {"batch": 32, "workers": 6, "ckpt_every_steps": 2000}


def gc(*args, check=True, capture=True):
    r = subprocess.run([G, *args, "--quiet"], capture_output=capture, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"gcloud {' '.join(args[:3])}: {(r.stderr or '')[-600:]}")
    return r


def state() -> dict:
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(**kw):
    st = state()
    st.update(kw, updated=time.strftime("%Y-%m-%dT%H:%M:%S"))
    STATE.write_text(json.dumps(st, ensure_ascii=False, indent=1))


def describe():
    z = state().get("zone")
    if not z:
        return None
    r = gc("compute", "instances", "describe", NAME, "--zone", z, "--format", "json", check=False)
    return json.loads(r.stdout) if r.returncode == 0 else None


def ip():
    d = describe()
    return d["networkInterfaces"][0]["accessConfigs"][0].get("natIP") if d else None


def ssh_base(host):
    return ["-i", str(KEY), "-o", "StrictHostKeyChecking=no", "-o", "UserKnownHostsFile=/dev/null",
            "-o", "LogLevel=ERROR", "-o", "ConnectTimeout=20"]


def ssh(cmd: str, check=True, capture=True, timeout=None):
    h = ip()
    r = subprocess.run(["ssh", *ssh_base(h), f"aivoice@{h}", cmd], capture_output=capture, text=True, timeout=timeout)
    if check and r.returncode != 0:
        raise RuntimeError(f"ssh: {(r.stderr or r.stdout or '')[-600:]}")
    return r


def scp(src: str, dst: str):
    h = ip()
    subprocess.run(["scp", *ssh_base(h), "-q", src, f"aivoice@{h}:{dst}"], check=True)


def cmd_create(slot: str | None = None):
    """create [slot] — ახალი მოდელისთვის ძველი state ინახება state.<slot>.json-ად."""
    global SLOT, RUN, JOB
    if slot and slot != SLOT:
        if STATE.exists():
            STATE.replace(STAGE / f"state.{SLOT}.json")
        SLOT, RUN, JOB = slot, cpath("runs") / slot, ROOT / "data" / "kaggle_jobs" / f"{slot}.json"
        save_state(slot=slot)
    assert JOB.exists(), f"{JOB} არ არის"
    pub = KEY.with_suffix(".pub").read_text().strip()
    zones = l4_zones() or ZONES
    # ჯერ spot (იაფი) ყველა სასურველ ზონაში, მერე on-demand — L4 ხშირად ამოწურულია (STOCKOUT)
    for spot in (True, False):
        for z in zones:
            opts = ["--provisioning-model", "SPOT", "--instance-termination-action", "STOP"] if spot else []
            r = gc("compute", "instances", "create", NAME, "--zone", z,
                   "--machine-type", "g2-standard-8", *opts, "--maintenance-policy", "TERMINATE",
                   "--image-family", "pytorch-2-9-cu129-ubuntu-2204-nvidia-580",
                   "--image-project", "deeplearning-platform-release",
                   "--boot-disk-size", "100GB", "--boot-disk-type", "pd-balanced",
                   "--no-service-account", "--no-scopes",
                   "--metadata", f"ssh-keys=aivoice:{pub}", check=False)
            if r.returncode == 0:
                save_state(zone=z, created=time.strftime("%Y-%m-%dT%H:%M:%S"), phase="created", slot=SLOT,
                           provisioning="spot" if spot else "on-demand", disk=NAME)
                print("created in", z, "spot" if spot else "on-demand", flush=True)
                return
            print(z, "spot" if spot else "on-demand", "→", (r.stderr or "").strip().splitlines()[-1:], flush=True)
    raise SystemExit("ვერცერთ ზონაში ვერ შეიქმნა (L4 ახლა არ არის) — მოგვიანებით სცადეთ")


def wait_ssh(timeout=600):
    t = time.time()
    while time.time() - t < timeout:
        try:
            if ssh("true", check=False, timeout=30).returncode == 0:
                return
        except subprocess.TimeoutExpired:
            pass
        time.sleep(15)
    raise RuntimeError("SSH არ პასუხობს")


def best_local_ckpt() -> Path:
    r = subprocess.run([str(ROOT / "training" / ".venv" / "bin" / "python"), "-c",
                        "import sys, json; sys.path.insert(0, 'tools'); import local_train as l; "
                        f"b = l.best_ckpt({SLOT!r}); print(b[2] if b else '')"],
                       capture_output=True, text=True, cwd=ROOT, env={"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin"})
    return Path(r.stdout.strip().splitlines()[-1])


def cmd_push():
    """dataset + Piper + checkpoint → მანქანა. ახალი მოდელი (job-ში "warmstart", ჯერ checkpoint არ აქვს) — LJSpeech-იდან."""
    wait_ssh()
    job = json.loads(JOB.read_text())
    ds = Path(job["dataset_dir"]).name
    tar = STAGE / f"{ds}.tar"
    assert tar.exists(), f"dataset-ის არქივი ჯერ არ არის მზად: {tar}"
    fresh = bool(job.get("warmstart")) and not (RUN / "gcp" / "latest.ckpt").exists() and SLOT != "base_ka"
    # გაგრძელება სხვა მოდელიდან (job: "resume_from", მაგ. v2-ის final.ckpt) — ან უკვე ჩამოტვირთული უახლესი
    if fresh:
        ckpt = LJSPEECH
    elif (RUN / "gcp" / "latest.ckpt").exists():
        ckpt = RUN / "gcp" / "latest.ckpt"
    elif job.get("resume_from"):
        ckpt = Path(job["resume_from"])
    else:
        ckpt = best_local_ckpt()
    vm_job = {**VM_JOB, "num_speakers": job["num_speakers"], "max_epochs": job["max_epochs"], "dataset": ds,
              "voice_name": SLOT, "fresh": fresh}
    with tempfile.TemporaryDirectory() as td:
        ov = Path(td) / "piper_overlay.tar"
        with tarfile.open(ov, "w") as t:  # ჩვენი შესწორებები Piper-ის train/-ზე + core.pyx (repo: piper_overlay/)
            t.add(ROOT / "piper_overlay" / "piper", arcname="piper")
        jp = Path(td) / "job.json"
        jp.write_text(json.dumps(vm_job))
        ssh("sudo mkdir -p /data/upload && sudo chown -R aivoice /data")
        files = [(str(ov), ov.name),
                 (str(jp), "job.json"), (str(ROOT / "gcp" / "setup_vm.sh"), "setup_vm.sh"),
                 (str(ROOT / "gcp" / "train_vm.sh"), "train_vm.sh"), (str(ROOT / "gcp" / "autostop.sh"), "autostop.sh"),
                 (str(ckpt), "warmstart.ckpt" if fresh else "resume.ckpt")]
        files.append((str(tar), tar.name))
        # უკვე ატვირთული (იგივე ზომის) ფაილები გამოტოვდება — ატვირთვა ~3 MB/წმ-ია
        sizes = dict(l.split(" ", 1)[::-1] for l in ssh("cd /data/upload && stat -c '%s %n' * 2>/dev/null",
                                                        check=False).stdout.split("\n") if " " in l)
        for s, name in files:
            if sizes.get(name) == str(Path(s).stat().st_size):
                continue
            print("upload", name, flush=True)
            scp(s, f"/data/upload/{name}")
    save_state(phase="setup", resume_from=str(ckpt))
    r = ssh("sudo bash /data/upload/setup_vm.sh 2>&1 | tail -20", check=False, capture=True)
    print(r.stdout)
    if "SETUP_OK" not in r.stdout:
        save_state(phase="setup_failed", error=r.stdout[-500:])
        raise SystemExit("setup ჩავარდა")
    ssh("sudo systemctl enable --now aivoice-train.service")
    save_state(phase="training", started=time.strftime("%Y-%m-%dT%H:%M:%S"), error=None)
    print("training started")


def cmd_status():
    d = describe()
    print("VM:", d["status"] if d else "არ არსებობს", "| zone:", state().get("zone"), "| ip:", ip())
    if d and d["status"] == "RUNNING":
        r = ssh("ls /data/DONE 2>/dev/null; systemctl is-active aivoice-train; "
                "tail -c 3000 /data/train.log | tr '\\r' '\\n' | grep -a 'Epoch' | tail -1; "
                "ls -t /data/run/lightning_logs/*/checkpoints/ 2>/dev/null | head -3; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader",
                check=False)
        print(r.stdout)


def cmd_start():
    gc("compute", "instances", "start", NAME, "--zone", state()["zone"])
    print("started")


def cmd_fetch():
    """უახლესი checkpoint → training/runs/base_ka/gcp/latest.ckpt (+ config.json)."""
    r = ssh("ls -t /data/run/lightning_logs/*/checkpoints/local-*.ckpt | head -1")
    remote = r.stdout.strip()
    out = RUN / "gcp"
    out.mkdir(parents=True, exist_ok=True)
    h = ip()
    tmp = out / "latest.ckpt.part"
    side = out / "latest.ckpt.part.src"
    if not side.exists() or side.read_text() != remote:  # სხვა ფაილის ნაწილი — თავიდან
        tmp.unlink(missing_ok=True)
        side.write_text(remote)
    # rsync --partial: შორეულ ზონასთან (სინგაპური) კავშირი ხანდახან წყდება — გაგრძელდება იქიდან, სადაც შეწყდა
    sh = "ssh " + " ".join(ssh_base(h))
    for attempt in range(40):
        r = subprocess.run(["rsync", "-q", "--partial", "--append-verify", "-e", sh,
                            f"aivoice@{h}:{remote}", str(tmp)])
        if r.returncode == 0:
            break
        time.sleep(10)
    else:
        raise RuntimeError("checkpoint-ის ჩამოტვირთვა ვერ მოხერხდა")
    tmp.replace(out / "latest.ckpt")
    subprocess.run(["scp", *ssh_base(h), "-q", f"aivoice@{h}:/data/run/config.json", str(out / "config.json")], check=False)
    save_state(fetched=remote, fetched_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    print("fetched", remote)


def cmd_stop():
    gc("compute", "instances", "stop", NAME, "--zone", state()["zone"])
    save_state(phase="stopped")


def cmd_delete():
    gc("compute", "instances", "delete", NAME, "--zone", state()["zone"])
    save_state(phase="deleted")


TPY = str(ROOT / "training" / ".venv" / "bin" / "python")
FETCH_EVERY_H = 4


def remote_progress() -> dict:
    """ეპოქა უახლესი checkpoint-ის სახელიდან, ბატჩი — progress bar-იდან."""
    import re
    o = ssh("ls -t /data/run/lightning_logs/*/checkpoints/*.ckpt 2>/dev/null | head -1; echo ---; "
            "tail -c 3000 /data/train.log | tr '\\r' '\\n' | grep -a -E '(Epoch|Training)' | tail -1; echo ---; "
            "ls /data/DONE 2>/dev/null; systemctl is-active aivoice-train", check=False, timeout=90).stdout.split("---")
    p = {}
    if m := re.search(r"e(\d+)-s(\d+)", o[0]):
        p.update(epoch=int(m.group(1)), step=int(m.group(2)), remote_ckpt=True)
    line = o[1] if len(o) > 1 else ""
    if m := re.search(r"\| *(\d+)/(\d+|\?)", line):
        p["batch"] = int(m.group(1))
    if m := re.search(r"Epoch (\d+):", line):  # progress bar-ის ეპოქა checkpoint-ზე ახალია
        p["epoch"] = int(m.group(1))
    tail = o[2] if len(o) > 2 else ""
    p["done"] = "/data/DONE" in tail
    p["service"] = tail.strip().splitlines()[-1] if tail.strip() else ""
    return p


def export_designer(ckpt: Path):
    # ახალი მოდელი (base_ka_v2…) — ცალკე საქაღალდეში, რომ მიმდინარე ხმის შემქმნელი წვრთნის დროს არ შეიცვალოს
    out = cpath("designer") if SLOT == "base_ka" else cpath("designer").parent / f"designer_{SLOT}"
    cfgp = RUN / "kaggle_out" / "config.json"
    if not cfgp.exists():
        cfgp = RUN / "gcp" / "config.json"
    r = subprocess.run([TPY, str(ROOT / "tools" / "export_designer.py"), str(ckpt), "--out", str(out),
                        "--config", str(cfgp), "--source", "gcp"],
                       capture_output=True, text=True, env={"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin", "OMP_NUM_THREADS": "4"})
    print("designer:", (r.stdout.strip() or r.stderr.strip())[-200:], flush=True)


def finalize():
    """წვრთნა დასრულდა: საბოლოო checkpoint → kaggle_out/final.ckpt, ONNX, ნიმუშები, დიზაინერი; მანქანა იშლება."""
    cmd_fetch()
    src = RUN / "gcp" / "latest.ckpt"
    out = RUN / "kaggle_out"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "final.ckpt").exists():
        (out / "final.ckpt").replace(out / "final.prev.ckpt")
    src.replace(out / "final.ckpt")
    r = subprocess.run([TPY, "-m", "piper.train.export_onnx", "--checkpoint", str(out / "final.ckpt"),
                        "--output-file", str(out / f"{SLOT}.onnx")], capture_output=True, text=True,
                       env={"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin"}, cwd=ROOT / "training")
    print("onnx:", r.returncode, r.stderr[-300:], flush=True)
    import shutil
    if not (out / "config.json").exists():
        shutil.copy2(RUN / "gcp" / "config.json", out / "config.json")
    shutil.copy2(out / "config.json", str(out / f"{SLOT}.onnx") + ".json")
    p = state().get("progress", {})
    st_p = out / "status.json"
    st = json.loads(st_p.read_text()) if st_p.exists() else {}
    st.update(status="done", epoch=p.get("epoch"), source="gcp", gpu="Google Cloud L4")
    st_p.write_text(json.dumps(st, ensure_ascii=False, indent=1))
    export_designer(out / "final.ckpt")  # status.json-ის შემდეგ — meta.json-ში სწორი ეპოქა ჩაიწეროს
    subprocess.run([TPY, str(ROOT / "tools" / "make_previews.py"), str(out / "final.ckpt"), "--label", "საბოლოო", "--slot", SLOT],
                   capture_output=True, env={"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin", "OMP_NUM_THREADS": "4"})
    sys.path.insert(0, str(ROOT / "tools"))
    import kaggle_train as kt
    kt.save_status(SLOT, status="done", epoch=p.get("epoch"), device="gcp", finished=time.strftime("%Y-%m-%dT%H:%M:%S"))
    cmd_delete()
    save_state(phase="done")


# L4-ის ზონები უპირატესობის რიგით: ჯერ აშშ (იაფი), მერე ევროპა. კვოტა (1 L4) ყველა რეგიონშია.
PREFERRED_REGIONS = [r.strip() for r in cfg()["gcp"]["regions"].split(",") if r.strip()]


def l4_zones() -> list[str]:
    r = gc("compute", "accelerator-types", "list", "--filter", "name=nvidia-l4", "--format", "value(zone)", check=False)
    zones = sorted(set(r.stdout.split()))
    rank = lambda z: (PREFERRED_REGIONS.index(z.rsplit("-", 1)[0]) if z.rsplit("-", 1)[0] in PREFERRED_REGIONS else 99, z)
    return sorted(zones, key=rank)


def relocate():
    """მიმდინარე ზონაში L4 ამოწურულია (STOCKOUT): დისკის snapshot → სხვა ზონაში ახალი მანქანა (on-demand —
    spot ხშირად წყდება). მონაცემები და checkpoint-ები დისკზეა, წვრთნა თავისით გრძელდება."""
    st = state()
    old_zone = st["zone"]
    snap = f"{NAME}-snap"
    gc("compute", "snapshots", "delete", snap, check=False)
    gc("compute", "snapshots", "create", snap, "--source-disk", st.get("disk", NAME), "--source-disk-zone", old_zone)
    pub = KEY.with_suffix(".pub").read_text().strip()
    for z in l4_zones():
        if z == old_zone:
            continue
        disk = f"{NAME}-{z}"
        if gc("compute", "disks", "create", disk, "--zone", z, "--source-snapshot", snap, "--type", "pd-balanced",
              check=False).returncode != 0:
            continue
        r = gc("compute", "instances", "create", NAME, "--zone", z, "--machine-type", "g2-standard-8",
               "--maintenance-policy", "TERMINATE", "--disk", f"name={disk},boot=yes,auto-delete=yes",
               "--no-service-account", "--no-scopes", "--metadata", f"ssh-keys=aivoice:{pub}", check=False)
        if r.returncode == 0:
            print("relocated to", z, flush=True)
            gc("compute", "instances", "delete", NAME, "--zone", old_zone, check=False)
            gc("compute", "snapshots", "delete", snap, check=False)
            save_state(zone=z, disk=disk, relocated=time.strftime("%Y-%m-%dT%H:%M:%S"),
                       relocations=st.get("relocations", 0) + 1, provisioning="on-demand", error=None)
            return z
        print(z, "→", (r.stderr or "").strip().splitlines()[-1:], flush=True)
        gc("compute", "disks", "delete", disk, "--zone", z, check=False)
    raise RuntimeError("L4 არცერთ ზონაში არ არის თავისუფალი — მოგვიანებით ვცდი")


def cmd_watch():
    """სერვისი (aivoice-gcp): ყოველ 10 წთ-ში — spot-ის შეწყვეტის შემდეგ ჩართვა, პროგრესი პანელისთვის,
    ყოველ 4 სთ-ში checkpoint → ხმის შემქმნელი, დასრულებისას finalize."""
    sys.path.insert(0, str(ROOT / "tools"))
    import kaggle_train as kt
    while True:
        st = state()
        if st.get("phase") == "training":
            try:
                d = describe()
                if not d:
                    save_state(error="მანქანა ვერ ვიპოვე")
                elif d["status"] in ("TERMINATED", "STOPPED", "SUSPENDED"):
                    print("VM", d["status"], "— ვრთავ", flush=True)
                    r = gc("compute", "instances", "start", NAME, "--zone", st["zone"], check=False)
                    save_state(restarts=st.get("restarts", 0) + 1, last_restart=time.strftime("%Y-%m-%dT%H:%M:%S"),
                               error=None if r.returncode == 0 else (r.stderr or "")[-300:])
                    if r.returncode != 0 and "STOCKOUT" in (r.stderr or "") + (r.stdout or "") \
                            or r.returncode != 0 and "enough resources" in (r.stderr or ""):
                        relocate()
                elif d["status"] == "RUNNING":
                    p = remote_progress()
                    if not p.get("service"):  # SSH-მა პასუხი არ დააბრუნა (მაგ. ქსელი ჯერ არ არის მზად ჩართვისას)
                        save_state(error="მანქანასთან SSH კავშირი ვერ დამყარდა — შემდეგ ციკლზე ვცდი")
                        time.sleep(600)
                        continue
                    if p.get("epoch") is None:  # პირველ checkpoint-მდე — საიდანაც გაგრძელდა
                        import re
                        m = re.search(r"e(\d+)-s(\d+)", st.get("resume_from", ""))
                        if m:
                            p.update(epoch=int(m.group(1)), step=int(m.group(2)))
                    save_state(progress=p, error=None)
                    kt.save_status(SLOT, epoch=p.get("epoch"), device="gcp", status="training",
                                   kaggle_state=f"Google Cloud L4 · ეპოქა {p.get('epoch', 0) + 1}, ბატჩი {p.get('batch', '?')}")
                    if p.get("done"):
                        finalize()
                    elif time.time() - st.get("fetched_ts", 0) > FETCH_EVERY_H * 3600 and p.get("remote_ckpt"):
                        cmd_fetch()
                        save_state(fetched_ts=time.time())
                        export_designer(RUN / "gcp" / "latest.ckpt")
                        n = len(list((RUN / "previews").glob("k*")))
                        subprocess.run([TPY, str(ROOT / "tools" / "make_previews.py"), str(RUN / "gcp" / "latest.ckpt"),
                                        "--label", f"Google Cloud #{n if SLOT != 'base_ka' else n - 1}", "--slot", SLOT],
                                       capture_output=True,
                                       env={"CUDA_VISIBLE_DEVICES": "", "PATH": "/usr/bin:/bin", "OMP_NUM_THREADS": "4"})
            except Exception as e:
                save_state(error=str(e)[-300:])
                print("error:", e, flush=True)
        time.sleep(600)


if __name__ == "__main__":
    if sys.argv[1] == "create" and len(sys.argv) > 2:
        cmd_create(sys.argv[2])
        sys.exit(0)
    {"create": cmd_create, "push": cmd_push, "status": cmd_status, "start": cmd_start,
     "fetch": cmd_fetch, "stop": cmd_stop, "delete": cmd_delete, "watch": cmd_watch}[sys.argv[1]]()
