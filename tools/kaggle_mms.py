"""MMS (ტექსტის დამთხვევა) Common Voice-ის ჩანაწერებზე — Kaggle-ის 2×T4-ზე (ამ კომპიუტერზე ~30-ჯერ ნელია).

    ./aivoice kaggle-mms push     # list.jsonl + clips.tar → dataset aivoice-cv27-clips, kernel-ის გაშვება
    ./aivoice kaggle-mms status
    ./aivoice kaggle-mms fetch    # mms.jsonl → <data_root>/analysis/cv27/mms.jsonl (უკვე არსებულს ემატება)
    ./aivoice kaggle-mms wait     # ლოდინი + fetch
წინ უნდა უსწრებდეს: ./aivoice quality measure (clips.jsonl + clips/*.mp3).
"""

import argparse
import json
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).parent))
from app.config import path as cpath  # noqa: E402
from kaggle_train import kaggle, upload_dataset, user, wait_dataset_ready  # noqa: E402

DIR = cpath("data_root") / "analysis" / "cv27"
WORK = cpath("work")
SLUG = "aivoice-cv27-clips"


def kid():
    return f"{user()}/aivoice-cv27-mms"


def cmd_push(_):
    rows = [json.loads(l) for l in (DIR / "clips.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    done = set()
    if (DIR / "mms.jsonl").exists():
        done = {json.loads(l)["id"] for l in (DIR / "mms.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    todo = [r for r in rows if not r.get("err") and r["id"] not in done]
    print(f"Kaggle-ზე: {len(todo)} ჩანაწერი", flush=True)
    up = WORK / "cv27-clips"
    up.mkdir(parents=True, exist_ok=True)
    with (up / "list.jsonl").open("w", encoding="utf-8") as f:
        for r in todo:
            f.write(json.dumps({k: r.get(k) for k in ("id", "text", "dur", "lead", "trail")}, ensure_ascii=False) + "\n")
    tar = up / "clips.tar"
    if not tar.exists():
        tmp = tar.with_suffix(".part")
        with tarfile.open(tmp, "w") as tf:
            for i, r in enumerate(todo, 1):
                tf.add(DIR / "clips" / r["id"], arcname=f"clips/{r['id']}")
                if i % 10000 == 0:
                    print(f"  tar {i}/{len(todo)}", flush=True)
        tmp.rename(tar)
    print(f"ატვირთვა: {tar.stat().st_size / 1e9:.1f} GB", flush=True)
    ref = upload_dataset(up, SLUG, "aivoice cv27 clips (quality check)")
    wait_dataset_ready(ref, timeout=4 * 3600)
    k = WORK / "kernel-cv27-mms"
    k.mkdir(parents=True, exist_ok=True)
    (k / "mms.py").write_text((ROOT / "kaggle" / "kernel_mms.py").read_text(encoding="utf-8"), encoding="utf-8")
    (k / "kernel-metadata.json").write_text(json.dumps({
        "id": kid(), "title": "aivoice cv27 mms", "code_file": "mms.py", "language": "python", "kernel_type": "script",
        "is_private": True, "enable_gpu": True, "enable_internet": True, "machine_shape": "NvidiaTeslaT4",
        "dataset_sources": [ref]}))
    kaggle("kernels", "push", "-p", str(k))
    print("✓ გაშვებულია:", f"https://www.kaggle.com/code/{kid()}")


def cmd_status(_):
    return kaggle("kernels", "status", kid(), check=False).stdout.lower()


def cmd_fetch(_):
    tmp = WORK / "cv27-mms-out"
    tmp.mkdir(parents=True, exist_ok=True)
    kaggle("kernels", "output", kid(), "-p", str(tmp))
    src = tmp / "mms.jsonl"
    if not src.exists():  # ჯერ არ დასრულებულა — ნაწილობრივი შედეგები
        parts = sorted(tmp.glob("mms.part*.jsonl"))
        lines = "".join(p.read_text(encoding="utf-8") for p in parts)
    else:
        lines = src.read_text(encoding="utf-8")
    have = set()
    if (DIR / "mms.jsonl").exists():
        have = {json.loads(l)["id"] for l in (DIR / "mms.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()}
    new = [l for l in lines.splitlines() if l.strip() and json.loads(l)["id"] not in have]
    with (DIR / "mms.jsonl").open("a", encoding="utf-8") as f:
        f.write("".join(l + "\n" for l in new))
    print(f"✓ დაემატა {len(new)} შედეგი (სულ {len(have) + len(new)})")


def cmd_wait(a):
    while True:
        s = cmd_status(a)
        if "complete" in s or "error" in s or "cancel" in s:
            break
        time.sleep(300)
    cmd_fetch(a)


def main():
    ap = argparse.ArgumentParser(prog="./aivoice kaggle-mms")
    ap.add_argument("cmd", choices=("push", "status", "fetch", "wait"))
    a = ap.parse_args()
    {"push": cmd_push, "status": cmd_status, "fetch": cmd_fetch, "wait": cmd_wait}[a.cmd](a)


if __name__ == "__main__":
    main()
