"""პროექტის განახლება GitHub-იდან (პანელი → „პარამეტრები“ → „განახლება“).

    ./aivoice update check [--json]   # არის თუ არა ახალი ვერსია
    ./aivoice update apply            # git pull + საჭიროების შემთხვევაში ბიბლიოთეკები / Piper-ის შესწორებები
პანელი ამის შემდეგ თავიდან უნდა ჩაიტვირთოს (ღილაკი ამას თავად აკეთებს).
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TPY = ROOT / "training" / ".venv" / "bin" / "python"


def git(*args, timeout=60) -> str:
    r = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError((r.stderr or r.stdout).strip().splitlines()[-1] if (r.stderr or r.stdout).strip() else "git შეცდომა")
    return r.stdout.strip()


def check() -> dict:
    out = {"ok": True}
    try:
        branch = git("rev-parse", "--abbrev-ref", "HEAD")
        git("fetch", "--quiet", "origin", branch, timeout=60)
    except (RuntimeError, OSError, subprocess.TimeoutExpired) as e:
        msg = str(e)
        if "Authentication" in msg or "could not read Username" in msg or "403" in msg or "not found" in msg.lower():
            msg = "GitHub-თან წვდომა არ არის — ამ კომპიუტერზე GitHub-ში შესვლა საჭიროა (gh auth login)"
        return {"ok": False, "error": msg}
    up = f"origin/{branch}"
    out["branch"] = branch
    out["current"] = git("log", "-1", "--format=%h %cs %s")
    behind = int(git("rev-list", "--count", f"HEAD..{up}"))
    ahead = int(git("rev-list", "--count", f"{up}..HEAD"))
    out.update(behind=behind, ahead=ahead)
    out["commits"] = git("log", "--format=%h|%cs|%s", f"HEAD..{up}").splitlines() if behind else []
    changed = git("diff", "--name-only", f"HEAD...{up}").splitlines() if behind else []
    out["changed"] = changed
    out["needs"] = {"requirements": "requirements.txt" in changed,
                    "training": any(f == "requirements-training.txt" or f.startswith("piper_overlay/") for f in changed)}
    # ხელით შეცვლილი ფაილები (git-ში), რომლებსაც განახლება გადაწერდა
    dirty = [l.split(None, 1)[1].split(" -> ")[-1] for l in git("status", "--porcelain", "--untracked-files=no").splitlines() if l.strip()]
    out["blocked"] = sorted(set(dirty) & set(changed))
    out["dirty"] = dirty
    return out


def apply():
    c = check()
    if not c["ok"]:
        raise SystemExit(c["error"])
    if c["ahead"]:
        raise SystemExit(f"ამ კომპიუტერზე {c['ahead']} საკუთარი ცვლილებაა (commit) — ავტომატური განახლება შეუძლებელია")
    if c["blocked"]:
        raise SystemExit("ეს ფაილები ხელით არის შეცვლილი და განახლება მათ შეცვლიდა: " + ", ".join(c["blocked"])
                         + "\nგადაარჩინეთ ასლი და დააბრუნეთ: git checkout -- <ფაილი>")
    if not c["behind"]:
        print("✓ უკვე უახლესი ვერსიაა:", c["current"])
        return
    print(f"↓ {c['behind']} ცვლილება:")
    for line in c["commits"]:
        h, d, s = line.split("|", 2)
        print(f"   {d}  {s}")
    print(git("pull", "--ff-only", "--quiet", "origin", c["branch"], timeout=300) or "✓ კოდი განახლდა", flush=True)
    if c["needs"]["requirements"]:
        print("… ბიბლიოთეკები (requirements.txt)", flush=True)
        subprocess.run([str(ROOT / ".venv" / "bin" / "pip"), "install", "-q", "-r", "requirements.txt"], cwd=ROOT, check=True)
    if c["needs"]["training"] and TPY.exists():
        print("… წვრთნის გარემო (requirements-training.txt, piper_overlay/)", flush=True)
        subprocess.run([str(TPY), "-m", "pip", "install", "-q", "-r", "requirements-training.txt"], cwd=ROOT, check=True)
        site = subprocess.run([str(TPY), "-c", "import piper, os; print(os.path.dirname(os.path.dirname(piper.__file__)))"],
                              capture_output=True, text=True, check=True).stdout.strip()
        if "site-packages" in site:  # რედაქტირებად კლონს (ძველი დაყენება) არ ვეხებით
            subprocess.run(["cp", "-R", str(ROOT / "piper_overlay" / "piper") + "/.", f"{site}/piper/"], check=True)
            ma = Path(site) / "piper" / "train" / "vits" / "monotonic_align"
            subprocess.run(f'"{TPY.parent}/cythonize" -i core.pyx >/dev/null && mkdir -p monotonic_align && mv -f core*.so monotonic_align/',
                           shell=True, cwd=ma, check=True)
    print("✓ განახლდა:", git("log", "-1", "--format=%h %cs %s"))
    print("პანელი თავიდან უნდა ჩაიტვირთოს.")


def main():
    ap = argparse.ArgumentParser(prog="./aivoice update")
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("check")
    p.add_argument("--json", action="store_true")
    sub.add_parser("apply")
    a = ap.parse_args()
    if a.cmd == "apply":
        apply()
    else:
        c = check()
        if getattr(a, "json", False):
            print(json.dumps(c, ensure_ascii=False))
        elif not c["ok"]:
            sys.exit(c["error"])
        else:
            print(f"ახლა: {c['current']}")
            print(f"ახალი ცვლილებები: {c['behind']}" + "".join(f"\n   {x.split('|')[1]}  {x.split('|')[2]}" for x in c["commits"]))
            if c["blocked"]:
                print("ხელით შეცვლილი (ხელს შეუშლის):", ", ".join(c["blocked"]))


if __name__ == "__main__":
    main()
