"""Common Voice 27 (ქართული) — ჩამოტვირთვა Mozilla Data Collective-იდან თქვენი API key-ით + დინამიკების ანალიზი.

    ./aivoice cv status                       რა არის უკვე ამ კომპიუტერზე (ქსელის გარეშე)
    ./aivoice cv download --dry-run           გეგმა: მონაცემთა ნაკრები, ზომა, ადგილი დისკზე (არაფერს წერს)
    ./aivoice cv download --check             გასაღების/პირობების შემოწმება: ჩამოტვირთვის სესია + 1 ბაიტი (არქივის გარეშე)
    ./aivoice cv download --accept-terms      ჩამოტვირთვა (~10.4 GB, გაგრძელებადი) → TSV ცხრილები → corpus.json

რას აკეთებს (რაც უკვე არის, გამოტოვებს):
  1. <sources>/cv27_ka/<slug>.tar.gz — datacollective.download_dataset (შეწყვეტისას .part + .checksum რჩება და
     შემდეგ ჯერზე Range-ით გრძელდება); დასრულებისას sha256 მოწმდება MDC-ის checksum-თან.
  2. არქივიდან მხოლოდ ცხრილები (validated.tsv, other.tsv, clip_durations.tsv …) — mp3 არქივში რჩება
     (tools/cv_corpus.py და tools/clip_quality.py მას პირდაპირ კითხულობენ).
  3. tools/cv_corpus.py → corpus.json (წვრთნის გარემოთი: ./install.sh --training).

API key: MDC_API_KEY (პანელი → პარამეტრები → ანგარიშები; ~/.config/aivoice/secrets.env).
მდგომარეობა და ლოგი პანელისთვის: <data_root>/analysis/cv_download/{state.json, log.txt}.

პასუხისმგებლობა: მონაცემები ჩამოიტვირთება თქვენი ანგარიშით, Mozilla-ს პირობებით; მასში ნამდვილი ადამიანების ხმებია.
დინამიკების ვინაობის დადგენა აკრძალულია. ნამდვილ ხმებზე გაწვრთნილი მოდელები და მათი გამოყენება თქვენი პასუხისმგებლობაა.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.config import SECRETS_FILE, path as cpath  # noqa: E402

DATASET = "common-voice-scripted-speech-27-0-georgi-6faf6605"   # Common Voice Scripted Speech 27.0 - Georgian (CC0-1.0)
EXPECTED_BYTES = 10_435_235_149                                  # MDC-ის sizeBytes (ქსელის გარეშე სტატუსისთვის)
DATASET_URL = "https://mozilladatacollective.com/datasets/cmu5x9qda00fenq07byl8rlc8"
TERMS = ["https://mozilladatacollective.com/terms", "https://commonvoice.mozilla.org/terms"]
REQUIRED = ("validated.tsv", "other.tsv", "clip_durations.tsv")
TABLES = REQUIRED + ("invalidated.tsv", "train.tsv", "dev.tsv", "test.tsv", "reported.tsv",
                     "validated_sentences.tsv", "unvalidated_sentences.tsv", "README.md")
LEGACY_ENV = Path.home() / ".config" / "aivoice" / "mdc.env"
TPY = ROOT / "training" / ".venv" / "bin" / "python"

SRC = cpath("sources") / "cv27_ka"
RUN = cpath("data_root") / "analysis" / "cv_download"


def api_key() -> str | None:
    if os.environ.get("MDC_API_KEY"):
        return os.environ["MDC_API_KEY"]
    env = {}
    for f in (LEGACY_ENV, SECRETS_FILE):
        if f.exists():
            for line in f.read_text(encoding="utf-8").splitlines():
                if "=" in line and not line.lstrip().startswith("#"):
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip()
    return env.get("MDC_API_KEY") or None


def archive() -> Path | None:
    return next(iter(sorted(SRC.glob("*.tar.gz"))), None) if SRC.exists() else None


def status() -> dict:
    """რა არის დისკზე (ქსელის გარეშე) — პანელიც ამას იყენებს."""
    a = archive()
    part = next(iter(sorted(SRC.glob("*.tar.gz.part"))), None) if SRC.exists() else None
    return {
        "dir": str(SRC), "key": bool(api_key()), "dataset": DATASET, "dataset_url": DATASET_URL, "terms": TERMS,
        "expected_bytes": EXPECTED_BYTES,
        "archive": a.name if a else None, "archive_bytes": a.stat().st_size if a else 0,
        "part_bytes": part.stat().st_size if part else 0,
        "tables": [t for t in TABLES if (SRC / t).exists()], "tables_ok": all((SRC / t).exists() for t in REQUIRED),
        "corpus": (SRC / "corpus.json").exists(), "training_env": TPY.exists(),
        "free_bytes": shutil.disk_usage(_existing(SRC)).free,
    }


def _existing(p: Path) -> Path:
    while not p.exists():
        p = p.parent
    return p


# ---------- მდგომარეობა პანელისთვის ----------
_state: dict = {}


def save_state(**kw):
    _state.update(kw, updated=time.strftime("%Y-%m-%dT%H:%M:%S"))
    RUN.mkdir(parents=True, exist_ok=True)
    tmp = RUN / "state.tmp"
    tmp.write_text(json.dumps(_state, ensure_ascii=False), encoding="utf-8")
    tmp.replace(RUN / "state.json")


def say(*a):
    print(*a, flush=True)


def gb(n: float) -> str:
    return f"{n / 1e9:.2f} GB"


# ---------- 1. ჩამოტვირთვა ----------
def details():
    import datacollective as dc
    return dc.get_dataset_details(DATASET)   # საჯარო endpoint — გასაღები არ სჭირდება


def explain(e: Exception) -> str:
    name = type(e).__name__
    if isinstance(e, PermissionError) or "403" in str(e):
        return ("წვდომა უარყოფილია (403). ყველაზე ხშირი მიზეზი: მონაცემთა ნაკრების პირობებს ჯერ არ დაეთანხმეთ — გახსენით "
                f"{DATASET_URL} თქვენი ანგარიშით და დაეთანხმეთ (Terms & Conditions), მერე სცადეთ თავიდან.")
    if name == "AuthenticationError" or "401" in str(e):
        return "API key არასწორია ან ვადაგასულია (401) — შექმენით ახალი mozilladatacollective.com-ზე და ჩაწერეთ პანელში."
    if name == "RateLimitError" or "429" in str(e):
        return "ძალიან ბევრი მოთხოვნა (429) — სცადეთ რამდენიმე წუთში."
    return f"{name}: {e}"


def check_key(d) -> None:
    """ჩამოტვირთვის სესია (POST /datasets/{id}/download) + პირველი ბაიტი Range-ით — არქივი არ ჩამოიტვირთება."""
    import requests
    from datacollective.api_utils import _get_api_url
    r = requests.post(f"{_get_api_url()}/datasets/{d.id}/download", headers={"Authorization": f"Bearer {api_key()}"}, timeout=30)
    if r.status_code != 200:
        msg = {401: "AuthenticationError 401", 403: "PermissionError 403", 429: "RateLimitError 429"}.get(r.status_code, f"HTTP {r.status_code}")
        raise SystemExit("✗ " + explain(PermissionError(msg) if r.status_code == 403 else RuntimeError(msg)) + f" · {r.text[:200]}")
    p = r.json()
    say(f"✓ გასაღები მუშაობს, პირობებზე დათანხმებულია · ზომა {gb(int(p.get('sizeBytes') or 0))} · checksum {str(p.get('checksum'))[:12]}…")
    g = requests.get(p["downloadUrl"], headers={"Range": "bytes=0-0"}, timeout=30, stream=True)
    say(f"✓ ჩამოტვირთვის მისამართი პასუხობს: HTTP {g.status_code}"
        + (" — შეწყვეტისას გაგრძელდება (Range)" if g.status_code == 206 else " — Range არ მუშაობს: შეწყვეტისას თავიდან დაიწყება"))
    g.close()


def download(d) -> Path:
    import datacollective as dc
    os.environ["MDC_API_KEY"] = api_key() or ""
    SRC.mkdir(parents=True, exist_ok=True)
    target = SRC / d.filename
    part = target.with_name(target.name + ".part")
    total = int(d.sizeBytes or EXPECTED_BYTES)
    have = part.stat().st_size if part.exists() else 0
    free = shutil.disk_usage(SRC).free
    need = total - have + 600e6   # + ცხრილები და ნიმუშები
    if free < need:
        raise SystemExit(f"✗ დისკზე ადგილი არ არის: საჭიროა ~{gb(need)}, თავისუფალია {gb(free)} ({SRC})")
    say(f"ჩამოტვირთვა: {d.filename} ({gb(total)})" + (f" — გაგრძელება {gb(have)}-დან" if have else ""))
    save_state(stage="download", done=have, total=total, rate=None, eta_min=None)

    stop = threading.Event()

    def watch():  # პროგრესი .part ფაილის ზომით (ბიბლიოთეკის საკუთარი ზოლი ლოგში არ ჩანს)
        last, t0 = (part.stat().st_size if part.exists() else 0), time.time()
        while not stop.wait(5):
            cur = part.stat().st_size if part.exists() else 0
            dt = time.time() - t0
            rate = (cur - last) / dt if dt > 0 else 0
            last, t0 = cur, time.time()
            eta = round((total - cur) / rate / 60) if rate > 0 else None
            save_state(done=cur, rate=round(rate / 1e6, 1), eta_min=eta)
            say(f"  {gb(cur)} / {gb(total)} ({cur * 100 // total}%) · {rate / 1e6:.1f} MB/წმ" + (f" · დარჩა ~{eta} წთ" if eta is not None else ""))

    th = threading.Thread(target=watch, daemon=True)
    th.start()
    try:
        out = dc.download_dataset(DATASET, download_directory=str(SRC), show_progress=False)
    except Exception as e:  # noqa: BLE001
        cause = e.__cause__ or e
        raise SystemExit("✗ ჩამოტვირთვა შეწყდა: " + explain(cause if type(e).__name__ == "DownloadError" else e)
                         + "\n  ხელახლა გაშვებისას გაგრძელდება იქიდან, სადაც შეწყდა.")
    finally:
        stop.set()
    save_state(done=total)
    if d.checksum and re.fullmatch(r"[0-9a-f]{64}", d.checksum):
        say("sha256-ის შემოწმება…")
        save_state(stage="verify", done=0)
        h, n = hashlib.sha256(), 0
        with open(out, "rb") as f:
            while b := f.read(8 << 20):
                h.update(b)
                n += len(b)
                if n % (1 << 30) < (8 << 20):
                    save_state(done=n)
        if h.hexdigest() != d.checksum:
            bad = out.with_name(out.name + ".bad")
            out.replace(bad)
            raise SystemExit(f"✗ checksum არ ემთხვევა — ფაილი გადაირქვა {bad.name}-ად; წაშალეთ და სცადეთ თავიდან.")
        say("✓ checksum ემთხვევა")
    return Path(out)


# ---------- 2. ცხრილები ----------
def extract_tables(a: Path) -> None:
    want = [t for t in TABLES if not (SRC / t).exists()]
    if all((SRC / t).exists() for t in REQUIRED):
        say("ცხრილები უკვე ამოღებულია")
        return
    say(f"ცხრილების ამოღება არქივიდან: {', '.join(want)}")
    save_state(stage="extract", done=0, total=len(want))
    got = set()
    with tarfile.open(a, "r|gz") as tf:
        for m in tf:
            name = Path(m.name).name
            if m.isfile() and name in want and "/clips/" not in m.name and re.fullmatch(r"[\w.-]+\.(tsv|md)", name):
                tmp = SRC / (name + ".tmp")
                with tf.extractfile(m) as src, tmp.open("wb") as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
                tmp.replace(SRC / name)
                got.add(name)
                save_state(done=len(got))
                say(f"  {name} ({m.size / 1e6:.1f} MB)")
            # Common Voice-ში ცხრილები არქივის დასაწყისშია — პირველი mp3-ის შემდეგ ვჩერდებით, თუ სავალდებულოები უკვე გვაქვს
            elif name.endswith(".mp3") and all((SRC / t).exists() for t in REQUIRED):
                break
            if got >= set(want):
                break
    miss = [t for t in REQUIRED if not (SRC / t).exists()]
    if miss:
        raise SystemExit(f"✗ არქივში ვერ მოიძებნა: {', '.join(miss)}")


# ---------- 3. დინამიკების ანალიზი ----------
def analyse(workers: int) -> None:
    if (SRC / "corpus.json").exists():
        say("corpus.json უკვე არის")
        return
    if not TPY.exists():
        raise SystemExit("✗ დინამიკების ანალიზს სჭირდება წვრთნის გარემო: ./install.sh --training — "
                         "მერე გაუშვით იგივე ბრძანება (ჩამოტვირთული აღარ ჩამოიტვირთება).")
    say("დინამიკების ანალიზი (tools/cv_corpus.py) — 10–40 წთ…")
    save_state(stage="analyse", done=None, total=None)
    r = subprocess.run(["nice", "-n", "10", str(TPY), str(ROOT / "tools" / "cv_corpus.py"), str(SRC), "--workers", str(workers)],
                       cwd=ROOT, env={**os.environ, "PYTHONUNBUFFERED": "1"})
    if r.returncode or not (SRC / "corpus.json").exists():
        raise SystemExit(f"✗ ანალიზი ვერ დასრულდა (კოდი {r.returncode})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("download", "status"))
    ap.add_argument("--accept-terms", action="store_true",
                    help="ვადასტურებ: ვკითხულობ Mozilla Data Collective-ისა და Common Voice-ის პირობებს და ვიღებ პასუხისმგებლობას")
    ap.add_argument("--dry-run", action="store_true", help="მხოლოდ გეგმა")
    ap.add_argument("--check", action="store_true", help="გასაღების შემოწმება (არქივის ჩამოტვირთვის გარეშე)")
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 1)))
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()

    s = status()
    if a.cmd == "status":
        if a.json:
            print(json.dumps(s, ensure_ascii=False))
            return
        say(f"საქაღალდე: {s['dir']}")
        say(f"API key: {'✓' if s['key'] else '✗ (პანელი → პარამეტრები → ანგარიშები)'}")
        say(f"არქივი: {s['archive'] + ' · ' + gb(s['archive_bytes']) if s['archive'] else '✗'}"
            + (f" (ნაწილობრივ: {gb(s['part_bytes'])})" if s["part_bytes"] else ""))
        say(f"ცხრილები: {'✓' if s['tables_ok'] else '✗'} · corpus.json: {'✓' if s['corpus'] else '✗'}")
        return

    need_dl = not s["archive"]
    if a.dry_run or a.check:
        try:
            d = details()
            say(f"მონაცემთა ნაკრები: {d.name} · {d.licenseAbbreviation} · {gb(int(d.sizeBytes or 0))} · {d.filename}")
            say(f"გვერდი (პირობებზე დასათანხმებლად): {d.datasetUrl or DATASET_URL}")
        except Exception as e:  # noqa: BLE001
            raise SystemExit("✗ MDC-ის კატალოგი არ პასუხობს: " + explain(e))
        say(f"საქაღალდე: {SRC} · თავისუფალი ადგილი {gb(s['free_bytes'])}")
        resume = f" (გაგრძელება {gb(s['part_bytes'])}-დან)" if s["part_bytes"] else ""
        say(f"1. ჩამოტვირთვა: {'საჭიროა' + resume if need_dl else 'უკვე არის — ' + s['archive']}")
        say(f"2. ცხრილები: {'უკვე არის' if s['tables_ok'] else 'ამოიღება არქივიდან'}")
        say(f"3. corpus.json: {'უკვე არის' if s['corpus'] else ('cv_corpus.py (წვრთნის გარემო)' if s['training_env'] else 'საჭიროა ./install.sh --training')}")
        say(f"API key: {'✓' if s['key'] else '✗ — პანელი → პარამეტრები → ანგარიშები → Mozilla Data Collective API key'}")
        if a.check:
            if not s["key"]:
                raise SystemExit(1)
            check_key(d)
        return

    if not a.accept_terms:
        raise SystemExit("საჭიროა --accept-terms: მონაცემები ჩამოიტვირთება თქვენი Mozilla Data Collective-ის ანგარიშით, Mozilla-ს\n"
                         "პირობებით; მასში ნამდვილი ადამიანების ხმებია; დინამიკების ვინაობის დადგენა აკრძალულია; ნამდვილ ხმებზე\n"
                         "გაწვრთნილი მოდელები და მათი გამოყენება თქვენი პასუხისმგებლობაა. პირობები: " + " · ".join(TERMS))
    if need_dl and not s["key"]:
        raise SystemExit("✗ MDC_API_KEY არ არის: პანელი → პარამეტრები → ანგარიშები → Mozilla Data Collective API key")

    RUN.mkdir(parents=True, exist_ok=True)
    lock = (RUN / "lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit("✗ ჩამოტვირთვა უკვე მიმდინარეობს (სხვა პროცესი)")
    _state.clear()
    save_state(stage="start", pgid=os.getpgid(0), pid=os.getpid(), started=time.strftime("%Y-%m-%dT%H:%M:%S"),
               accepted_terms=True, error=None)
    say(f"==== Common Voice 27 (ქართული) → {SRC}  [{time.strftime('%Y-%m-%d %H:%M')}; პირობები დადასტურებულია]")
    try:
        a_path = download(details()) if need_dl else SRC / s["archive"]
        if not need_dl:
            say(f"არქივი უკვე არის: {a_path.name} ({gb(a_path.stat().st_size)})")
        extract_tables(a_path)
        analyse(a.workers)
    except SystemExit as e:
        save_state(stage="error", error=str(e.code) if e.code not in (None, 0) else None)
        raise
    except Exception as e:  # noqa: BLE001
        save_state(stage="error", error=explain(e))
        raise SystemExit("✗ " + explain(e))
    save_state(stage="done", error=None)
    say("✓ მზადაა: „მონაცემები → ხარისხი“ → „შემოწმების დაწყება“")


if __name__ == "__main__":
    main()
