"""მოდელები Hugging Face-ზე (საჯარო რეპოზიტორია, aivoice.toml → [hf].models_repo; ჩამოტვირთვას token არ სჭირდება).

    ./aivoice models status [--json]   # რა არის რეპოზიტორიაში და რა — ამ კომპიუტერზე
    ./aivoice models pull              # ხმის შემქმნელი, ხმები, ნარევები
    ./aivoice models pull --base       # + საბაზო checkpoint-ები (ხმის მომზადება / საბაზოს გაგრძელება)
    ./aivoice models push              # ატვირთვა: ხმები, ნარევები, შემქმნელი (მხოლოდ შეცვლილი) — Write token
    ./aivoice models push --base       # + საბაზო checkpoint-ები
    ./aivoice models pull --data       # + ხარისხის მონაცემები (~4.5 GB) — საბაზო მოდელის გასაუმჯობესებლად
    ./aivoice models push --data       # + ხარისხის მონაცემები (მხოლოდ შეცვლილი; clips/ ერთ tar-ად)

რეპოზიტორიის სტრუქტურა:
    designer/      designer.onnx, embeddings.npy, config.json, meta.json, speakers.json   (ხმის შემქმნელი)
    designs/       <id>.json                                                              (შენახული ნარევები)
    lexicon/       lexicon.json                                                           (წარმოთქმის ლექსიკონი)
    rank/          mms_rank.json                                                          (ხმების რეიტინგი, MMS)
    voices/<id>/   <id>.onnx, <id>.onnx.json, voice.json (კატალოგის ჩანაწერი)             (ხმები)
    base_ka/       final.ckpt, config.json                                                (--base)
    ljspeech/      ljspeech-medium.clean.ckpt                                             (--base)
    cv27/          clips.tar (Common Voice-ის mp3, CC0), clips.jsonl, mms.jsonl, mms.cpu.jsonl, corpus.json  (--data)
    data/          dataset_decisions.json (ყურით ამოგდებული ჩანაწერები/დინამიკები)       (--data)

რამდენიმე ადამიანი ერთ რეპოზიტორიას იყენებს: pull უმატებს ახალ ხმებს/ნარევებს და არ გადაწერს
ამ კომპიუტერზე შეცვლილ პარამეტრებს (onnx.json, voice.json) — --force-ის გარდა.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.config import SECRETS_FILE, cfg, path as cpath  # noqa: E402

LEGACY_ENV = Path.home() / ".config" / "aivoice" / "mdc.env"


def token(write=False) -> str | None:
    names = ["HF_WRITE_TOKEN", "HF_TOKEN", "HF_API_KEY"] if write else ["HF_TOKEN", "HF_WRITE_TOKEN", "HF_API_KEY"]
    env = {}
    for f in (LEGACY_ENV, SECRETS_FILE):
        if f.exists():
            for line in f.read_text(encoding="utf-8").splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip()
    for n in names:
        if os.environ.get(n) or env.get(n):
            return os.environ.get(n) or env[n]
    return None


def repo() -> str:
    r = cfg()["hf"]["models_repo"]
    if not r:
        raise SystemExit("Hugging Face-ის რეპოზიტორია არ არის მითითებული: პანელი → პარამეტრები → Hugging Face")
    return r


DESIGNS = ROOT / "data" / "designs"
CATALOG = ROOT / "voices" / "catalog.json"
RANK = ROOT / "data" / "mms_rank.json"  # ხმის შემქმნელის რეიტინგი (MMS)
LEXICON = ROOT / "data" / "lexicon.json"  # ლექსიკონის დამატებები (pull: აერთიანებს, ადგილობრივი უპირატესია)
SHARED_STATUS = ("ready", "testing")  # ეს ხმები იტვირთება; „დამალული“/„იქმნება“ — არა


def catalog() -> dict:
    try:
        return json.loads(CATALOG.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {"voices": []}


def save_catalog(cat: dict):
    CATALOG.parent.mkdir(parents=True, exist_ok=True)
    tmp = CATALOG.with_suffix(".tmp")
    tmp.write_text(json.dumps(cat, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(CATALOG)


def local_files(base: bool) -> list[tuple[Path, str]]:
    """(ლოკალური ფაილი, მისამართი რეპოზიტორიაში)."""
    des = cpath("designer")
    out = [(des / f, f"designer/{f}") for f in ("designer.onnx", "embeddings.npy", "config.json", "meta.json")]
    out.append((cpath("datasets") / "base_ka_v1" / "speakers.json", "designer/speakers.json"))
    out += [(p, f"designs/{p.name}") for p in sorted(DESIGNS.glob("*.json"))] if DESIGNS.exists() else []
    if LEXICON.exists():
        out.append((LEXICON, "lexicon/lexicon.json"))
    out += [(r, f"rank/{r.name}") for r in sorted((ROOT / "data").glob("mms_rank*.json")) if "progress" not in r.name]
    # ახალი საბაზო მოდელები (v2, v3 …): ხმის შემქმნელი + დინამიკების სია
    for d in sorted(cpath("designer").parent.glob("designer_base_ka_v*")):
        slot = d.name.removeprefix("designer_")
        out += [(d / f, f"{d.name}/{f}") for f in ("designer.onnx", "embeddings.npy", "config.json", "meta.json")]
        out.append((cpath("datasets") / slot / "speakers.json", f"{d.name}/speakers.json"))
    for v in catalog()["voices"]:
        if v.get("status") in SHARED_STATUS and v.get("model"):
            m = ROOT / "voices" / v["model"]
            out += [(m, f"voices/{v['model']}"), (Path(str(m) + ".json"), f"voices/{v['model']}.json")]
    if base:
        out += base_files()
    return out


def base_files() -> list[tuple[Path, str]]:
    extra = []
    for ko in sorted(cpath("runs").glob("base_ka_v*/kaggle_out")):  # base_ka_v2/, base_ka_v3/ … (ახალი ხმების მოსამზადებლად)
        slot = ko.parent.name
        extra += [(ko / "final.ckpt", f"{slot}/final.ckpt"), (ko / "config.json", f"{slot}/config.json")]
    ko = cpath("runs") / "base_ka" / "kaggle_out"
    return extra + [(ko / "final.ckpt", "base_ka/final.ckpt"), (ko / "config.json", "base_ka/config.json"),
            (ROOT / "training" / "checkpoints" / "en_US-ljspeech-medium" / "ljspeech-medium.clean.ckpt",
             "ljspeech/ljspeech-medium.clean.ckpt")]


# ---------- ხარისხის მონაცემები (--data): საბაზო მოდელის შერჩევისთვის (tools/select_best.py, base_ctl.py) ----------
QDIR = cpath("data_root") / "analysis" / "cv27"
DECISIONS = ROOT / "data" / "dataset_decisions.json"
CLIPS_TAR = "cv27/clips.tar"
CLIPS_MARK = QDIR / "clips.hf.json"  # {fp, sha256}: რომელ ატვირთულ/ჩამოტვირთულ tar-ს შეესაბამება clips/


def data_files() -> list[tuple[Path, str]]:
    """პატარა ფაილები (clips/ — ცალკე, tar-ად: CLIPS_TAR)."""
    return [(QDIR / n, f"cv27/{n}") for n in ("clips.jsonl", "mms.jsonl", "mms.cpu.jsonl")] + [
        (cpath("sources") / "cv27_ka" / "corpus.json", "cv27/corpus.json"),
        (DECISIONS, "data/dataset_decisions.json")]


def clips_fp() -> str | None:
    """clips/-ის „ანაბეჭდი“ (სახელი, ზომა, დრო) — tar-ს თავიდან აწყობა მხოლოდ ცვლილებისას სჭირდება."""
    d = QDIR / "clips"
    if not d.is_dir():
        return None
    h = hashlib.sha1()
    with os.scandir(d) as it:
        rows = sorted((e.name, e.stat().st_size, int(e.stat().st_mtime)) for e in it if e.is_file())
    for r in rows:
        h.update(f"{r[0]}|{r[1]}|{r[2]}\n".encode())
    return f"{len(rows)}:{h.hexdigest()}" if rows else None


def clips_mark() -> dict:
    try:
        return json.loads(CLIPS_MARK.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def build_clips_tar(dst: Path) -> Path:
    """დეტერმინისტული tar (სახელებით დალაგებული, მფლობელის გარეშე) — იგივე ფაილებიდან იგივე sha256."""
    import tarfile
    d = QDIR / "clips"
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    names = sorted(e.name for e in os.scandir(d) if e.is_file())
    with tarfile.open(tmp, "w", format=tarfile.GNU_FORMAT) as t:
        for i, n in enumerate(names):
            p = d / n
            st = p.stat()
            ti = tarfile.TarInfo(f"clips/{n}")
            ti.size, ti.mtime, ti.mode = st.st_size, int(st.st_mtime), 0o644
            with p.open("rb") as f:
                t.addfile(ti, f)
            if i % 20000 == 0:
                print(f"  tar: {i}/{len(names)}", flush=True)
    tmp.replace(dst)
    return dst


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------- შედარება: ლოკალური ფაილი = რეპოზიტორიისა? (LFS → sha256, პატარა → git blob sha1) ----------
HASH_CACHE = cpath("cache") / "hf_hashes.json"


def _hashes() -> dict:
    try:
        return json.loads(HASH_CACHE.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def local_hash(p: Path, kind: str, cache: dict) -> str:
    st = p.stat()
    key = f"{p}|{st.st_size}|{int(st.st_mtime)}|{kind}"
    if key not in cache:
        if kind == "sha256":
            cache[key] = sha256_file(p)
        else:
            h = hashlib.sha1(b"blob %d\0" % st.st_size)
            h.update(p.read_bytes())
            cache[key] = h.hexdigest()
    return cache[key]


def remote_index(api, t) -> dict:
    """{path: (size, kind, hash)}"""
    out = {}
    try:
        for f in api.list_repo_tree(repo(), recursive=True, token=t):
            if not hasattr(f, "size"):
                continue  # საქაღალდე
            lfs = getattr(f, "lfs", None)
            if lfs:
                out[f.path] = (f.size, "sha256", lfs["sha256"] if isinstance(lfs, dict) else lfs.sha256)
            else:
                out[f.path] = (f.size, "sha1", f.blob_id)
    except Exception as e:  # რეპოზიტორია ჯერ არ არსებობს / წვდომა არ არის
        if "404" in str(e) or "Repository Not Found" in str(e):
            return {}
        raise
    return out


def same(p: Path, rem, cache) -> bool:
    if rem and set(str(rem[2])) == {"*"}:  # token-ის გარეშე Hugging Face LFS-ის ჰეშს მალავს — ვადარებთ ზომით
        return p.exists() and p.stat().st_size == rem[0]
    return bool(rem) and p.exists() and p.stat().st_size == rem[0] and local_hash(p, rem[1], cache) == rem[2]


def voice_entry_bytes(v: dict) -> bytes:
    keep = dict(v)
    return (json.dumps(keep, ensure_ascii=False, indent=2) + "\n").encode()


def cmd_status(a):
    from huggingface_hub import HfApi
    t = token()
    api = HfApi(token=t)
    rem, cache = remote_index(api, t), _hashes()
    loc = {dst: src for src, dst in local_files(base=True) + data_files()}
    rows = []
    if CLIPS_TAR in rem or clips_mark() or (QDIR / "clips").is_dir():  # tar-ს აქ არ ვაწყობთ — ნიშნულით ვადარებთ
        r, have = rem.get(CLIPS_TAR), (QDIR / "clips").is_dir()
        state = ("same" if clips_mark().get("sha256") == r[2] else "differs") if (r and have) else ("remote" if r else "local")
        rows.append({"path": CLIPS_TAR, "state": state, "size": r[0] if r else 0})
    for path in sorted(set(rem) | set(loc)):
        if path == CLIPS_TAR:
            continue
        src = loc.get(path) or target_for(path, base=True, data=True)
        if src is None or path.endswith("/voice.json"):
            continue
        r = rem.get(path)
        have = bool(src and src.exists())
        state = ("same" if same(src, r, cache) else "differs") if (r and have) else ("remote" if r else "local")
        rows.append({"path": path, "state": state, "size": r[0] if r else (src.stat().st_size if have else 0)})
    _save_hashes(cache)
    if a.json:
        print(json.dumps({"repo": repo(), "files": rows}, ensure_ascii=False))
    else:
        L = {"same": "✓", "differs": "≠", "remote": "↓", "local": "↑"}
        for r in rows:
            print(f"  {L[r['state']]} {r['path']}  ({r['size'] / 1e6:.1f} MB)")
        print("✓ ერთნაირია · ≠ განსხვავდება · ↓ მხოლოდ რეპოზიტორიაში · ↑ მხოლოდ აქ")


def _save_hashes(cache):
    HASH_CACHE.parent.mkdir(parents=True, exist_ok=True)
    HASH_CACHE.write_text(json.dumps(cache), encoding="utf-8")


def cmd_push(a):
    from huggingface_hub import HfApi
    t = token(write=True)
    if not t:
        raise SystemExit("საჭიროა Hugging Face Write token (პანელი → პარამეტრები → ანგარიშები)")
    api = HfApi(token=t)
    api.create_repo(repo(), private=True, exist_ok=True)
    rem, cache = remote_index(api, t), _hashes()
    n = 0
    files = local_files(base=a.base) + (data_files() if getattr(a, "data", False) else [])
    if getattr(a, "data", False):
        n += push_clips(api, rem)
    for src, dst in files:
        if not src.exists():
            print("  გამოტოვებულია (აქ არ არის):", src)
            continue
        if same(src, rem.get(dst), cache):
            continue
        print(f"  ↑ {dst} ({src.stat().st_size / 1e6:.1f} MB)", flush=True)
        api.upload_file(path_or_fileobj=str(src), path_in_repo=dst, repo_id=repo(), repo_type="model")
        n += 1
    for v in catalog()["voices"]:  # კატალოგის ჩანაწერი — რომ სხვა კომპიუტერზე სახელი/აღწერა გადავიდეს
        if v.get("status") in SHARED_STATUS and v.get("model"):
            dst, data = f"voices/{v['id']}/voice.json", voice_entry_bytes(v)
            r = rem.get(dst)
            if r and r[1] == "sha1" and hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest() == r[2]:
                continue
            print(f"  ↑ {dst}", flush=True)
            api.upload_file(path_or_fileobj=data, path_in_repo=dst, repo_id=repo(), repo_type="model")
            n += 1
    _save_hashes(cache)
    print(f"მზადაა: ატვირთულია {n} ფაილი" if n else "ყველაფერი უკვე ატვირთულია — ცვლილება არ არის",
          f"(https://huggingface.co/{repo()})")


def push_clips(api, rem) -> int:
    """clips/ → CLIPS_TAR, მხოლოდ თუ ფაილები შეიცვალა (ანაბეჭდით; tar დროებითია და ატვირთვის შემდეგ იშლება)."""
    fp = clips_fp()
    if not fp:
        print("  გამოტოვებულია (აქ არ არის):", QDIR / "clips")
        return 0
    r, mark = rem.get(CLIPS_TAR), clips_mark()
    if r and mark.get("fp") == fp and mark.get("sha256") == r[2]:
        return 0
    print(f"  tar: {QDIR / 'clips'} → {CLIPS_TAR} (ცვლილება ან პირველი ატვირთვა)…", flush=True)
    tar = build_clips_tar(cpath("tmp") / "hf_data" / "clips.tar")
    try:
        sha = sha256_file(tar)
        if not (r and r[2] == sha):
            print(f"  ↑ {CLIPS_TAR} ({tar.stat().st_size / 1e9:.2f} GB) — შეიძლება დიდხანს გაგრძელდეს", flush=True)
            api.upload_file(path_or_fileobj=str(tar), path_in_repo=CLIPS_TAR, repo_id=repo(), repo_type="model")
        CLIPS_MARK.write_text(json.dumps({"fp": fp, "sha256": sha}), encoding="utf-8")
    finally:
        tar.unlink(missing_ok=True)
    return 0 if (r and r[2] == sha) else 1


def pull_clips(r, t, force=False) -> int:
    """CLIPS_TAR → <data_root>/analysis/cv27/clips/ (ჩამოტვირთვა დროებით საქაღალდეში, ამოღების შემდეგ იშლება)."""
    import tarfile
    from huggingface_hub import hf_hub_download
    if not force and clips_mark().get("sha256") == r[2] and (QDIR / "clips").is_dir():
        return 0
    tmp = cpath("tmp") / "hf_data"
    print(f"  ↓ {CLIPS_TAR} ({r[0] / 1e9:.2f} GB) — შეიძლება დიდხანს გაგრძელდეს", flush=True)
    try:
        p = Path(hf_hub_download(repo(), CLIPS_TAR, token=t, local_dir=tmp))
        QDIR.mkdir(parents=True, exist_ok=True)
        print(f"  ამოღება → {QDIR / 'clips'}", flush=True)
        with tarfile.open(p) as tf:
            tf.extractall(QDIR, filter="data")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    CLIPS_MARK.write_text(json.dumps({"fp": clips_fp(), "sha256": r[2]}), encoding="utf-8")
    return 1


def fetch_file(f: str, dst: Path) -> Path:
    """ერთი ფაილი რეპოზიტორიიდან (მაგ. kaggle_train.py-სთვის LJSpeech checkpoint)."""
    from huggingface_hub import hf_hub_download
    print(f"  ↓ {f} (Hugging Face: {repo()})", flush=True)
    p = hf_hub_download(repo(), f, token=token())
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.name + ".part")
    shutil.copy2(p, tmp)
    tmp.replace(dst)
    return dst


def target_for(f: str, base: bool, data: bool = False) -> Path | None:
    top, rest = (f.split("/", 1) + [""])[:2]
    if data and (f in {d: s for s, d in data_files()}):
        return {d: s for s, d in data_files()}[f]
    if top == "designer":
        return cpath("datasets") / "base_ka_v1" / "speakers.json" if rest == "speakers.json" else cpath("designer") / rest
    if top == "designs":
        return DESIGNS / rest
    if top == "lexicon":
        return LEXICON
    if top == "rank":
        return ROOT / "data" / (rest or "mms_rank.json")
    if top.startswith("designer_base_ka_v"):
        slot = top.removeprefix("designer_")
        return cpath("datasets") / slot / "speakers.json" if rest == "speakers.json" else cpath("designer").parent / top / rest
    if base and top.startswith("base_ka_v"):
        return cpath("runs") / top / "kaggle_out" / rest
    if top == "voices":
        return ROOT / f
    if base and top == "base_ka":
        return cpath("runs") / "base_ka" / "kaggle_out" / rest
    if base and top == "ljspeech":
        return ROOT / "training" / "checkpoints" / "en_US-ljspeech-medium" / rest
    return None


def cmd_pull(a):
    from huggingface_hub import HfApi, hf_hub_download
    t = token()
    api = HfApi(token=t)
    rem, cache = remote_index(api, t), _hashes()
    if not rem:
        raise SystemExit(f"რეპოზიტორია {repo()} ცარიელია ან წვდომა არ არის (token?)")
    cat = catalog()
    known = {v["id"]: v for v in cat["voices"]}
    n = 0
    data = getattr(a, "data", False)
    if data and CLIPS_TAR in rem:
        n += pull_clips(rem[CLIPS_TAR], t, a.force)
    for f, r in sorted(rem.items()):
        dst = target_for(f, a.base, data)
        if dst is None or f.endswith("/voice.json"):
            continue
        if dst.exists() and not a.force and f.endswith(".jsonl") and dst.stat().st_size >= r[0]:
            continue  # ანალიზის შედეგები მხოლოდ ემატება — აქ უფრო სრული (ან იგივე) ფაილი არ გადაიწერება
        if f == "lexicon/lexicon.json":  # გაერთიანება: ახალი სიტყვები ემატება, აქაური არ იცვლება
            remote = json.loads(Path(hf_hub_download(repo(), f, token=t)).read_text(encoding="utf-8"))
            local = json.loads(LEXICON.read_text(encoding="utf-8")) if LEXICON.exists() else {}
            new = {k: v for k, v in remote.items() if k not in local}
            if new:
                LEXICON.write_text(json.dumps({**remote, **local}, ensure_ascii=False, indent=1), encoding="utf-8")
                print(f"  + ლექსიკონი: {len(new)} ახალი სიტყვა")
            continue
        # ამ კომპიუტერზე შეცვლილი პარამეტრები/ნარევები არ გადაიწერება (--force-ის გარდა)
        keep_local = f.endswith(".onnx.json") or f.startswith("designs/") or f.startswith("rank/") or f.startswith("data/")
        if dst.exists() and (same(dst, r, cache) or (keep_local and not a.force)):
            continue
        print(f"  ↓ {f} ({r[0] / 1e6:.1f} MB)", flush=True)
        p = hf_hub_download(repo(), f, token=t)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_name(dst.name + ".part")
        shutil.copy2(p, tmp)
        tmp.replace(dst)
        n += 1
    # კატალოგი: ახალი ხმები რეპოზიტორიიდან (voice.json); არსებულს — მხოლოდ მოდელის მისამართი
    for f in rem:
        if not f.endswith("/voice.json"):
            continue
        vid = f.split("/")[1]
        entry = json.loads(Path(hf_hub_download(repo(), f, token=t)).read_text(encoding="utf-8"))
        if vid not in known:
            entry.setdefault("status", "testing")  # ავტორის სტატუსი (დამტკიცებული → „მზადაა“)
            cat["voices"].append(entry)
            print(f"  + ახალი ხმა: {entry.get('name', vid)} ({vid})")
        elif a.force:
            known[vid].update({k: x for k, x in entry.items() if k != "status"})
    for v in cat["voices"]:
        if v.get("model") and (ROOT / "voices" / v["model"]).exists() and v.get("status") == "planned":
            v["status"] = "testing"
    save_catalog(cat)
    _save_hashes(cache)
    print(f"მზადაა: ჩამოტვირთულია {n} ფაილი" if n else "ყველაფერი უკვე ჩამოტვირთულია")


def main():
    ap = argparse.ArgumentParser(prog="./aivoice models")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pull")
    p.add_argument("--base", action="store_true", help="საბაზო checkpoint-ებიც (~1.8 GB)")
    p.add_argument("--data", action="store_true", help="ხარისხის მონაცემებიც (~4.5 GB) — საბაზოს გასაუმჯობესებლად")
    p.add_argument("--force", action="store_true", help="ამ კომპიუტერზე შეცვლილი პარამეტრებიც გადაიწეროს")
    p = sub.add_parser("push")
    p.add_argument("--base", action="store_true", help="საბაზო checkpoint-ებიც")
    p.add_argument("--data", action="store_true", help="ხარისხის მონაცემებიც (clips/ → tar, მხოლოდ შეცვლისას)")
    p = sub.add_parser("status")
    p.add_argument("--json", action="store_true")
    a = ap.parse_args()
    try:
        {"pull": cmd_pull, "push": cmd_push, "status": cmd_status}[a.cmd](a)
    except Exception as e:
        if type(e).__name__ == "GatedRepoError" or ("401" in str(e) and "gated" in str(e).lower()):
            raise SystemExit(
                f"რეპოზიტორია {repo()} „gated“-ია: ჩამოსატვირთად საჭიროა უფასო Hugging Face ანგარიში —\n"
                f"  1) https://huggingface.co/{repo()} → „Agree and access repository“ (ავტომატურად დასტურდება)\n"
                "  2) huggingface.co → Settings → Access Tokens → Read token → პანელი → პარამეტრები → ანგარიშები") from None
        raise


if __name__ == "__main__":
    main()
