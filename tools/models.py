"""მოდელები Hugging Face-ზე (პირადი რეპოზიტორია, aivoice.toml → [hf].models_repo).

    ./aivoice models status [--json]   # რა არის რეპოზიტორიაში და რა — ამ კომპიუტერზე
    ./aivoice models pull              # ხმის შემქმნელი, ხმები, ნარევები
    ./aivoice models pull --base       # + საბაზო checkpoint-ები (ხმის მომზადება / საბაზოს გაგრძელება)
    ./aivoice models push              # ატვირთვა: ხმები, ნარევები, შემქმნელი (მხოლოდ შეცვლილი) — Write token
    ./aivoice models push --base       # + საბაზო checkpoint-ები

რეპოზიტორიის სტრუქტურა:
    designer/      designer.onnx, embeddings.npy, config.json, meta.json, speakers.json   (ხმის შემქმნელი)
    designs/       <id>.json                                                              (შენახული ნარევები)
    lexicon/       lexicon.json                                                           (წარმოთქმის ლექსიკონი)
    rank/          mms_rank.json                                                          (ხმების რეიტინგი, MMS)
    voices/<id>/   <id>.onnx, <id>.onnx.json, voice.json (კატალოგის ჩანაწერი)             (ხმები)
    base_ka/       final.ckpt, config.json                                                (--base)
    ljspeech/      ljspeech-medium.clean.ckpt                                             (--base)

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
    if RANK.exists():
        out.append((RANK, "rank/mms_rank.json"))
    for v in catalog()["voices"]:
        if v.get("status") in SHARED_STATUS and v.get("model"):
            m = ROOT / "voices" / v["model"]
            out += [(m, f"voices/{v['model']}"), (Path(str(m) + ".json"), f"voices/{v['model']}.json")]
    if base:
        out += base_files()
    return out


def base_files() -> list[tuple[Path, str]]:
    ko = cpath("runs") / "base_ka" / "kaggle_out"
    return [(ko / "final.ckpt", "base_ka/final.ckpt"), (ko / "config.json", "base_ka/config.json"),
            (ROOT / "training" / "checkpoints" / "en_US-ljspeech-medium" / "ljspeech-medium.clean.ckpt",
             "ljspeech/ljspeech-medium.clean.ckpt")]


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
            h = hashlib.sha256()
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 22), b""):
                    h.update(chunk)
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
    return bool(rem) and p.exists() and p.stat().st_size == rem[0] and local_hash(p, rem[1], cache) == rem[2]


def voice_entry_bytes(v: dict) -> bytes:
    keep = dict(v)
    return (json.dumps(keep, ensure_ascii=False, indent=2) + "\n").encode()


def cmd_status(a):
    from huggingface_hub import HfApi
    t = token()
    api = HfApi(token=t)
    rem, cache = remote_index(api, t), _hashes()
    loc = {dst: src for src, dst in local_files(base=True)}
    rows = []
    for path in sorted(set(rem) | set(loc)):
        src = loc.get(path) or target_for(path, base=True)
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
    files = local_files(base=a.base)
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


def target_for(f: str, base: bool) -> Path | None:
    top, rest = (f.split("/", 1) + [""])[:2]
    if top == "designer":
        return cpath("datasets") / "base_ka_v1" / "speakers.json" if rest == "speakers.json" else cpath("designer") / rest
    if top == "designs":
        return DESIGNS / rest
    if top == "lexicon":
        return LEXICON
    if top == "rank":
        return RANK
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
    for f, r in sorted(rem.items()):
        dst = target_for(f, a.base)
        if dst is None or f.endswith("/voice.json"):
            continue
        if f == "lexicon/lexicon.json":  # გაერთიანება: ახალი სიტყვები ემატება, აქაური არ იცვლება
            remote = json.loads(Path(hf_hub_download(repo(), f, token=t)).read_text(encoding="utf-8"))
            local = json.loads(LEXICON.read_text(encoding="utf-8")) if LEXICON.exists() else {}
            new = {k: v for k, v in remote.items() if k not in local}
            if new:
                LEXICON.write_text(json.dumps({**remote, **local}, ensure_ascii=False, indent=1), encoding="utf-8")
                print(f"  + ლექსიკონი: {len(new)} ახალი სიტყვა")
            continue
        # ამ კომპიუტერზე შეცვლილი პარამეტრები/ნარევები არ გადაიწერება (--force-ის გარდა)
        keep_local = f.endswith(".onnx.json") or f.startswith("designs/") or f.startswith("rank/")
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
    p.add_argument("--force", action="store_true", help="ამ კომპიუტერზე შეცვლილი პარამეტრებიც გადაიწეროს")
    p = sub.add_parser("push")
    p.add_argument("--base", action="store_true", help="საბაზო checkpoint-ებიც")
    p = sub.add_parser("status")
    p.add_argument("--json", action="store_true")
    a = ap.parse_args()
    {"pull": cmd_pull, "push": cmd_push, "status": cmd_status}[a.cmd](a)


if __name__ == "__main__":
    main()
