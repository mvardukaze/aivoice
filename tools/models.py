"""მოდელები Hugging Face-ზე (პირადი რეპოზიტორია, aivoice.toml → [hf].models_repo).

    ./aivoice models pull              # ყველაფერი, რაც პანელს და ხმების შექმნას სჭირდება
    ./aivoice models pull --base       # + საბაზო checkpoint-ები (ხმის მომზადება / საბაზოს გაგრძელება)
    ./aivoice models push              # (ავტორი) ატვირთვა — საჭიროა Write token

რეპოზიტორიის სტრუქტურა:
    designer/      designer.onnx, embeddings.npy, config.json, meta.json, speakers.json   (ხმის შემქმნელი)
    voices/<id>/   <id>.onnx, <id>.onnx.json                                             (მზა ხმები)
    base_ka/       final.ckpt, config.json                                                (--base)
    ljspeech/      ljspeech-medium.clean.ckpt                                             (--base)
"""

import argparse
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


def catalog() -> dict:
    return json.loads((ROOT / "voices" / "catalog.json").read_text(encoding="utf-8"))


def local_files(base: bool) -> list[tuple[Path, str]]:
    """(ლოკალური ფაილი, მისამართი რეპოზიტორიაში)."""
    des = cpath("designer")
    out = [(des / f, f"designer/{f}") for f in ("designer.onnx", "embeddings.npy", "config.json", "meta.json")]
    out.append((cpath("datasets") / "base_ka_v1" / "speakers.json", "designer/speakers.json"))
    for v in catalog()["voices"]:
        if v.get("status") == "ready" and v.get("model"):
            m = ROOT / "voices" / v["model"]
            out += [(m, f"voices/{v['model']}"), (Path(str(m) + ".json"), f"voices/{v['model']}.json")]
    if base:
        ko = cpath("runs") / "base_ka" / "kaggle_out"
        out += [(ko / "final.ckpt", "base_ka/final.ckpt"), (ko / "config.json", "base_ka/config.json"),
                (ROOT / "training" / "checkpoints" / "en_US-ljspeech-medium" / "ljspeech-medium.clean.ckpt",
                 "ljspeech/ljspeech-medium.clean.ckpt")]
    return out


def cmd_push(a):
    from huggingface_hub import HfApi
    t = token(write=True)
    if not t:
        raise SystemExit("საჭიროა Hugging Face Write token (პანელი → პარამეტრები → ანგარიშები)")
    api = HfApi(token=t)
    api.create_repo(repo(), private=True, exist_ok=True)
    for src, dst in local_files(base=True):
        if not src.exists():
            print("  გამოტოვებულია (არ არის):", src)
            continue
        print(f"  ↑ {dst} ({src.stat().st_size / 1e6:.0f} MB)", flush=True)
        api.upload_file(path_or_fileobj=str(src), path_in_repo=dst, repo_id=repo(), repo_type="model")
    print("მზადაა:", f"https://huggingface.co/{repo()}")


def cmd_pull(a):
    from huggingface_hub import hf_hub_download, list_repo_files
    t = token()
    files = list_repo_files(repo(), token=t)
    targets = {}
    for f in files:
        top = f.split("/", 1)[0]
        if top == "designer":
            targets[f] = (cpath("designer") / f.split("/", 1)[1]) if not f.endswith("speakers.json") \
                else cpath("datasets") / "base_ka_v1" / "speakers.json"
        elif top == "voices":
            targets[f] = ROOT / f
        elif a.base and top == "base_ka":
            targets[f] = cpath("runs") / "base_ka" / "kaggle_out" / f.split("/", 1)[1]
        elif a.base and top == "ljspeech":
            targets[f] = ROOT / "training" / "checkpoints" / "en_US-ljspeech-medium" / f.split("/", 1)[1]
    for f, dst in targets.items():
        if dst.exists() and not a.force:
            print("  ✓ უკვე არის:", dst)
            continue
        print("  ↓", f, flush=True)
        p = hf_hub_download(repo(), f, token=t)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
    # კატალოგი: ჩამოტვირთული ხმები — „მზა“
    cat = catalog()
    for v in cat["voices"]:
        if v.get("model") and (ROOT / "voices" / v["model"]).exists():
            v["status"] = "ready"
    (ROOT / "voices" / "catalog.json").write_text(json.dumps(cat, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("მზადაა. პანელი: ./aivoice serve")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pull")
    p.add_argument("--base", action="store_true", help="საბაზო checkpoint-ებიც (~1.8 GB)")
    p.add_argument("--force", action="store_true")
    sub.add_parser("push")
    a = ap.parse_args()
    {"pull": cmd_pull, "push": cmd_push}[a.cmd](a)


if __name__ == "__main__":
    main()
