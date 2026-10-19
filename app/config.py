"""პროექტის კონფიგურაცია — ერთი ადგილი ყველა მისამართისა და ანგარიშისთვის.

ფაილი: <repo>/aivoice.toml (git-ში არ ინახება; არ არსებობს → ნაგულისხმევი მნიშვნელობები).
პანელის „პარამეტრების“ ჩანართი იგივე ფაილს წერს. პანელიც და tools/*-იც ამას კითხულობს:

    from app.config import cfg, path
    path("datasets") / "base_ka_v1"
    cfg()["server"]["port"]

მისამართებში შეიძლება {data_root} და ~ — მაგ. "{data_root}/datasets".
საიდუმლოებები (Kaggle/HF/Mozilla გასაღებები) აქ არ ინახება — იხ. SECRETS_FILE.
"""

from __future__ import annotations

import os
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_FILE = Path(os.environ.get("AIVOICE_CONFIG", ROOT / "aivoice.toml"))
SECRETS_FILE = Path.home() / ".config" / "aivoice" / "secrets.env"

# სქემა: სექცია → გასაღები → (ნაგულისხმევი, სათაური, ახსნა). პანელის ფორმა აქედან იწყობა.
SCHEMA: dict[str, dict[str, tuple]] = {
    "paths": {
        "data_root": ("~/aivoice-data", "მონაცემების საქაღალდე",
                      "დიდი ფაილები: dataset-ები, მოდელები, ჩანაწერები, წვრთნის შედეგები. სასურველია სწრაფი დისკი, ≥ 50 GB."),
        "datasets": ("{data_root}/datasets", "Dataset-ები", "წვრთნის ჩანაწერები (wav/flac + metadata.csv)."),
        "runs": ("{data_root}/runs", "წვრთნის შედეგები", "checkpoint-ები, ნიმუშები, სტატუსი."),
        "designer": ("{data_root}/designer", "ხმის შემქმნელის მოდელი", "საბაზო მოდელის ONNX + დინამიკების ვექტორები."),
        "sources": ("{data_root}/sources", "წყაროები", "Common Voice-ის არქივები და ცხრილები."),
        "work": ("{data_root}/kaggle", "Kaggle-ის დროებითი ფაილები", "ატვირთვამდე მომზადებული საქაღალდეები."),
        "gcp": ("{data_root}/gcp", "Google Cloud-ის ფაილები", "მანქანის მდგომარეობა და ატვირთვები."),
        "recordings": ("{data_root}/recordings", "ჩანაწერები", "„ჩაწერის“ ჩანართის ხმები (პირადი — არსად იტვირთება)."),
        "cache": ("{data_root}/cache", "ქეში", "გენერირებული აუდიო; შეიძლება წაიშალოს."),
        "tmp": ("{data_root}/tmp", "დროებითი", ""),
    },
    "server": {
        "host": ("127.0.0.1", "პანელის მისამართი", "127.0.0.1 — მხოლოდ ეს კომპიუტერი; 0.0.0.0 — ლოკალური ქსელიდანაც (პაროლი არ აქვს!)."),
        "port": (8200, "პორტი", ""),
    },
    "kaggle": {
        "base_dataset": ("aivoice-base2", "საბაზო dataset Kaggle-ზე",
                         "Piper-ის ფაილები + LJSpeech checkpoint თქვენს Kaggle-ის ანგარიშზე (იქმნება ავტომატურად)."),
        "week_hours": (30, "კვირის GPU კვოტა (სთ)", ""),
    },
    "gcp": {
        "gcloud": ("gcloud", "gcloud ბრძანება", "სრული მისამართი, თუ PATH-ში არ არის (მაგ. ~/google-cloud-sdk/bin/gcloud)."),
        "project": ("", "Google Cloud პროექტი", "პროექტის ID (მაგ. my-aivoice-123). ცარიელი — gcloud-ის ნაგულისხმევი (gcloud config set project …)."),
        "ssh_key": ("~/.ssh/aivoice_gcp", "SSH გასაღები", "იქმნება ავტომატურად პირველ გაშვებაზე."),
        "regions": ("us-central1,us-east4,us-east1,us-west1,us-west4,europe-west4,europe-west1,asia-southeast1",
                    "რეგიონები (უპირატესობის რიგით)", "L4 GPU სადაც თავისუფალია. აშშ ყველაზე იაფია."),
    },
    "hf": {
        "models_repo": ("mvarduka/aivoice-models", "Hugging Face მოდელების რეპოზიტორია",
                        "აქედან მოდის ხმის შემქმნელი, ხმები და საბაზო მოდელი (პირადი — საჭიროა Read token და ავტორის წვდომა)."),
    },
    "training": {
        "local_threads": (12, "ლოკალური წვრთნის ნაკადები", "პროცესორის ნაკადები ლოკალური (CPU) წვრთნისთვის."),
    },
}

_cache: dict = {"mtime": None, "cfg": None}


def defaults() -> dict:
    return {s: {k: v[0] for k, v in keys.items()} for s, keys in SCHEMA.items()}


def cfg() -> dict:
    """მიმდინარე კონფიგურაცია (ფაილი + ნაგულისხმევი); ფაილის შეცვლისას თავიდან იკითხება."""
    m = CONFIG_FILE.stat().st_mtime if CONFIG_FILE.exists() else None
    if _cache["cfg"] is None or _cache["mtime"] != m:
        c = defaults()
        if CONFIG_FILE.exists():
            with CONFIG_FILE.open("rb") as f:
                user = tomllib.load(f)
            for s, vals in user.items():
                if isinstance(vals, dict):
                    c.setdefault(s, {}).update(vals)
        _cache.update(mtime=m, cfg=c)
    return _cache["cfg"]


def expand(value: str) -> Path:
    c = cfg()
    v = str(value).replace("{data_root}", str(c["paths"]["data_root"]))
    return Path(os.path.expanduser(v))


def path(name: str) -> Path:
    """მისამართი სახელით (paths სექციიდან), {data_root} და ~ გახსნილი."""
    return expand(cfg()["paths"][name])


def api_url(sub: str = "") -> str:
    s = cfg()["server"]
    host = "127.0.0.1" if s["host"] in ("0.0.0.0", "::") else s["host"]
    return f"http://{host}:{s['port']}{sub}"


def configured() -> bool:
    return CONFIG_FILE.exists()


def _toml_value(v) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return str(v)
    return '"' + str(v).replace("\\", "\\\\").replace('"', '\\"') + '"'


def save(values: dict) -> dict:
    """ინახავს მხოლოდ ნაგულისხმევისგან განსხვავებულ მნიშვნელობებს (ფაილი მოკლე და გასაგები რჩება)."""
    d = defaults()
    lines = ["# aivoice — კონფიგურაცია (პანელი → „პარამეტრები“ ან ხელით). იხ. app/config.py", ""]
    for s, keys in SCHEMA.items():
        changed = []
        for k, meta in keys.items():
            v = values.get(s, {}).get(k, cfg()[s].get(k, meta[0]))
            if isinstance(meta[0], int) and not isinstance(meta[0], bool):
                v = int(v)
            if k == "data_root" or v != d[s][k]:
                changed.append(f"{k} = {_toml_value(v)}")
        if changed:
            lines += [f"[{s}]", *changed, ""]
    tmp = CONFIG_FILE.with_suffix(".tmp")
    tmp.write_text("\n".join(lines), encoding="utf-8")
    tmp.replace(CONFIG_FILE)
    _cache["cfg"] = None
    return cfg()


def ensure_dirs():
    for k in SCHEMA["paths"]:
        try:
            path(k).mkdir(parents=True, exist_ok=True)
        except OSError:
            pass
