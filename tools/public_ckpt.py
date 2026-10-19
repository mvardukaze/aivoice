"""საჯარო საბაზო checkpoint: ნამდვილი ხმების „ანაბეჭდების“ ცხრილი იცვლება სინთეზური ხმებით (tools/public_voices.py).

    ./aivoice public-ckpt [--model base_ka_v3]
შედეგი: <runs>/<model>_public/kaggle_out/{final.ckpt, config.json} — ამით ნებისმიერს შეუძლია „v3 საჯარო“-ს
ნარევიდან ახალი ხმის წვრთნა („წვრთნის“ ჩანართი → Kaggle), ნამდვილ ხმებზე წვდომის გარეშე.

რა იცვლება:
  • model_g.emb_g: პირველი N ხაზი = სინთეზური ხმები (designer_<model>_public/embeddings.npy), დანარჩენი — ნულები;
  • იშლება წვრთნის მდგომარეობა (optimizer, lr_scheduler, loops, callbacks) — ხმის შესაქმნელად არ სჭირდება;
  • hyper_parameters: num_speakers = N, ფაილის მისამართები — None.
ბოლოს მოწმდება: ცხრილში ნამდვილი ხმის არცერთი ხაზი არ არის (მაქს. კოსინუსური მსგავსება < 0.8).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.config import path as cpath  # noqa: E402


def main():
    ap = argparse.ArgumentParser(prog="./aivoice public-ckpt")
    ap.add_argument("--model", default="base_ka_v3")
    a = ap.parse_args()
    src = cpath("runs") / a.model / "kaggle_out"
    pub_des = cpath("designer").parent / f"designer_{a.model}_public"
    out = cpath("runs") / f"{a.model}_public" / "kaggle_out"
    out.mkdir(parents=True, exist_ok=True)

    ck = torch.load(src / "final.ckpt", map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    real = sd["model_g.emb_g.weight"].clone()
    syn = torch.from_numpy(np.load(pub_des / "embeddings.npy")).to(real.dtype)
    n = syn.shape[0]
    emb = torch.zeros_like(real)
    emb[:n] = syn
    sd["model_g.emb_g.weight"] = emb

    hp = dict(ck.get("hyper_parameters") or {})
    hp["num_speakers"] = real.shape[0]  # ცხრილის ზომა რჩება (600) — შესწორებული კოდი და export ამას ელის
    for k, v in list(hp.items()):
        if isinstance(v, str) and ("/" in v or "\\" in v):
            hp[k] = None  # ჩვენი ფაილების მისამართები (warmstart, dataset …) საჯაროდ არ გადის
    pub = {"state_dict": sd, "hyper_parameters": hp, "pytorch-lightning_version": ck.get("pytorch-lightning_version"),
           "epoch": ck.get("epoch"), "global_step": ck.get("global_step"), "hparams_name": ck.get("hparams_name")}

    # შემოწმება: ნამდვილი ხმები ცხრილში აღარ არის
    ru = torch.nn.functional.normalize(real, dim=1)
    pu = torch.nn.functional.normalize(emb[:n], dim=1)
    max_sim = float((pu @ ru.T).max())
    assert max_sim < 0.8, f"სინთეზური ხმა ნამდვილს ძალიან ჰგავს ({max_sim:.3f})"
    assert torch.count_nonzero(emb[n:]) == 0

    torch.save(pub, out / "final.ckpt")
    cfg = json.loads((src / "config.json").read_text())
    pcfg = json.loads((pub_des / "config.json").read_text())
    cfg["speaker_id_map"] = pcfg["speaker_id_map"]
    (out / "config.json").write_text(json.dumps(cfg, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps({"out": str(out / "final.ckpt"), "size_mb": round((out / "final.ckpt").stat().st_size / 1e6),
                      "synthetic": n, "max_sim_to_real": round(max_sim, 3)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
