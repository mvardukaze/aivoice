"""multi-speaker საბაზო checkpoint + ნარევის ვექტორი → single-speaker საწყისი checkpoint (წვრთნისთვის warmstart).

VITS-ში დინამიკის ვექტორი g ყველგან მუდმივ წანაცვლებად ემატება (x + cond(g)) — ამიტომ ის შეიძლება
შესაბამისი შრის bias-ში „ჩაცხვეს“: dec.conv_pre, dp.pre (SDP), WN in_layers (enc_q და flow-ები).
შედეგი: single-speaker მოდელი, რომელიც პირველივე ნაბიჯიდან ზუსტად ნარევის ხმით ლაპარაკობს.

    training/.venv/bin/python tools/fold_speaker.py --design "ქალი 1" --out <paths.work>/init-f_mix_01/init.ckpt
"""

import argparse
import json
import sys
from pathlib import Path


import torch

from piper.train.vits.lightning import VitsModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.config import path as cpath  # noqa: E402,F401
BASE = cpath("runs") / "base_ka" / "kaggle_out" / "final.ckpt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=str(BASE))
    a = ap.parse_args()

    design = next(d for p in (ROOT / "data" / "designs").glob("*.json")
                  if (d := json.loads(p.read_text(encoding="utf-8")))["name"] == a.design)
    base = VitsModel.load_from_checkpoint(a.base, map_location="cpu", weights_only=False).eval()
    gm = base.model_g
    smap = json.loads((Path(a.base).parent / "config.json").read_text())["speaker_id_map"]
    emb = gm.emb_g.weight.detach()
    tot = sum(design["weights"].values())
    g = sum((w / tot) * emb[smap[n]] for n, w in design["weights"].items())[None, :, None]  # [1, gin, 1]

    hp = dict(base.hparams)
    hp.update(num_speakers=1, gin_channels=0)
    hp = {k: v for k, v in hp.items() if k not in ("warmstart_ckpt", "vocoder_warmstart_ckpt")}
    single = VitsModel(**hp)
    new_sd = single.state_dict()
    base_sd = base.state_dict()
    copied = 0
    for k in new_sd:
        if k in base_sd and base_sd[k].shape == new_sd[k].shape:
            new_sd[k] = base_sd[k].clone()
            copied += 1

    with torch.no_grad():
        folds = 0
        new_sd["model_g.dec.conv_pre.bias"] += gm.dec.cond(g)[0, :, 0]
        folds += 1
        new_sd["model_g.dp.pre.bias"] += gm.dp.cond(g)[0, :, 0]
        folds += 1
        wns = {"model_g.enc_q.enc": gm.enc_q.enc}
        for i, f in enumerate(gm.flow.flows):
            if hasattr(f, "enc"):
                wns[f"model_g.flow.flows.{i}.enc"] = f.enc
        for name, wn in wns.items():
            gl = wn.cond_layer(g)[0, :, 0]
            h2 = 2 * wn.hidden_channels
            for j in range(wn.n_layers):
                new_sd[f"{name}.in_layers.{j}.bias"] += gl[j * h2:(j + 1) * h2]
                folds += 1
    single.load_state_dict(new_sd)
    single.eval()

    # შემოწმება: ხმაურის გარეშე (noise=0) ორივე მოდელმა ერთი და იგივე უნდა გამოსცეს
    x = torch.randint(1, 50, (1, 40))
    xl = torch.LongTensor([40])
    with torch.no_grad():
        gm_out = base.model_g
        sid_g = g[:, :, 0]
        gm_out.emb_g = torch.nn.Identity()
        y1 = gm_out.infer(x, xl, sid=sid_g, noise_scale=0.0, noise_scale_w=0.0, length_scale=1.0)[0]
        y2 = single.model_g.infer(x, xl, sid=None, noise_scale=0.0, noise_scale_w=0.0, length_scale=1.0)[0]
    n = min(y1.shape[-1], y2.shape[-1])
    diff = float((y1[..., :n] - y2[..., :n]).abs().max())
    print(f"copied {copied}/{len(new_sd)} params, folded {folds} biases; lengths {y1.shape[-1]} vs {y2.shape[-1]}, max diff {diff:.2e}")

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": single.state_dict(), "design": design["name"]}, a.out)
    print("saved", a.out, round(Path(a.out).stat().st_size / 1e6), "MB")


if __name__ == "__main__":
    main()
