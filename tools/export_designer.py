"""ხმის შემქმნელისთვის: multi-speaker checkpoint → ONNX, რომელიც დინამიკის ნომრის ნაცვლად
დინამიკის ვექტორს (g, 512 რიცხვი) იღებს + ყველა დინამიკის ვექტორების ცხრილი.

გამოყენება (training/.venv-ით):
    training/.venv/bin/python tools/export_designer.py [<checkpoint>] [--out <paths.designer>]
შედეგი:
    <out>/designer.onnx       inputs: input, input_lengths, scales, g[1,gin]
    <out>/embeddings.npy      [n_speakers, gin] — speaker_id_map-ის რიგით
    <out>/config.json         Piper-ის კონფიგი (phoneme_id_map, speaker_id_map, ...)
    <out>/meta.json           საიდან, როდის, რომელი ეპოქა

ხერხი: model_g.emb_g იცვლება Identity-ით — infer() აკეთებს g = emb_g(sid).unsqueeze(-1), ასე რომ
"sid"-ის ადგილას მიწოდებული ვექტორი პირდაპირ g ხდება. Piper-ის კოდი უცვლელია.
"""

import argparse
import datetime as dt
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

from piper.train.vits.lightning import VitsModel

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.config import api_url, cfg, expand, path as cpath  # noqa: E402,F401
DEFAULT_CKPT = cpath("runs") / "base_ka" / "kaggle_out" / "final.ckpt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", nargs="?", default=str(DEFAULT_CKPT))
    ap.add_argument("--out", default=str(cpath("designer")))  # app/designer.py აქედან კითხულობს
    ap.add_argument("--config", help="Piper-ის config.json (ნაგულისხმევად checkpoint-ის გვერდით)")
    ap.add_argument("--source", default="kaggle", help="მეტამონაცემისთვის: kaggle | local")
    a = ap.parse_args()
    ckpt, out = Path(a.checkpoint), Path(a.out)
    tmp = out.with_name(out.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)

    torch.manual_seed(1234)
    model = VitsModel.load_from_checkpoint(ckpt, map_location="cpu", weights_only=False)
    g = model.model_g
    g.eval()
    assert g.n_speakers > 1, "საჭიროა multi-speaker მოდელი"
    with torch.no_grad():
        g.dec.remove_weight_norm()
        emb = g.emb_g.weight.detach().cpu().numpy().astype(np.float32)
    np.save(tmp / "embeddings.npy", emb)
    g.emb_g = torch.nn.Identity()

    def forward(text, text_lengths, scales, gvec):
        return g.infer(text, text_lengths, noise_scale=scales[0], length_scale=scales[1],
                       noise_scale_w=scales[2], sid=gvec)[0].unsqueeze(1)

    g.forward = forward
    x = torch.randint(0, g.n_vocab, (1, 50), dtype=torch.long)
    torch.onnx.export(
        g, (x, torch.LongTensor([50]), torch.FloatTensor([0.667, 1.0, 0.8]), torch.from_numpy(emb[:1])),
        str(tmp / "designer.onnx"), opset_version=15, dynamo=False,
        input_names=["input", "input_lengths", "scales", "g"], output_names=["output"],
        dynamic_axes={"input": {0: "batch_size", 1: "phonemes"}, "input_lengths": {0: "batch_size"},
                      "output": {0: "batch_size", 2: "time"}},
    )
    cfg = Path(a.config) if a.config else ckpt.parent / "config.json"
    shutil.copy2(cfg, tmp / "config.json")
    st = json.loads((ckpt.parent / "status.json").read_text()) if (ckpt.parent / "status.json").exists() else {}
    if st.get("epoch") is None:
        st["epoch"] = torch.load(ckpt, map_location="cpu", weights_only=False, mmap=True)["epoch"]
    (tmp / "meta.json").write_text(json.dumps({
        "checkpoint": str(ckpt), "epoch": st.get("epoch"), "speakers": int(emb.shape[0]), "gin": int(emb.shape[1]), "source": a.source,
        "exported": dt.datetime.now().isoformat(timespec="seconds")}, ensure_ascii=False, indent=1))
    shutil.rmtree(out, ignore_errors=True)
    tmp.rename(out)
    print("OK", out, emb.shape, "epoch", st.get("epoch"))


if __name__ == "__main__":
    main()
