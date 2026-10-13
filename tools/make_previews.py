"""checkpoint → ONNX → მოსასმენი ნიმუშები პანელისთვის (training/runs/<slot>/previews/kNN), training/.venv-ით.

    training/.venv/bin/python tools/make_previews.py <checkpoint> --label "Google Cloud" [--slot base_ka]
"""

import argparse
import json
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from app.config import api_url, cfg, expand, path as cpath  # noqa: E402,F401
sys.path.insert(0, str(ROOT / "tools"))
from common import PHRASES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint")
    ap.add_argument("--label", required=True)
    ap.add_argument("--slot", default="base_ka")
    a = ap.parse_args()
    import torch
    epoch = torch.load(a.checkpoint, map_location="cpu", weights_only=False, mmap=True)["epoch"]
    run = cpath("runs") / a.slot
    job = json.loads((ROOT / "data" / "kaggle_jobs" / f"{a.slot}.json").read_text())
    cfg = run / "kaggle_out" / "config.json"
    cpath("tmp").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=cpath("tmp")) as td:
        onnx = Path(td) / "m.onnx"
        subprocess.run([sys.executable, "-m", "piper.train.export_onnx", "--checkpoint", a.checkpoint,
                        "--output-file", str(onnx)], check=True, capture_output=True, cwd=ROOT / "training")
        (Path(td) / "m.onnx.json").write_text(cfg.read_text())
        from piper import PiperVoice, SynthesisConfig
        voice = PiperVoice.load(onnx)
        smap = json.loads(cfg.read_text())["speaker_id_map"]
        prev = run / "previews"
        n = len(list(prev.glob("k*"))) + 1
        pdir = prev / f"k{n:02d}"
        pdir.mkdir(parents=True)
        files = []
        for spk in job["preview_speakers"]:
            for i, text in enumerate(PHRASES, 1):
                name = f"{spk}_{i}.wav"
                with wave.open(str(pdir / name), "wb") as w:
                    voice.synthesize_wav(text, w, syn_config=SynthesisConfig(speaker_id=smap[spk]))
                files.append({"file": f"{pdir.name}/{name}", "text": text, "speaker": spk})
    (pdir / "meta.json").write_text(json.dumps({"label": a.label, "epoch": epoch, "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                                "files": files}, ensure_ascii=False, indent=1), encoding="utf-8")
    print("OK", pdir, "epoch", epoch)


if __name__ == "__main__":
    main()
