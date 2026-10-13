# Kaggle-ზე გასაშვები წვრთნის სკრიპტი. tools/kaggle_train.py ავსებს __CONFIG__-ს და ატვირთავს.
# შედეგი /kaggle/working-ში: final.ckpt, <slot>.onnx(+.json), previews/*.wav, status.json, train.log
#
# CONFIG: slot, max_epochs, batch, devices (1|2), num_speakers, warmstart (ფაილის სახელი aivoice-base-ში),
#         audio_from_source (PIPER_AUDIO_FROM_SOURCE — ქეშში მხოლოდ ფონემები), trim_silence,
#         preview_speakers (multi-speaker-ზე: ვისი ხმით დამზადდეს ნიმუშები), phrases

import glob
import json
import os
import shutil
import subprocess
import sys
import time
import wave
from pathlib import Path

CFG = json.loads('''__CONFIG__''')
SLOT = CFG["slot"]
IN = Path("/kaggle/input")
WARM = next(p for p in IN.rglob(CFG["warmstart"]))  # საწყისი checkpoint — aivoice-base-ში ან ცალკე dataset-ში (init-*)
OVERLAY = next(IN.rglob("piper_overlay.tar"))  # ჩვენი შესწორებები Piper-ის train/-ზე + core.pyx (repo: piper_overlay/)
SLUG = SLOT.replace("_", "-")
VOICE = next(p for p in IN.rglob("metadata.csv") if f"aivoice-{SLUG}" in str(p)).parent
RESUME = next(iter(IN.rglob("resume.ckpt")), None)


def _max_time() -> str:
    """Kaggle-ის სესია 12 სთ-ია — ~1 სთ ვტოვებთ ექსპორტისთვის. Lightning-ის Timer გასულ დროს checkpoint-ში
    ინახავს და resume-ზე აგრძელებს (ლიმიტი კუმულატიურია, 2026-10-07). ლოკალურ წვრთნას Timer არ აქვს, ამიტომ
    სესიების თვლის ნაცვლად checkpoint-იდან ვკითხულობთ, რამდენი დრო აქვს უკვე დათვლილი, და ვუმატებთ 11 სთ-ს."""
    elapsed = 0.0
    if RESUME:
        import torch
        ck = torch.load(RESUME, map_location="cpu", weights_only=False, mmap=True)
        for k, v in (ck.get("callbacks") or {}).items():
            if "Timer" in str(k):
                elapsed = float((v.get("time_elapsed") or {}).get("train") or 0)
        del ck
    t = int(elapsed + 11 * 3600)
    print("max_time: elapsed", round(elapsed / 3600, 2), "h", flush=True)
    return f"{t // 86400:02d}:{t % 86400 // 3600:02d}:{t % 3600 // 60:02d}:{t % 60:02d}"


MAX_TIME = _max_time()
OUT = Path("/kaggle/working")
RUN = Path("/kaggle/temp/run")
STATUS = {"voice": SLOT, "device": "kaggle", "max_epochs": CFG["max_epochs"], "batch": CFG["batch"],
          "devices": CFG["devices"], "started": time.strftime("%Y-%m-%dT%H:%M:%S")}


def status(**kw):
    STATUS.update(kw, updated=time.strftime("%Y-%m-%dT%H:%M:%S"))
    (OUT / "status.json").write_text(json.dumps(STATUS, ensure_ascii=False, indent=1))


def sh(cmd, **kw):
    print("+", cmd if isinstance(cmd, str) else " ".join(map(str, cmd)), flush=True)
    return subprocess.run(cmd, check=True, shell=isinstance(cmd, str), **kw)


status(status="installing")
# ოფიციალური Piper (PyPI-ზე train/ უკვე შედის) + ჩვენი ფაილები ზემოდან — კომპილაცია არ სჭირდება
sh([sys.executable, "-m", "pip", "install", "-q", "piper-tts==1.8.0",
    "lightning>=2,<3", "tensorboard", "tensorboardX", "jsonargparse[signatures]>=4.27.7",
    "onnx", "onnxscript", "pysilero-vad>=2.1,<3", "cython>=3", "librosa<1", "soundfile"])

import tarfile  # noqa: E402

import piper  # noqa: E402

with tarfile.open(OVERLAY) as t:
    t.extractall(Path(piper.__file__).parent.parent)  # piper/train/... ჩანაცვლდება
ma = Path(piper.__file__).parent / "train" / "vits" / "monotonic_align"
sh(f"cd {ma} && cythonize -i core.pyx && mkdir -p monotonic_align && mv core*.so monotonic_align/")
for marker, f in [("aivoice: with DDP", "train/vits/dataset.py"), ("aivoice: resuming", "train/vits/lightning.py")]:
    assert marker in (Path(piper.__file__).parent / f).read_text(), f"overlay: {f}"

# აუდიო: თუ Kaggle-მა audio.zip არ გახსნა — ვხსნით /kaggle/temp-ში
AUDIO = VOICE / "audio"
if not any(AUDIO.rglob("*.flac")):
    import zipfile
    AUDIO = Path("/kaggle/temp/audio")
    AUDIO.mkdir(parents=True, exist_ok=True)
    zipfile.ZipFile(next(VOICE.rglob("audio*.zip"))).extractall(AUDIO)
AUDIO = next(p.parent for p in AUDIO.rglob("*.flac"))
# Piper-ის csv.reader ჩვეულებრივ ბრჭყალს (") ველის დასაწყისად თვლის და ათასობით სტრიქონს ერთ ველად კითხულობს
# ("field larger than field limit") — ტექსტიდან ბრჭყალებს ვიღებთ (2026-10-06-ის შეცდომა)
META = Path("/kaggle/temp/metadata.csv")
META.parent.mkdir(parents=True, exist_ok=True)
META.write_text("\n".join(l.replace('"', "") for l in (VOICE / "metadata.csv").read_text(encoding="utf-8").splitlines()) + "\n",
                encoding="utf-8")
print("voice data:", VOICE, "audio:", AUDIO, "resume:", RESUME, flush=True)

import torch  # noqa: E402

ngpu = torch.cuda.device_count()
devices = min(CFG["devices"], ngpu)
print("torch", torch.__version__, "gpus", ngpu, [torch.cuda.get_device_name(i) for i in range(ngpu)], flush=True)
clips = sum(1 for _ in open(META, encoding="utf-8"))
status(status="training", clips=clips, gpu=f"{devices}× {torch.cuda.get_device_name(0)}", devices=devices)

cmd = [sys.executable, "-m", "piper.train", "fit",
       "--data.voice_name", SLOT,
       "--data.csv_path", str(META),
       "--data.audio_dir", str(AUDIO),
       "--model.sample_rate", "22050",
       "--data.espeak_voice", "ka",
       "--data.cache_dir", "/kaggle/temp/cache",
       "--data.config_path", str(OUT / "config.json"),
       "--data.batch_size", str(CFG["batch"]),
       "--data.num_workers", "3",
       "--data.validation_split", "0.01" if CFG["num_speakers"] > 1 else "0.03",
       "--data.num_test_examples", "2",
       "--data.trim_silence", "true" if CFG.get("trim_silence", True) else "false",
       "--model.mos_metric", "none",
       "--model.num_speakers", str(CFG["num_speakers"]),
       "--trainer.accelerator", "gpu",
       "--trainer.devices", str(devices),
       "--trainer.max_epochs", str(CFG["max_epochs"]),
       "--trainer.max_time", MAX_TIME,
       "--trainer.default_root_dir", str(RUN)]
if devices > 1:
    # VITS GAN: გენერატორის ნაბიჯში დისკრიმინატორის პარამეტრები არ მონაწილეობს — DDP-ს ეს უნდა იცოდეს
    cmd += ["--trainer.strategy", "ddp_find_unused_parameters_true"]
cmd += ["--ckpt_path", str(RESUME)] if RESUME else ["--model.warmstart_ckpt", str(WARM)]

env = dict(os.environ)
if CFG.get("audio_from_source"):
    env["PIPER_AUDIO_FROM_SOURCE"] = "1"

# მონაცემების მომზადება (ფონემები, ქეში) ჯერ ერთ პროცესში: DDP-ში ამას rank 0 აკეთებს, და თუ ჩავარდა,
# rank 1 30 წუთი ელოდება NCCL timeout-ს (2026-10-06). აქ შეცდომა წამებში გამოჩნდება.
status(status="preparing")
prep = f"""
from piper.train.vits.dataset import VitsDataModule
dm = VitsDataModule(csv_path={str(META)!r}, cache_dir="/kaggle/temp/cache", espeak_voice="ka",
    config_path={str(OUT / "config.json")!r}, voice_name={SLOT!r}, audio_dir={str(AUDIO)!r},
    num_speakers={CFG["num_speakers"]}, trim_silence={bool(CFG.get("trim_silence", True))}, batch_size={CFG["batch"]})
dm.prepare_data()
print("prepared", flush=True)
"""
with open(OUT / "prepare.log", "w") as log:
    rc = subprocess.run([sys.executable, "-c", prep], stdout=log, stderr=subprocess.STDOUT, env=env).returncode
if rc != 0:
    status(status="failed", error=f"მონაცემების მომზადება ჩავარდა (exit {rc}) — იხ. prepare.log")
    sys.exit(1)
status(status="training")

t0 = time.time()
with open(OUT / "train.log", "w") as log:
    rc = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, env=env).returncode
lasts = sorted(glob.glob(str(RUN / "lightning_logs/*/checkpoints/last.ckpt")), key=os.path.getmtime)
if rc != 0 or not lasts:
    status(status="failed", error=f"exit code {rc}", hours=round((time.time() - t0) / 3600, 2))
    sys.exit(1)

final = OUT / "final.ckpt"
shutil.copy(os.path.realpath(lasts[-1]), final)
epoch = torch.load(final, map_location="cpu", weights_only=True, mmap=True)["epoch"]
shutil.rmtree(RUN, ignore_errors=True)

onnx = OUT / f"{SLOT}.onnx"
sh([sys.executable, "-m", "piper.train.export_onnx", "--checkpoint", str(final), "--output-file", str(onnx)])
shutil.copy(OUT / "config.json", str(onnx) + ".json")

from piper import PiperVoice, SynthesisConfig  # noqa: E402

voice = PiperVoice.load(onnx)
smap = json.loads((OUT / "config.json").read_text()).get("speaker_id_map") or {}
speakers = [s for s in CFG.get("preview_speakers", []) if s in smap] or [None]
(OUT / "previews").mkdir(exist_ok=True)
for spk in speakers:
    for i, text in enumerate(CFG["phrases"], 1):
        name = f"{spk}_{i}.wav" if spk else f"{i}.wav"
        with wave.open(str(OUT / "previews" / name), "wb") as w:
            voice.synthesize_wav(text, w, syn_config=SynthesisConfig(speaker_id=smap[spk]) if spk else None)

# checkpoint ყოველ 2 ეპოქაში ინახება — ბოლო შენახული შეიძლება max_epochs-ზე ნაკლები იყოს, თუმცა წვრთნა დასრულდა
done = epoch + 1 >= CFG["max_epochs"] or "max_epochs=" in (OUT / "train.log").read_text(errors="replace")
status(status="done" if done else "partial", epoch=epoch, hours=round((time.time() - t0) / 3600, 2),
       preview_speakers=[s for s in speakers if s])
print("FINISHED epoch", epoch, "done" if done else "partial (resume needed)", flush=True)
