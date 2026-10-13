import logging

import torch
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.cli import LightningCLI

from .vits.dataset import VitsDataModule
from .vits.lightning import VitsModel

_LOGGER = logging.getLogger(__package__)

# Checkpoint-selection signal. The raw mel L1 ("val_mel") tracks reconstruction
# fidelity and trends down as the model improves, so it is a reasonable key for
# keeping the best checkpoints. It is NOT, however, used to stop training: mel
# L1 saturates early in VITS while the adversarial (GAN) losses keep removing
# audible artifacts, so an early-stop on val_mel fires well before the audio is
# clean. Instead we train on a fixed/open-ended schedule and let the user pick a
# final checkpoint by listening (and/or by "val_mos"). Override from the
# LightningCLI config if you want different behavior.
_MONITOR = "val_mel"

_DEFAULT_CALLBACKS = [
    # aivoice: fewer/smaller checkpoint writes — the system NVMe overheats under
    # sustained 846 MB writes (suspected cause of BSODs, 2026-09-29). top_k=2,
    # every 2 epochs, and last.ckpt as a symlink instead of a second full copy.
    ModelCheckpoint(
        monitor=_MONITOR,
        mode="min",
        save_top_k=2,
        every_n_epochs=2,
        save_last="link",
        filename="epoch={epoch}-val_mel={val_mel:.4f}",
        auto_insert_metric_name=False,
    ),
    # Also keep the best-by-perceptual-quality checkpoints. "val_mos" (UTMOS)
    # tracks audible artifacts that mel L1 misses, so its winner is often the
    # better-sounding model. If the MOS predictor can't be loaded, val_mos is
    # never logged and this callback simply finds nothing to save (Lightning
    # warns once and skips) -- it does not interfere with the val_mel/last
    # checkpoints above. No save_last here to avoid clobbering last.ckpt.
    ModelCheckpoint(
        monitor="val_mos",
        mode="max",
        save_top_k=5,
        save_last=False,
        filename="epoch={epoch}-val_mos={val_mos:.4f}",
        auto_insert_metric_name=False,
    ),
]


class VitsLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.batch_size", "model.batch_size")
        parser.link_arguments("data.num_symbols", "model.num_symbols")
        parser.link_arguments("model.num_speakers", "data.num_speakers")
        parser.link_arguments("model.sample_rate", "data.sample_rate")
        parser.link_arguments("model.filter_length", "data.filter_length")
        parser.link_arguments("model.hop_length", "data.hop_length")
        parser.link_arguments("model.win_length", "data.win_length")
        parser.link_arguments("model.segment_size", "data.segment_size")


def _mos_disabled(argv) -> bool:
    """--model.mos_metric none/null => the val_mos callback would raise at epoch end."""
    for i, arg in enumerate(argv):
        value = None
        if arg.startswith("--model.mos_metric="):
            value = arg.split("=", 1)[1]
        elif arg == "--model.mos_metric" and i + 1 < len(argv):
            value = argv[i + 1]
        if value is not None:
            return value.lower() in ("none", "null", "")
    return False


def main():
    import sys

    logging.basicConfig(level=logging.INFO)
    callbacks = _DEFAULT_CALLBACKS
    if _mos_disabled(sys.argv[1:]):
        # Lightning raises MisconfigurationException when a monitored key is
        # never logged, so drop the val_mos checkpoint callback entirely.
        callbacks = [c for c in callbacks if getattr(c, "monitor", None) != "val_mos"]
    import os
    every_steps = int(os.environ.get("PIPER_CKPT_EVERY_STEPS") or 0)
    if every_steps > 0:
        # aivoice (ლოკალური წვრთნა, დენი ხშირად ითიშება): checkpoint ყოველ N ნაბიჯში, ეპოქის შუაშიც.
        # ორი უახლესი ინახება — თუ გათიშვამ ერთი ჩაწერისას გააფუჭა, მეორე რჩება.
        # train_time_interval არ გამოდგება: VITS manual optimization-ია და Lightning მაშინ მხოლოდ
        # every_n_train_steps-ს ამოწმებს (2026-10-08). global_step ბატჩზე 2-ით იზრდება (2 ოპტიმიზატორი) —
        # N ლუწი უნდა იყოს.
        callbacks = [ModelCheckpoint(
            monitor="step", mode="max", save_top_k=2, save_last=False,
            every_n_train_steps=every_steps,
            filename="local-e{epoch}-s{step}", auto_insert_metric_name=False,
        ), ModelCheckpoint(
            # ეპოქის ბოლოს — მათ შორის ბოლო ეპოქისაც (ნაბიჯების checkpoint-ი წვრთნის ბოლოს არ ინახება)
            monitor="step", mode="max", save_top_k=1, save_last=False, every_n_epochs=1,
            filename="local-end-e{epoch}-s{step}", auto_insert_metric_name=False,
        )]
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.deterministic = False
    _cli = VitsLightningCLI(  # noqa: ignore=F841
        VitsModel,
        VitsDataModule,
        trainer_defaults={"max_epochs": -1, "callbacks": callbacks},
    )


# -----------------------------------------------------------------------------


if __name__ == "__main__":
    main()
