#!/bin/bash
# Google Cloud-ის მანქანაზე: წვრთნა უახლესი checkpoint-იდან (systemd: aivoice-train.service).
# checkpoint-ები: /data/run/lightning_logs/version_*/checkpoints/local-e*-s*.ckpt (ორი უახლესი)
# შედეგი: /data/DONE, როცა max_epochs მიღწეულია. ლოგი: /data/train.log
set -uo pipefail
[ -f /data/ready ] || { echo "setup ჯერ არ დასრულებულა"; exit 1; }
V=/opt/aivoice/venv/bin
J=/data/job.json
jq() { $V/python -c "import json,sys; print(json.load(open('$J'))['$1'])"; }

# უახლესი გამართული checkpoint (დაზიანებული — მაგ. შეწყვეტა ჩაწერისას — გამოტოვდება)
CKPT=$($V/python - <<'EOF'
import glob, torch
best = None
for p in glob.glob("/data/run/lightning_logs/version_*/checkpoints/local-*.ckpt") + ["/data/upload/resume.ckpt"]:
    try:
        s = torch.load(p, map_location="cpu", weights_only=False, mmap=True)["global_step"]
    except Exception:
        continue
    if best is None or s > best[0]:
        best = (s, p)
print(best[1] if best else "")
EOF
)
[ -n "$CKPT" ] || { echo "checkpoint ვერ ვიპოვე"; exit 1; }
echo "==== $(date -Is) resume $CKPT" >> /data/train.log

export PIPER_AUDIO_FROM_SOURCE=1 PIPER_CKPT_EVERY_STEPS=$(jq ckpt_every_steps)
$V/python -m piper.train fit \
  --data.voice_name base_ka \
  --data.csv_path /data/run/metadata.csv \
  --data.audio_dir /data/base_ka_v1/audio \
  --model.sample_rate 22050 \
  --data.espeak_voice ka \
  --data.cache_dir /data/run/cache \
  --data.config_path /data/run/config.json \
  --data.batch_size $(jq batch) \
  --data.num_workers $(jq workers) \
  --data.validation_split 0.01 \
  --data.num_test_examples 2 \
  --data.trim_silence false \
  --model.mos_metric none \
  --model.num_speakers $(jq num_speakers) \
  --trainer.accelerator gpu \
  --trainer.devices 1 \
  --trainer.max_epochs $(jq max_epochs) \
  --trainer.default_root_dir /data/run \
  --ckpt_path "$CKPT" >> /data/train.log 2>&1
rc=$?
echo "==== $(date -Is) exit $rc" >> /data/train.log
if [ $rc -eq 0 ]; then
  # დასრულდა: ბოლო მდგომარეობა ცალკე ფაილად (ეპოქის ბოლოს checkpoint-ი შეიძლება არ იყოს)
  touch /data/DONE
  systemctl disable aivoice-train.service
fi
exit $rc
