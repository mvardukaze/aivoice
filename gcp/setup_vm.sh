#!/bin/bash
# Google Cloud-ის მანქანაზე ერთჯერადი მომზადება (tools/gcp_train.py push უშვებს, ssh-ით, root-ად).
# მოელის /data/upload-ში: <dataset>.tar, piper_overlay.tar, resume.ckpt ან warmstart.ckpt, job.json
set -euo pipefail
U=/data/upload
mkdir -p /data/run

# 1. dataset (job.json → "dataset", ნაგულისხმევად base_ka_v1)
DS=$(python3 -c "import json; print(json.load(open('$U/job.json')).get('dataset', 'base_ka_v1'))")
if [ ! -f /data/$DS/metadata.csv ]; then
  tar -xf $U/$DS.tar -C /data
fi
sed 's/"//g' /data/$DS/metadata.csv > /data/run/metadata.csv  # ბრჭყალები Piper-ის csv.reader-ს არღვევს

# 2. python: image-ის torch + ჩვენი piper (abi3 wheel) + შესწორებული train/*.py
PY=$(for p in /opt/conda/bin/python /usr/bin/python3 /usr/local/bin/python3; do
       [ -x $p ] && $p -c "import torch" 2>/dev/null && echo $p && break; done)
[ -n "$PY" ] || { echo "torch-იანი python ვერ ვიპოვე"; exit 1; }
if ! $PY -m ensurepip --version >/dev/null 2>&1; then
  apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq "python3.$($PY -c 'import sys; print(sys.version_info.minor)')-venv" >/dev/null
fi
if [ ! -x /opt/aivoice/venv/bin/pip ]; then
  rm -rf /opt/aivoice/venv
  $PY -m venv --system-site-packages /opt/aivoice/venv
fi
V=/opt/aivoice/venv/bin
$V/pip install -q "piper-tts==1.8.0" "lightning>=2,<3" tensorboard "jsonargparse[signatures]>=4.27.7" \
  onnx onnxscript "pysilero-vad>=2.1,<3" "cython>=3" "librosa<1" soundfile
SITE=$($V/python -c "import piper, os; print(os.path.dirname(os.path.dirname(piper.__file__)))")
tar -xf $U/piper_overlay.tar -C $SITE          # piper/train/... (ჩვენი შესწორებები) + core.pyx
MA=$SITE/piper/train/vits/monotonic_align
(cd $MA && $V/cythonize -i core.pyx >/dev/null && mkdir -p monotonic_align && mv core*.so monotonic_align/)

# 3. წვრთნის სერვისი — ჩართვისას (spot-ის შეწყვეტის შემდეგაც) თავისით აგრძელებს
cp $U/train_vm.sh /opt/aivoice/train_vm.sh
cp $U/job.json /data/job.json
chmod +x /opt/aivoice/train_vm.sh
cat > /etc/systemd/system/aivoice-train.service <<'EOF'
[Unit]
Description=aivoice base model training (resumes from newest checkpoint)
After=network-online.target

[Service]
ExecStart=/opt/aivoice/train_vm.sh
Restart=on-failure
RestartSec=60

[Install]
WantedBy=multi-user.target
EOF
# 4. უქმად დარჩენილი მანქანის გათიშვა (gcp/autostop.sh)
install -m 755 $U/autostop.sh /opt/aivoice/autostop.sh
printf '[Unit]\nDescription=aivoice: shut down idle VM after training\n[Service]\nExecStart=/opt/aivoice/autostop.sh\nRestart=always\n[Install]\nWantedBy=multi-user.target\n' \
  > /etc/systemd/system/aivoice-autostop.service
systemctl daemon-reload
systemctl enable --now aivoice-autostop.service
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$V/python -c "import torch, piper, lightning; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'lightning', lightning.__version__)"
touch /data/ready
echo SETUP_OK
