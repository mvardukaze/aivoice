#!/bin/bash
# Google Cloud-ის მანქანაზე ერთჯერადი მომზადება (tools/gcp_train.py push უშვებს, ssh-ით, root-ად).
# მოელის /data/upload-ში: base_ka_v1.tar, piper_tts-*.whl, core.pyx, train_overlay.tar, resume.ckpt, job.json
set -euo pipefail
U=/data/upload
mkdir -p /data/run

# 1. dataset
if [ ! -f /data/base_ka_v1/metadata.csv ]; then
  tar -xf $U/base_ka_v1.tar -C /data
fi
sed 's/"//g' /data/base_ka_v1/metadata.csv > /data/run/metadata.csv  # ბრჭყალები Piper-ის csv.reader-ს არღვევს

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
$V/pip install -q $U/piper_tts-*.whl "lightning>=2,<3" tensorboard "jsonargparse[signatures]>=4.27.7" \
  onnx onnxscript "pysilero-vad>=2.1,<3" "cython>=3" "librosa<1" soundfile
SITE=$($V/python -c "import piper, os; print(os.path.dirname(piper.__file__))")
tar -xf $U/train_overlay.tar -C $SITE          # train/__main__.py, vits/dataset.py, vits/lightning.py, export_onnx.py
MA=$SITE/train/vits/monotonic_align
cp $U/core.pyx $MA/core.pyx
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
systemctl daemon-reload
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
$V/python -c "import torch, piper, lightning; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'lightning', lightning.__version__)"
touch /data/ready
echo SETUP_OK
