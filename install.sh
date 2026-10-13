#!/usr/bin/env bash
# aivoice — დაყენება (macOS / Linux).
#
#   ./install.sh                 # პანელი + ხმის შემქმნელი (.venv)
#   ./install.sh --training      # + ხმების მომზადება/ექსპორტი/შემოწმება (training/.venv, torch)
#   ./install.sh --data ~/aivoice-data   # მონაცემების საქაღალდე (ნაგულისხმევი ~/aivoice-data)
#
# შემდეგ: ./aivoice serve  →  http://127.0.0.1:8200  (პირველად გაიხსნება „პარამეტრები“ — დაყენების ოსტატი)
set -euo pipefail
cd "$(dirname "$0")"

TRAINING=0
DATA=""
while [ $# -gt 0 ]; do
  case "$1" in
    --training) TRAINING=1 ;;
    --data) DATA="$2"; shift ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) echo "უცნობი პარამეტრი: $1"; exit 1 ;;
  esac
  shift
done

say()  { printf "\n\033[1;34m==> %s\033[0m\n" "$*"; }
ok()   { printf "   \033[32m✓\033[0m %s\n" "$*"; }
warn() { printf "   \033[33m!\033[0m %s\n" "$*"; }
die()  { printf "\n\033[31m✗ %s\033[0m\n" "$*"; exit 1; }

OS="$(uname -s)"

# ---------- 1. სისტემური ხელსაწყოები ----------
say "სისტემური ხელსაწყოები ($OS)"
if [ "$OS" = "Darwin" ]; then
  command -v brew >/dev/null || die "Homebrew არ არის. დააყენეთ: https://brew.sh  და გაუშვით ./install.sh თავიდან."
  for pkg in python@3.12 ffmpeg espeak-ng; do
    if brew list --formula "$pkg" >/dev/null 2>&1; then ok "$pkg"; else say "brew install $pkg"; brew install "$pkg"; fi
  done
  PY="$(brew --prefix python@3.12)/bin/python3.12"
else
  PY=""
  for c in python3.13 python3.12 python3.11 python3; do
    if command -v "$c" >/dev/null && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 11))'; then PY="$(command -v "$c")"; break; fi
  done
  [ -n "$PY" ] || die "საჭიროა Python 3.11+ (მაგ. Ubuntu: sudo apt install python3.12 python3.12-venv)"
  command -v ffmpeg >/dev/null && ok "ffmpeg" || die "ffmpeg არ არის (Ubuntu: sudo apt install ffmpeg · Fedora/Alma: sudo dnf install ffmpeg)"
  command -v espeak-ng >/dev/null && ok "espeak-ng" || warn "espeak-ng არ არის — არასავალდებულოა (Piper-ს თავისი აქვს)"
fi
ok "Python: $PY ($("$PY" --version))"

# ---------- 2. პანელის გარემო ----------
say "პანელის გარემო (.venv)"
[ -x .venv/bin/python ] || "$PY" -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt
ok "ბიბლიოთეკები დაყენდა"

# ---------- 3. კონფიგურაცია ----------
say "კონფიგურაცია (aivoice.toml)"
if [ -f aivoice.toml ]; then
  ok "უკვე არსებობს — არ ვცვლი"
else
  DATA="${DATA:-$HOME/aivoice-data}"
  DATA="${DATA/#\~/$HOME}"
  mkdir -p "$DATA"
  printf '# aivoice — კონფიგურაცია (პანელი → „პარამეტრები“ ან ხელით). იხ. app/config.py\n\n[paths]\ndata_root = "%s"\n' "$DATA" > aivoice.toml
  ok "მონაცემების საქაღალდე: $DATA (შეცვლა: პანელი → პარამეტრები)"
fi
.venv/bin/python -c "import sys; sys.path.insert(0, '.'); from app.config import ensure_dirs; ensure_dirs()"

# ---------- 4. წვრთნის/მომზადების გარემო (არასავალდებულო) ----------
if [ "$TRAINING" = 1 ]; then
  say "წვრთნის გარემო (training/.venv)"
  [ -x training/.venv/bin/python ] || { mkdir -p training; "$PY" -m venv training/.venv; }
  TP=training/.venv/bin/pip
  $TP install -q --upgrade pip
  if [ "$OS" = "Darwin" ]; then
    $TP install -q torch torchaudio                      # Apple Silicon: MPS
  elif command -v nvidia-smi >/dev/null && nvidia-smi >/dev/null 2>&1; then
    $TP install -q torch torchaudio --index-url https://download.pytorch.org/whl/cu126
  else
    $TP install -q torch torchaudio --index-url https://download.pytorch.org/whl/cpu
  fi
  $TP install -q -r requirements-training.txt
  # ჩვენი შესწორებები Piper-ის train/-ზე (piper_overlay/) + monotonic_align-ის აწყობა
  SITE="$(training/.venv/bin/python -c 'import piper, os; print(os.path.dirname(os.path.dirname(piper.__file__)))')"
  cp -R piper_overlay/piper/. "$SITE/piper/"
  HERE="$(pwd)"
  ( cd "$SITE/piper/train/vits/monotonic_align" && "$HERE/training/.venv/bin/cythonize" -i core.pyx >/dev/null \
      && mkdir -p monotonic_align && mv -f core*.so monotonic_align/ )
  training/.venv/bin/python -c "import piper.train.vits.monotonic_align as m, torch; print('   torch', torch.__version__)"
  ok "წვრთნის გარემო მზადაა"
fi

say "მზადაა!"
cat <<EOF
   პანელის გაშვება:    ./aivoice serve        →  http://127.0.0.1:8200
   მოდელების ჩამოტვირთვა (პირველად):  ./aivoice models pull
   ყველა ბრძანება:      ./aivoice help
EOF
