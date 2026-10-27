#!/bin/bash
# One-time setup for a fresh RunPod pod. Safe to run again.
set -e
REPO=/workspace/maddis-4DGS-Video-Generator
cd $REPO

echo "[1/6] Submodules (ml-sharp)..."
git submodule update --init --recursive

echo "[2/6] Python packages..."
pip install -r requirements-headless.txt
python3 -c "import cv2" 2>/dev/null || { pip uninstall -y opencv-python; pip install opencv-python-headless; }

echo "[3/6] Node 22..."
if ! command -v node >/dev/null || [ "$(node -v | cut -d. -f1 | tr -d v)" -lt 22 ]; then
  curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
  apt-get install -y nodejs
fi

echo "[4/6] Chronosplat..."
[ -d /workspace/Chronosplat ] || git clone https://github.com/tcwilsonart/Chronosplat.git /workspace/Chronosplat
(cd /workspace/Chronosplat && npm install)

echo "[5/6] Shortcuts..."
chmod +x $REPO/*.sh
for c in check:progress live:live start_run:startjob stop_run:stopjob convert_on_pod:convertjob convert_check:convertprogress convert_live:convertlive; do
  ln -sf "$REPO/${c%%:*}.sh" "/usr/local/bin/${c##*:}"
done

echo "[6/6] Terminal + git..."
grep -q "maddis-pod" ~/.bashrc || echo "PS1='\[\e[1;38;5;45m\]maddis-pod\[\e[0m\]:\w# '" >> ~/.bashrc
grep -q "alias claer" ~/.bashrc || echo "alias claer=clear" >> ~/.bashrc
grep -q "alias tips" ~/.bashrc || echo "alias tips='cat $REPO/TIPS.md'" >> ~/.bashrc
grep -q "Type tips" ~/.bashrc || echo "echo -e '\e[1;38;5;45mType tips to see all commands.\e[0m'" >> ~/.bashrc
git config --global credential.helper store

echo
echo "=============================================="
echo " Setup done. Run: source ~/.bashrc"
echo "=============================================="
echo
cat $REPO/TIPS.md
