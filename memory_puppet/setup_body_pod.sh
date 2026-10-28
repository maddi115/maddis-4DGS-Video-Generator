#!/bin/bash
# setup_body_pod.sh - install SAM 3D Body (Meta) on a RunPod Linux GPU pod.
# Run in the pod terminal:   bash setup_body_pod.sh
# Safe to run again: it skips what is already done.
set -e
cd /workspace

echo "== 1/5 system tools"
apt-get update -qq && apt-get install -y -qq ffmpeg libegl1 libgl1 libosmesa6 git >/dev/null
python -c "import torch; print('   torch', torch.__version__, '| GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NONE')"

echo "== 2/5 SAM 3D Body code"
[ -d sam-3d-body ] || git clone -q https://github.com/facebookresearch/sam-3d-body
cd sam-3d-body

echo "== 3/5 Python packages (one at a time, so one bad package can't stop the rest)"
for p in pytorch-lightning pyrender opencv-python yacs scikit-image einops timm dill pandas rich \
         hydra-core hydra-submitit-launcher hydra-colorlog pyrootutils webdataset networkx==3.2.1 \
         roma joblib seaborn jsonlines xtcocotools loguru optree fvcore pycocotools huggingface_hub \
         cython iopath omegaconf cloudpickle tabulate termcolor matplotlib tqdm pillow wandb appdirs scipy; do
  pip install -q "$p" >/dev/null 2>&1 && echo "   ok   $p" || echo "   SKIP $p"
done

echo "== 4/5 Detectron2 (person finder) - this one takes 5-10 minutes to build"
python -c "import detectron2" 2>/dev/null && echo "   already installed" || \
  pip install -q 'git+https://github.com/facebookresearch/detectron2.git@a1ce2f9' --no-build-isolation --no-deps

echo "== 5/5 model files (needs your Hugging Face login + approved access)"
if [ -f checkpoints/sam-3d-body-dinov3/model.ckpt ]; then
  echo "   already downloaded"
elif hf auth whoami >/dev/null 2>&1; then
  hf download facebook/sam-3d-body-dinov3 --local-dir checkpoints/sam-3d-body-dinov3
else
  echo ""
  echo "   NOT LOGGED IN to Hugging Face. Do this, then run this script again:"
  echo "     hf auth login        (paste your token HERE in the pod, never in chat)"
  exit 1
fi

python -c "import sam_3d_body, detectron2; print('\nREADY: SAM 3D Body installed')"
