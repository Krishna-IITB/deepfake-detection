#!/bin/bash
#SBATCH --job-name=deepfake-b4
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/deepfake_%j.out
#SBATCH --error=logs/deepfake_%j.err

# ============================================================
# PRAJNA SLURM script — train EfficientNet-B4 deepfake detector
# Submit from repo root: sbatch slurm/train_b4.sh
# ============================================================

set -euo pipefail

echo "==> Job:   $SLURM_JOB_ID on $(hostname)"
echo "==> Start: $(date)"
echo "==> CWD:   $(pwd)"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# --- env activation -----------------------------------------
# adjust the path if your conda lives elsewhere
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate deepfake

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'device', torch.cuda.get_device_name(0))"

# --- training -----------------------------------------------
export PYTHONUNBUFFERED=1

python -m src.train \
    --train-dir data/frames/train \
    --val-dir   data/frames/val \
    --backbone  efficientnet_b4 \
    --img-size  380 \
    --epochs    15 \
    --batch-size 64 \
    --num-workers 8 \
    --lr 3e-4

echo "==> End: $(date)"
