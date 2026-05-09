#!/bin/bash
#SBATCH --job-name=deepfake-eval
#SBATCH --partition=a40
#SBATCH --qos=a40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/eval_%j.out
#SBATCH --error=logs/eval_%j.err

set -euo pipefail

echo "==> Job: $SLURM_JOB_ID on $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate deepfake

python -m src.evaluate \
    --ckpt checkpoints/best.pt \
    --data-dir data/frames/test \
    --out-dir reports

echo "==> Done. See reports/ for metrics + plots."
