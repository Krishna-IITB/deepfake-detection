# Training on PRAJNA HPC (IIT Bombay)

End-to-end runbook for training this project on PRAJNA's A40 partition.
Assumes you have a working PRAJNA account and SSH access.

---

## 1. Clone the repo on PRAJNA

From your **Mac**:
```bash
ssh <username>@prajna.iitb.ac.in   # whatever the actual hostname is
```

On the **PRAJNA login node**:
```bash
cd $HOME
git clone https://github.com/Krishna-IITB/deepfake-detection.git
cd deepfake-detection
mkdir -p logs
```

## 2. Create the conda env (one-time, ~5 min)

```bash
# load whatever conda module PRAJNA exposes; or use your existing miniconda
module load miniconda3 2>/dev/null || true   # if module system is used

conda create -n deepfake python=3.11 -y
conda activate deepfake

# install pytorch with CUDA support matching PRAJNA's driver
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# project deps (skip facenet-pytorch — only needed for predict.py inference)
pip install albumentations scikit-learn matplotlib tqdm tensorboard pillow opencv-python kaggle
```

Verify the env sees a GPU on a compute node (don't bother on the login node — login nodes typically have no GPU):
```bash
srun --partition=a40 --qos=a40 --gres=gpu:1 --time=00:05:00 --pty \
    python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
Expected: `True NVIDIA A40`.

## 3. Download the dataset on the login node

Compute nodes usually don't have internet — download on the login node, then read from shared storage during training.

```bash
# place kaggle.json on PRAJNA (copy from your Mac):
#    scp ~/.kaggle/kaggle.json <user>@prajna.iitb.ac.in:~/.kaggle/kaggle.json
mkdir -p ~/.kaggle
chmod 600 ~/.kaggle/kaggle.json

# download dataset into the project
mkdir -p data && cd data
kaggle datasets download -d xhlulu/140k-real-and-fake-faces
unzip -q 140k-real-and-fake-faces.zip
rm 140k-real-and-fake-faces.zip
cd ..

# wire pipeline paths
mkdir -p data/frames
ln -sfn "$(pwd)/data/real_vs_fake/real-vs-fake/train" data/frames/train
ln -sfn "$(pwd)/data/real_vs_fake/real-vs-fake/valid" data/frames/val
ln -sfn "$(pwd)/data/real_vs_fake/real-vs-fake/test"  data/frames/test
```

## 4. Submit the training job

```bash
sbatch slurm/train_b4.sh
```

Track it:
```bash
squeue -u $USER                       # see queue position / running state
tail -f logs/deepfake_<jobid>.out     # follow training output live
```

The script trains EfficientNet-B4, batch 64, 15 epochs. On A40 this should finish in ~1–1.5 hours (vs 30–45 min on Colab T4 only because A40 lets us use a much bigger batch).

Saves `checkpoints/best.pt` whenever val AUC improves.

## 5. Evaluate

```bash
sbatch slurm/eval.sh
```

When it finishes, `reports/metrics.json` + `reports/roc.png` + `reports/confusion_matrix.png` will exist in the repo dir.

## 6. Pull results back to your Mac

From your Mac:
```bash
mkdir -p ~/deepfake-results
scp <user>@prajna.iitb.ac.in:~/deepfake-detection/checkpoints/best.pt        ~/deepfake-results/
scp -r <user>@prajna.iitb.ac.in:~/deepfake-detection/reports                 ~/deepfake-results/
```

Then update the README "Results" table with the actual numbers, commit reports/ to git, and push.

---

## Memory + time tuning

If you want to push for higher accuracy:
- `--batch-size 96` on A40 (48GB has room for it)
- `--epochs 20` for a bit more convergence
- `--img-size 456` (matches B4's native input resolution)

If queue is slow:
- A T4 partition (if PRAJNA has one) is fine for B0/B3 — drop `--gres=gpu:1` and adjust partition
- Smaller dataset subset for a quick run: just symlink a subset of the train images

## Fallback if conda env breaks

If your existing `ddp` conda env already has a recent torch + torchvision + albumentations, you can skip step 2 entirely — change the `conda activate deepfake` line in the SLURM script to `conda activate ddp` and submit. Reuse over reinstall when it works.
