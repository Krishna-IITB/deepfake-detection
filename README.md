# Deepfake Detection with EfficientNet

A face-level deepfake / face-forgery detector built on top of an ImageNet-pretrained EfficientNet backbone. The model takes a still image or a short video clip and returns a **`REAL` / `FAKE` verdict** along with a calibrated probability score.

The codebase runs end-to-end on a **MacBook Pro M2** (Apple Silicon, MPS), as well as on CUDA GPUs and CPU — device selection is automatic.

---

## Highlights

- **Backbone:** EfficientNet-B4 (configurable: B0 / B3 / B4) — strong accuracy/parameter tradeoff for face-manipulation detection.
- **Face pipeline:** MTCNN-based face cropping (`facenet-pytorch`) so the classifier only sees normalized face crops, not raw frames.
- **Training:** AdamW + cosine LR schedule, BCE-with-logits loss, ImageNet-stat normalization, Albumentations augmentations (horizontal flip, brightness / hue jitter, Gaussian noise, random JPEG compression).
- **Evaluation:** accuracy, ROC-AUC, **EER**, F1, confusion matrix, and ROC-curve plots — what's actually reported in deepfake-detection literature.
- **Inference:** single image **and** video. For videos, frames are sampled (every Nth frame) and per-frame predictions are averaged.
- **Streamlit demo:** drag-and-drop a file to get a verdict — useful for showing the project off in interviews.
- **Apple Silicon ready:** MPS → CUDA → CPU device picker; MTCNN runs on CPU to dodge MPS fallbacks and keep things stable.

---

## Repo structure

```
deepfake-detection/
├── README.md
├── requirements.txt
├── LICENSE
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── data.py              # FaceFrameDataset + Albumentations transforms
│   ├── model.py             # EfficientNet backbone + binary head
│   ├── face_extractor.py    # MTCNN wrapper for face crop / video sampling
│   ├── train.py             # Training entry point (TensorBoard, checkpointing)
│   ├── evaluate.py          # Test-time metrics + ROC plot
│   ├── predict.py           # Inference on a single image or video
│   └── utils.py             # Seed, device picker, param counter
├── scripts/
│   └── extract_faces.py     # Convert a video corpus → face-crop dataset
└── app/
    └── app.py               # Streamlit demo
```

---

## Setup (MacBook Pro M2)

```bash
git clone https://github.com/<your-username>/deepfake-detection.git
cd deepfake-detection

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

Quick sanity check that PyTorch sees the M2 GPU:

```bash
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"
# expected: MPS available: True
```

---

## Datasets

The pipeline expects this on-disk layout for the actual training/eval images:

```
data/
├── train/
│   ├── real/   *.jpg
│   └── fake/   *.jpg
├── val/
│   ├── real/   *.jpg
│   └── fake/   *.jpg
└── test/
    ├── real/   *.jpg
    └── fake/   *.jpg
```

Common datasets that fit this pipeline (each requires its own access agreement, so apply via the official sources):

| Dataset | Real videos | Fake videos | Notes |
|---|---|---|---|
| **FaceForensics++** | 1000 YouTube | 4000 (DeepFakes / Face2Face / FaceSwap / NeuralTextures) | Standard benchmark; multiple compression levels (raw / c23 / c40). |
| **Celeb-DF v2** | 590 celebrity | 5639 high-quality fakes | Harder than FF++ — visually smoother fakes. |
| **DFDC (Preview / Full)** | thousands | thousands | Largest; very diverse but very large to download. |

Download videos into `data/videos/{real,fake}/`, then convert to face crops:

```bash
python scripts/extract_faces.py \
    --videos data/videos \
    --out    data/frames \
    --every-n 10 \
    --max-frames 32
```

Then split `data/frames/` into `train/`, `val/`, `test/` (any 80/10/10 split by source video — **never** split by frame, that leaks).

For a quick smoke test you can build a tiny dataset (say 50 real + 50 fake images per split) just to verify the pipeline runs end to end before downloading anything large.

---

## Training

```bash
python -m src.train \
    --train-dir data/train \
    --val-dir   data/val \
    --backbone  efficientnet_b4 \
    --img-size  380 \
    --epochs    15 \
    --batch-size 32 \
    --lr 3e-4
```

Outputs:

- `checkpoints/best.pt` — best model by validation AUC
- `checkpoints/tb/` — TensorBoard event files
  ```bash
  tensorboard --logdir checkpoints/tb
  ```

**Tips for the M2:**
- `efficientnet_b4` at `--img-size 380` and `--batch-size 32` fits comfortably in 16 GB of unified memory. If you're on a base 8 GB M2, drop to `efficientnet_b0` + `--img-size 224` + `--batch-size 16`.
- MPS is much faster than CPU but still ~3–5× slower than a single CUDA T4 — plan training time accordingly.

---

## Evaluation

```bash
python -m src.evaluate \
    --ckpt checkpoints/best.pt \
    --data-dir data/test
```

Prints accuracy / AUC / EER / precision / recall / F1, writes:

- `reports/metrics.json`
- `reports/roc.png`
- `reports/confusion_matrix.png`

---

## Inference

```bash
# image
python -m src.predict --ckpt checkpoints/best.pt --input path/to/photo.jpg

# video
python -m src.predict --ckpt checkpoints/best.pt --input path/to/clip.mp4 --every-n 10 --max-frames 32
```

Sample output:

```json
{
  "fake_probability": 0.873,
  "label": "FAKE",
  "num_frames_analyzed": 16,
  "per_frame_probs": [0.91, 0.84, 0.88, ...]
}
```

---

## Streamlit demo

```bash
streamlit run app/app.py
```

Opens a browser tab where you can drag and drop an image or video and see a verdict + raw probabilities. Useful for live demos in interviews.

---

## Results

Fill these in once you've trained on your dataset of choice. Example layout:

| Dataset | Backbone | Img size | Acc | AUC | EER |
|---|---|---|---|---|---|
| FaceForensics++ (c23) | EfficientNet-B4 | 380 | TBA | TBA | TBA |
| Celeb-DF v2 | EfficientNet-B4 | 380 | TBA | TBA | TBA |

Cross-dataset generalization (train on FF++, test on Celeb-DF) is the hardest setting and the one that's most worth highlighting if you go that direction.

---

## What this project demonstrates

For placement readers — this codebase touches:

- **Computer vision / deep learning:** transfer learning on a modern CNN backbone, binary classification, augmentation strategy tuned to the task.
- **PyTorch engineering:** clean module separation, `BCEWithLogitsLoss`, AdamW + cosine schedule, TensorBoard logging, checkpointing on best validation metric.
- **Real-world ML pipeline:** raw video → face detection → cropping → model input — not just `.fit()` on a CSV.
- **Apple Silicon awareness:** MPS / CUDA / CPU device picker, careful placement of MTCNN on CPU to avoid MPS fallbacks.
- **Productisation:** CLI training / eval / inference scripts, Streamlit demo, reproducible setup, license, sensible `.gitignore`.

---

## Roadmap

- Multi-frame temporal model (LSTM / Transformer over per-frame features) for video-level prediction.
- Test-time augmentation for more stable scores.
- ONNX / CoreML export for on-device M-series inference.
- Cross-dataset evaluation (FF++ → Celeb-DF, FF++ → DFDC) and reporting in the table above.
- Grad-CAM visualizations of which face regions trigger the "fake" prediction.

---

## License

MIT — see `LICENSE`.

---

## Acknowledgements

- [FaceForensics++](https://github.com/ondyari/FaceForensics)
- [Celeb-DF](https://github.com/yuezunli/celeb-deepfakeforensics)
- [facenet-pytorch](https://github.com/timesler/facenet-pytorch) for MTCNN
- [PyTorch](https://pytorch.org/) and [torchvision](https://pytorch.org/vision/) for the EfficientNet backbones
