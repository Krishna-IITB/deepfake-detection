"""Run a trained checkpoint on a labelled dataset and produce metrics.

Usage::

    python -m src.evaluate --ckpt checkpoints/best.pt --data-dir data/test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from .data import FaceFrameDataset, get_val_transforms
from .model import DeepfakeDetector
from .utils import get_device


@torch.no_grad()
def collect_predictions(model, loader, device):
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        x = x.to(device)
        y = y.to(device).unsqueeze(1)
        p = torch.sigmoid(model(x)).cpu().numpy().ravel()
        probs.extend(p.tolist())
        labels.extend(y.cpu().numpy().ravel().tolist())
    return np.array(probs), np.array(labels)


def equal_error_rate(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float, float]:
    """Compute EER + the operating threshold where FPR == FNR."""
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    fnr = 1 - tpr
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    eer = float((fpr[idx] + fnr[idx]) / 2)
    return eer, float(thresholds[idx])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--img-size", type=int, default=380)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--out-dir", default="reports")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device()
    print(f"Device: {device}")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    backbone = ckpt["args"].get("backbone", "efficientnet_b4")
    model = DeepfakeDetector(backbone=backbone, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])

    ds = FaceFrameDataset(args.data_dir, transform=get_val_transforms(args.img_size))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    probs, labels = collect_predictions(model, loader, device)
    preds = (probs >= 0.5).astype(int)

    acc = accuracy_score(labels, preds)
    auc = roc_auc_score(labels, probs)
    eer, eer_th = equal_error_rate(labels, probs)
    cm = confusion_matrix(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary")

    metrics = {
        "n_samples": int(len(labels)),
        "accuracy": float(acc),
        "auc": float(auc),
        "eer": eer,
        "eer_threshold": eer_th,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "confusion_matrix": cm.tolist(),
    }

    print("\n=== Test metrics ===")
    for k, v in metrics.items():
        print(f"{k:>20s}: {v}")

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "metrics.json").open("w") as f:
        json.dump(metrics, f, indent=2)

    fpr, tpr, _ = roc_curve(labels, probs)
    plt.figure(figsize=(5, 5))
    plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
    plt.plot([0, 1], [0, 1], "--", color="gray", linewidth=1)
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title("ROC curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out / "roc.png", dpi=150)
    plt.close()

    plt.figure(figsize=(4, 4))
    plt.imshow(cm, cmap="Blues")
    plt.xticks([0, 1], ["real", "fake"])
    plt.yticks([0, 1], ["real", "fake"])
    plt.xlabel("predicted")
    plt.ylabel("actual")
    plt.title("Confusion matrix")
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(out / "confusion_matrix.png", dpi=150)
    plt.close()

    print(f"\nSaved metrics + plots to {out}/")


if __name__ == "__main__":
    main()
