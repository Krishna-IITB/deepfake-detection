"""Training entry point.

Usage::

    python -m src.train \
        --train-dir data/train \
        --val-dir   data/val   \
        --backbone  efficientnet_b4 \
        --epochs    15

Training writes:
* ``checkpoints/best.pt``   — best model by validation AUC
* ``checkpoints/tb/...``    — TensorBoard event files
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, roc_auc_score
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .data import make_dataloaders
from .model import DeepfakeDetector
from .utils import count_parameters, get_device, set_seed


def _binary_metrics(probs: list, labels: list) -> tuple[float, float]:
    preds = [1 if p >= 0.5 else 0 for p in probs]
    acc = accuracy_score(labels, preds)
    try:
        auc = roc_auc_score(labels, probs)
    except ValueError:
        # Single-class batch / val set — AUC is undefined.
        auc = float("nan")
    return acc, auc


def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    total_loss = 0.0
    probs, labels = [], []
    pbar = tqdm(loader, desc=f"epoch {epoch:02d} [train]", leave=False)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).unsqueeze(1)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * x.size(0)
        probs.extend(torch.sigmoid(logits).detach().cpu().numpy().ravel().tolist())
        labels.extend(y.detach().cpu().numpy().ravel().tolist())
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    avg_loss = total_loss / len(loader.dataset)
    acc, auc = _binary_metrics(probs, labels)
    return avg_loss, acc, auc


@torch.no_grad()
def validate(model, loader, criterion, device, epoch):
    model.eval()
    total_loss = 0.0
    probs, labels = [], []
    pbar = tqdm(loader, desc=f"epoch {epoch:02d} [val]  ", leave=False)
    for x, y in pbar:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).unsqueeze(1)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * x.size(0)
        probs.extend(torch.sigmoid(logits).cpu().numpy().ravel().tolist())
        labels.extend(y.cpu().numpy().ravel().tolist())

    avg_loss = total_loss / len(loader.dataset)
    acc, auc = _binary_metrics(probs, labels)
    return avg_loss, acc, auc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train deepfake detector.")
    p.add_argument("--train-dir", required=True, help="dir with real/ and fake/ subdirs")
    p.add_argument("--val-dir", required=True, help="dir with real/ and fake/ subdirs")
    p.add_argument("--out-dir", default="checkpoints")
    p.add_argument("--backbone", default="efficientnet_b4",
                   choices=["efficientnet_b0", "efficientnet_b3", "efficientnet_b4"])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--img-size", type=int, default=380)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out_dir / "tb"))

    train_loader, val_loader = make_dataloaders(
        args.train_dir,
        args.val_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size,
    )

    model = DeepfakeDetector(backbone=args.backbone, pretrained=True).to(device)
    print(f"Backbone: {args.backbone}  trainable params: {count_parameters(model):,}")

    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_auc = -np.inf
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc, tr_auc = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_loss, val_acc, val_auc = validate(model, val_loader, criterion, device, epoch)
        scheduler.step()

        print(
            f"E{epoch:02d}  "
            f"train  loss={tr_loss:.4f}  acc={tr_acc:.4f}  auc={tr_auc:.4f}  |  "
            f"val  loss={val_loss:.4f}  acc={val_acc:.4f}  auc={val_auc:.4f}"
        )
        writer.add_scalars("loss", {"train": tr_loss, "val": val_loss}, epoch)
        writer.add_scalars("acc",  {"train": tr_acc,  "val": val_acc},  epoch)
        writer.add_scalars("auc",  {"train": tr_auc,  "val": val_auc},  epoch)
        writer.add_scalar("lr", optimizer.param_groups[0]["lr"], epoch)

        if val_auc > best_auc:
            best_auc = val_auc
            ckpt = {
                "model_state": model.state_dict(),
                "args": vars(args),
                "epoch": epoch,
                "val_auc": val_auc,
                "val_acc": val_acc,
            }
            torch.save(ckpt, out_dir / "best.pt")
            print(f"  ✓ saved {out_dir/'best.pt'}  (val AUC {val_auc:.4f})")

    writer.close()
    print(f"Done. Best val AUC: {best_auc:.4f}")


if __name__ == "__main__":
    main()
