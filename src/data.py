"""Datasets and dataloaders for cropped-face deepfake images.

The detector consumes pre-extracted face crops, not raw video frames.
Run ``scripts/extract_faces.py`` first to convert a corpus of real/fake
videos into face crops organised as::

    root/
      real/
        clip001_000.jpg
        clip001_001.jpg
        ...
      fake/
        clip004_000.jpg
        ...

Label convention: ``0 = real``, ``1 = fake``.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, List, Optional, Tuple

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2
from PIL import Image
from torch.utils.data import DataLoader, Dataset


# ImageNet stats — what the EfficientNet backbone was pretrained with.
IMG_MEAN = (0.485, 0.456, 0.406)
IMG_STD = (0.229, 0.224, 0.225)


def get_train_transforms(size: int = 380) -> A.Compose:
    """Augmentations for training. JPEG compression + brightness jitter
    are particularly important for deepfake generalisation — many forgeries
    leave compression artefacts the model would otherwise overfit on."""
    return A.Compose(
        [
            A.Resize(size, size),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=15, val_shift_limit=10, p=0.3),
            A.GaussNoise(p=0.2),
            A.ImageCompression(quality_lower=60, quality_upper=100, p=0.3),
            A.Normalize(mean=IMG_MEAN, std=IMG_STD),
            ToTensorV2(),
        ]
    )


def get_val_transforms(size: int = 380) -> A.Compose:
    """Deterministic transforms for validation / inference."""
    return A.Compose(
        [
            A.Resize(size, size),
            A.Normalize(mean=IMG_MEAN, std=IMG_STD),
            ToTensorV2(),
        ]
    )


class FaceFrameDataset(Dataset):
    """Image-folder dataset where ``real/`` and ``fake/`` subdirs hold crops."""

    EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root: str, transform: Optional[Callable] = None) -> None:
        self.root = Path(root)
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []
        for label, subdir in [(0, "real"), (1, "fake")]:
            d = self.root / subdir
            if not d.exists():
                continue
            for p in sorted(d.iterdir()):
                if p.suffix.lower() in self.EXTS:
                    self.samples.append((str(p), label))
        if not self.samples:
            raise FileNotFoundError(
                f"No images found under {self.root}/real or {self.root}/fake. "
                f"Did you run scripts/extract_faces.py first?"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path, label = self.samples[idx]
        img = np.array(Image.open(path).convert("RGB"))
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        return img, torch.tensor(label, dtype=torch.float32)

    def class_counts(self) -> dict:
        counts = {"real": 0, "fake": 0}
        for _, lbl in self.samples:
            counts["fake" if lbl else "real"] += 1
        return counts


def make_dataloaders(
    train_dir: str,
    val_dir: str,
    batch_size: int = 32,
    num_workers: int = 4,
    img_size: int = 380,
) -> Tuple[DataLoader, DataLoader]:
    train_ds = FaceFrameDataset(train_dir, transform=get_train_transforms(img_size))
    val_ds = FaceFrameDataset(val_dir, transform=get_val_transforms(img_size))
    print(f"train: {len(train_ds)} samples  {train_ds.class_counts()}")
    print(f"val:   {len(val_ds)} samples  {val_ds.class_counts()}")
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader
