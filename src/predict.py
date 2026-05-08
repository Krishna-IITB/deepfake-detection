"""End-to-end inference: image / video in -> verdict + score out.

Usage::

    python -m src.predict --ckpt checkpoints/best.pt --input path/to/clip.mp4
    python -m src.predict --ckpt checkpoints/best.pt --input path/to/photo.jpg
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from PIL import Image

from .data import get_val_transforms
from .face_extractor import FaceExtractor
from .model import DeepfakeDetector
from .utils import get_device


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def load_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    backbone = ckpt["args"].get("backbone", "efficientnet_b4")
    img_size = ckpt["args"].get("img_size", 380)
    model = DeepfakeDetector(backbone=backbone, pretrained=False).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, img_size


@torch.no_grad()
def predict_image(image_path: str, ckpt_path: str) -> Dict:
    device = get_device()
    model, img_size = load_model(ckpt_path, device)
    extractor = FaceExtractor(image_size=img_size, device="cpu")
    transform = get_val_transforms(img_size)

    img = np.array(Image.open(image_path).convert("RGB"))
    faces = extractor.detect(img)
    if not faces:
        return {"error": "no face detected"}

    per_face = []
    for face in faces:
        t = transform(image=face)["image"].unsqueeze(0).to(device)
        per_face.append(float(torch.sigmoid(model(t)).item()))

    avg = float(np.mean(per_face))
    return {
        "fake_probability": avg,
        "label": "FAKE" if avg >= 0.5 else "REAL",
        "num_faces": len(faces),
        "per_face_probs": per_face,
    }


@torch.no_grad()
def predict_video(
    video_path: str,
    ckpt_path: str,
    every_n_frames: int = 10,
    max_frames: int = 32,
) -> Dict:
    device = get_device()
    model, img_size = load_model(ckpt_path, device)
    extractor = FaceExtractor(image_size=img_size, device="cpu")
    transform = get_val_transforms(img_size)

    crops = extractor.extract_from_video(
        video_path, every_n_frames=every_n_frames, max_frames=max_frames
    )
    if not crops:
        return {"error": "no faces detected in video"}

    per_frame = []
    for face in crops:
        t = transform(image=face)["image"].unsqueeze(0).to(device)
        per_frame.append(float(torch.sigmoid(model(t)).item()))

    avg = float(np.mean(per_frame))
    return {
        "fake_probability": avg,
        "label": "FAKE" if avg >= 0.5 else "REAL",
        "num_frames_analyzed": len(per_frame),
        "per_frame_probs": per_frame,
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--input", required=True, help="path to image or video")
    p.add_argument("--every-n", type=int, default=10)
    p.add_argument("--max-frames", type=int, default=32)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    ext = Path(args.input).suffix.lower()
    if ext in VIDEO_EXTS:
        result = predict_video(args.input, args.ckpt, args.every_n, args.max_frames)
    else:
        result = predict_image(args.input, args.ckpt)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
