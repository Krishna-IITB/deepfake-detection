"""Convert a corpus of real/fake videos into a folder of face crops.

Expected input layout::

    videos_root/
        real/  *.mp4
        fake/  *.mp4

Produces::

    out_root/
        real/  *.jpg
        fake/  *.jpg

Usage::

    python scripts/extract_faces.py --videos data/videos --out data/frames
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
from tqdm import tqdm

# Make ``src`` importable when running this script directly.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.face_extractor import FaceExtractor  # noqa: E402


VIDEO_GLOBS = ("*.mp4", "*.mov", "*.avi", "*.mkv", "*.webm")


def extract(videos_root: Path, out_root: Path, every_n: int, max_frames: int, image_size: int) -> None:
    extractor = FaceExtractor(image_size=image_size, device="cpu")
    for cls in ("real", "fake"):
        in_dir = videos_root / cls
        if not in_dir.exists():
            print(f"[skip] {in_dir} does not exist")
            continue
        out_dir = out_root / cls
        out_dir.mkdir(parents=True, exist_ok=True)

        videos = []
        for pattern in VIDEO_GLOBS:
            videos.extend(in_dir.glob(pattern))

        for v in tqdm(videos, desc=f"{cls:>4s}"):
            try:
                crops = extractor.extract_from_video(
                    str(v), every_n_frames=every_n, max_frames=max_frames
                )
            except Exception as e:  # noqa: BLE001
                print(f"  ! skipped {v.name}: {e}")
                continue

            for i, face in enumerate(crops):
                bgr = cv2.cvtColor(face, cv2.COLOR_RGB2BGR)
                cv2.imwrite(
                    str(out_dir / f"{v.stem}_{i:03d}.jpg"),
                    bgr,
                    [cv2.IMWRITE_JPEG_QUALITY, 95],
                )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--videos", required=True, help="root with real/ and fake/ video subdirs")
    p.add_argument("--out", required=True, help="output root for face crops")
    p.add_argument("--every-n", type=int, default=10, help="sample every N frames per video")
    p.add_argument("--max-frames", type=int, default=32, help="max crops per video")
    p.add_argument("--image-size", type=int, default=300, help="output crop size (px)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    extract(
        videos_root=Path(args.videos),
        out_root=Path(args.out),
        every_n=args.every_n,
        max_frames=args.max_frames,
        image_size=args.image_size,
    )
