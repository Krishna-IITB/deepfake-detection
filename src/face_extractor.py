"""Face detection + cropping wrapper.

Uses ``facenet-pytorch``'s MTCNN to find faces in still images and video frames.
On Apple Silicon we run MTCNN on CPU because some of its ops fall back from
MPS anyway — the speed loss is negligible and avoids occasional MPS quirks.
"""
from __future__ import annotations

from typing import List

import cv2
import numpy as np
from PIL import Image


class FaceExtractor:
    """Wraps MTCNN for face detection + tight cropping with margin."""

    def __init__(
        self,
        image_size: int = 300,
        margin: float = 0.3,
        device: str = "cpu",
    ) -> None:
        from facenet_pytorch import MTCNN  # local import — heavy
        self.image_size = image_size
        self.margin = margin
        self.mtcnn = MTCNN(
            image_size=image_size,
            margin=int(image_size * margin),
            keep_all=True,
            post_process=False,
            device=device,
        )

    def detect(self, image: np.ndarray) -> List[np.ndarray]:
        """Detect faces in an RGB image; return list of square uint8 crops."""
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected RGB image of shape (H, W, 3); got {image.shape}")

        pil = Image.fromarray(image)
        boxes, _ = self.mtcnn.detect(pil)
        crops: List[np.ndarray] = []
        if boxes is None:
            return crops

        h, w = image.shape[:2]
        for box in boxes:
            x1, y1, x2, y2 = box.astype(int)
            mx = int((x2 - x1) * self.margin)
            my = int((y2 - y1) * self.margin)
            x1 = max(0, x1 - mx)
            y1 = max(0, y1 - my)
            x2 = min(w, x2 + mx)
            y2 = min(h, y2 + my)
            crop = image[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (self.image_size, self.image_size))
            crops.append(crop)
        return crops

    def extract_from_video(
        self,
        video_path: str,
        every_n_frames: int = 10,
        max_frames: int = 32,
    ) -> List[np.ndarray]:
        """Sample frames from a video and return one face crop per sampled frame."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")

        crops: List[np.ndarray] = []
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok or len(crops) >= max_frames:
                break
            if idx % every_n_frames == 0:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = self.detect(rgb)
                if faces:
                    crops.append(faces[0])  # take the largest/first face per frame
            idx += 1
        cap.release()
        return crops
