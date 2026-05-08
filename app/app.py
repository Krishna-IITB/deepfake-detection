"""Streamlit demo for the deepfake detector.

Run from the repo root::

    streamlit run app/app.py

Drag-and-drop an image or video, optionally tweak the sampling settings in
the sidebar, and the app will display a verdict + per-face / per-frame
probabilities.
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import streamlit as st
from PIL import Image

# Make ``src`` importable when Streamlit launches the script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.predict import predict_image, predict_video  # noqa: E402


VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


st.set_page_config(page_title="Deepfake Detector", page_icon="🕵️", layout="centered")

st.title("🕵️ Deepfake Detector")
st.caption(
    "EfficientNet-B4 trained on face crops. Drop in an image or video and "
    "the model will flag whether the face looks real or AI-manipulated."
)

with st.sidebar:
    st.header("Settings")
    ckpt_path = st.text_input("Checkpoint path", "checkpoints/best.pt")
    every_n = st.slider("Sample every N frames (video)", 1, 30, 10)
    max_frames = st.slider("Max frames to analyze (video)", 4, 64, 32)
    st.markdown("---")
    st.markdown(
        "**Tip:** for short clips, lower `every N` to get more samples. "
        "For long clips, raise `Max frames` carefully — inference time scales linearly."
    )

uploaded = st.file_uploader(
    "Upload an image or video",
    type=["jpg", "jpeg", "png", "mp4", "mov", "avi", "mkv", "webm"],
)

if uploaded is None:
    st.info("⬆ Upload a file to get started.")
    st.stop()

if not Path(ckpt_path).exists():
    st.error(
        f"Checkpoint not found at `{ckpt_path}`.\n\n"
        "Train one first with `python -m src.train ...` or update the path in the sidebar."
    )
    st.stop()

suffix = Path(uploaded.name).suffix.lower()
with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
    tmp.write(uploaded.read())
    tmp_path = tmp.name

is_video = suffix in VIDEO_EXTS

if is_video:
    st.video(tmp_path)
    with st.spinner("Detecting faces and scoring frames..."):
        result = predict_video(tmp_path, ckpt_path, every_n, max_frames)
else:
    st.image(Image.open(tmp_path), use_column_width=True)
    with st.spinner("Detecting faces and scoring..."):
        result = predict_image(tmp_path, ckpt_path)

if "error" in result:
    st.error(result["error"])
else:
    label = result["label"]
    fake_p = result["fake_probability"]

    col1, col2 = st.columns(2)
    col1.metric("Verdict", label)
    col2.metric("Fake probability", f"{fake_p * 100:.1f}%")

    bar_color = "🔴" if label == "FAKE" else "🟢"
    st.write(f"{bar_color} Confidence")
    st.progress(fake_p if label == "FAKE" else 1 - fake_p)

    with st.expander("Raw output"):
        st.json(result)
