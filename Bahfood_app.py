import streamlit as st
import datetime
import pandas as pd
import random
import json
import time
from pathlib import Path
import numpy as np
import cv2
import av
import torch
import timm
from torchvision import transforms
from PIL import Image
import os
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase

st.set_page_config(
    page_title="Bahraini Food Explorer",
    page_icon="🍽️",
    layout="wide"
)

# Title
st.title("🍽️ Bahraini Food Explorer")
st.write( "Take a photo, upload an image, or stream video to detect "
         "Bahraini food."
)

st.divider()

CLASSES = [
    "balaleet", "egg_tomato", "fish", "gaimat", "halwa",
    "karak", "liver", "ma3krona", "nakhaj", "samboosa", "tikka"
]

DET_MODEL_PATH = Path("models/best.pt")
CLF_MODEL_PATH = Path("models/big_model.pt")
CONF_THRESHOLD = 0.25

RTC_CONFIGURATION = {
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
}

# ------------------------------------------------------------------
# Models (module-level globals so webrtc processor threads can use them)
# ------------------------------------------------------------------

DET_MODEL = None
CLF_MODEL = None
CLF_TRANSFORM = None
STREAM_STATE = {"mode": "yolo", "conf": CONF_THRESHOLD}


@st.cache_resource
def load_detector():
    return YOLO(str(DET_MODEL_PATH))


@st.cache_resource
def load_classifier():
    if not CLF_MODEL_PATH.exists():
        return None
    ckpt = torch.load(str(CLF_MODEL_PATH), map_location="cpu", weights_only=False)
    clf = timm.create_model("vit_base_patch16_224", pretrained=False,
                            num_classes=len(CLASSES))
    clf.load_state_dict(ckpt["state_dict"])
    clf.eval()
    return clf


@st.cache_resource
def get_transform():
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


@st.cache_resource
def get_model_info(_model):
    if hasattr(_model, "names") and _model.names:
        names = [_model.names[i] for i in sorted(_model.names)]
        return names
    return CLASSES


def run_yolo(img_bgr, conf=CONF_THRESHOLD, imgsz=640):
    r = DET_MODEL.predict(img_bgr, imgsz=imgsz, conf=conf, verbose=False)[0]
    labels = [DET_MODEL.names[int(c)] for c in r.boxes.cls]
    confs = [float(c) for c in r.boxes.conf]
    return r.plot(), labels, confs


def classify_crop(bgr):
    x = CLF_TRANSFORM(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).unsqueeze(0)
    with torch.no_grad():
        return CLASSES[CLF_MODEL(x).argmax(1).item()]


def run_two_stage(img_bgr, conf=CONF_THRESHOLD, imgsz=640):
    boxes = DET_MODEL.predict(img_bgr, imgsz=imgsz, conf=conf, verbose=False)[0].boxes
    out = img_bgr.copy()
    labels, confs = [], []
    for b in boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        name = classify_crop(img_bgr[max(0, y1):y2, max(0, x1):x2])
        labels.append(name)
        confs.append(float(b.conf.item()))
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"{name} {b.conf.item():.2f}", (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return out, labels, confs


def process_frame(img_bgr, mode, conf=CONF_THRESHOLD, imgsz=640):
    if mode == "two_stage" and CLF_MODEL is not None:
        return run_two_stage(img_bgr, conf, imgsz)
    return run_yolo(img_bgr, conf, imgsz)


def show_results(annotated_bgr, labels, confs, col):
    if len(labels) == 0:
        col.warning("No food detected in the image.")
        col.write("Food Class: —")
        col.write("Confidence: —")
        col.write("Objects Detected: 0")
    else:
        best_idx = int(np.argmax(confs))
        col.image(annotated_bgr[..., ::-1], use_container_width=True,
                  caption="Detection Result")
        col.write(f"Food Class: **{labels[best_idx]}**")
        col.write(f"Confidence: **{confs[best_idx]:.2%}**")
        col.write(f"Objects Detected: **{len(labels)}**")
        col.dataframe(pd.DataFrame({
            "Class": labels,
            "Confidence": [f"{c:.2%}" for c in confs],
        }))


class FoodVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self._last_process = 0.0
        self._last_output = None

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")

        now = time.monotonic()
        interval = STREAM_STATE.get("interval", 0.25)
        if now - self._last_process < interval and self._last_output is not None:
            return self._last_output

        self._last_process = now
        mode = STREAM_STATE["mode"]
        conf = STREAM_STATE["conf"]
        imgsz = STREAM_STATE.get("imgsz", 480)
        if mode == "two_stage" and CLF_MODEL is None:
            mode = "yolo"

        out, _, _ = process_frame(img, mode, conf, imgsz)
        self._last_output = av.VideoFrame.from_ndarray(out, format="bgr24")
        return self._last_output


# -----------------------------------------------------
# DETECTION MODE
# -----------------------------------------------------

st.subheader("Choose Detection Mode")

mode = st.radio(
    "Select a detection pipeline:",
    ["YOLO Detection (best.pt)",
     "Two-Stage Detection + Classification (big_model.pt)"],
    horizontal=True,
)

two_stage = mode.startswith("Two-Stage")

DET_MODEL = load_detector()
CLF_MODEL = load_classifier()
CLF_TRANSFORM = get_transform()

if two_stage and CLF_MODEL is None:
    st.warning("⚠️ models/big_model.pt is not available (it is not in the "
               "repository). Falling back to YOLO-only detection.")
    two_stage = False

STREAM_STATE["mode"] = "two_stage" if two_stage else "yolo"
STREAM_STATE["conf"] = CONF_THRESHOLD

st.divider()

# -----------------------------------------------------
# SELECT IMAGE SOURCE
# -----------------------------------------------------

st.subheader("Choose Image Source")

image_source = st.radio(
    "Select an option:",
    ["📷 Camera", "📁 Upload Image", "🎥 Live Streaming"],
    horizontal=True,
)

# =====================================================
# CAMERA
# =====================================================

if image_source == "📷 Camera":
    st.subheader("📷 Take a Photo")

    camera_image = st.camera_input("Take a picture of the food")

    if camera_image is not None:
        st.success("Image captured successfully!")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📷 Captured Image")
            st.image(camera_image, use_container_width=True)

        with col2:
            st.subheader("🎯 Detection Result")

            if st.button("🔍 Detect Food", use_container_width=True):
                img_bgr = cv2.cvtColor(np.array(Image.open(camera_image).convert("RGB")),
                                       cv2.COLOR_RGB2BGR)
                annotated, labels, confs = process_frame(img_bgr, STREAM_STATE["mode"])
                show_results(annotated, labels, confs, col2)

# =====================================================
# UPLOAD IMAGE
# =====================================================

elif image_source == "📁 Upload Image":
    st.subheader("📁 Upload an Image")
    uploaded_image = st.file_uploader(
        "Choose a food image", type=["jpg", "jpeg", "png"])

    if uploaded_image is not None:
        st.success("Image uploaded successfully!")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📷 Uploaded Image")
            st.image(uploaded_image, use_container_width=True)

        with col2:
            st.subheader("🎯 Detection Result")

            if st.button("🔍 Detect Food", use_container_width=True):
                img_bgr = cv2.cvtColor(np.array(Image.open(uploaded_image).convert("RGB")),
                                       cv2.COLOR_RGB2BGR)
                annotated, labels, confs = process_frame(img_bgr, STREAM_STATE["mode"])
                show_results(annotated, labels, confs, col2)

# =====================================================
# LIVE STREAMING
# =====================================================

else:
    st.subheader("🎥 Live Streaming")

    if two_stage and CLF_MODEL is None:
        st.info("Two-Stage mode is not available on this deployment, "
                "so streaming uses YOLO detection only.")

    st.write("Click **Start** below and allow camera access. "
             "The processed video will stream back in real time.")

    process_hz = st.slider("Processing rate (inference per second)",
                           min_value=1, max_value=10, value=4)
    STREAM_STATE["interval"] = 1.0 / process_hz
    STREAM_STATE["imgsz"] = 480

    webrtc_streamer(
        key="food-stream",
        video_processor_factory=FoodVideoProcessor,
        rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    st.caption(f"Detection mode: "
               f"{'Two-Stage Detection + Classification' if two_stage else 'YOLO Detection'}"
               f" · confidence ≥ {CONF_THRESHOLD:.0%}")
