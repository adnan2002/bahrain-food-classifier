import gradio as gr
import pandas as pd
from pathlib import Path
import numpy as np
import cv2
import torch
import timm
from torchvision import transforms
from ultralytics import YOLO

CLASSES = [
    "balaleet", "egg_tomato", "fish", "gaimat", "halwa",
    "karak", "liver", "ma3krona", "nakhaj", "samboosa", "tikka"
]

DET_MODEL_PATH = Path("models/best.pt")
CLF_MODEL_PATH = Path("models/big_model.pt")
CONF_THRESHOLD = 0.25

_models = {"det": None, "clf": None, "tf": None}


def load_models():
    if _models["det"] is None:
        _models["det"] = YOLO(str(DET_MODEL_PATH))
        if CLF_MODEL_PATH.exists():
            ckpt = torch.load(str(CLF_MODEL_PATH), map_location="cpu",
                              weights_only=False)
            clf = timm.create_model("vit_base_patch16_224", pretrained=False,
                                    num_classes=len(CLASSES))
            clf.load_state_dict(ckpt["state_dict"])
            clf.eval()
            _models["clf"] = clf
            _models["tf"] = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406],
                                     [0.229, 0.224, 0.225]),
            ])
    return _models


def classify_crop(bgr):
    m = load_models()
    x = m["tf"](cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)).unsqueeze(0)
    with torch.no_grad():
        return CLASSES[m["clf"](x).argmax(1).item()]


def run_yolo(rgb, conf=CONF_THRESHOLD, imgsz=640):
    det = load_models()["det"]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    r = det.predict(bgr, imgsz=imgsz, conf=conf, verbose=False)[0]
    labels = [det.names[int(c)] for c in r.boxes.cls]
    confs = [float(c) for c in r.boxes.conf]
    return r.plot()[..., ::-1], labels, confs


def run_two_stage(rgb, conf=CONF_THRESHOLD, imgsz=640):
    det = load_models()["det"]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    boxes = det.predict(bgr, imgsz=imgsz, conf=conf, verbose=False)[0].boxes
    out = bgr.copy()
    labels, confs = [], []
    for b in boxes:
        x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
        name = classify_crop(bgr[max(0, y1):y2, max(0, x1):x2])
        labels.append(name)
        confs.append(float(b.conf.item()))
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"{name} {b.conf.item():.2f}", (x1, max(0, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    return out[..., ::-1], labels, confs


def process_frame(rgb, two_stage, imgsz=640):
    if two_stage and load_models()["clf"] is not None:
        return run_two_stage(rgb, imgsz=imgsz)
    return run_yolo(rgb, imgsz=imgsz)


def detect(rgb, mode):
    if rgb is None:
        return None, {}, pd.DataFrame()
    annotated, labels, confs = process_frame(rgb, mode == "two_stage")
    if not labels:
        return annotated, {"None": 0.0}, pd.DataFrame()
    best_idx = int(np.argmax(confs))
    df = pd.DataFrame({
        "Class": labels,
        "Confidence": [f"{c:.2%}" for c in confs],
    })
    return annotated, {labels[best_idx]: confs[best_idx]}, df


def stream_frame(rgb, mode):
    if rgb is None:
        return None
    annotated, _, _ = process_frame(rgb, mode == "two_stage", imgsz=480)
    return annotated


TWO_STAGE_LABEL = "Two-Stage Detection + Classification (big_model.pt)"
YOLO_LABEL = "YOLO Detection (best.pt)"

with gr.Blocks(title="🍽️ Bahraini Food Explorer") as demo:
    gr.Markdown("# 🍽️ Bahraini Food Explorer\n"
                "Take a photo, upload an image, or stream your webcam to "
                "detect Bahraini food.")

    mode = gr.Radio(
        [YOLO_LABEL, TWO_STAGE_LABEL],
        label="Detection Mode",
        value=YOLO_LABEL,
    )

    if not CLF_MODEL_PATH.exists():
        gr.Warning("⚠️ models/big_model.pt is not available. Two-Stage mode "
                   "will fall back to YOLO detection.")

    with gr.Tab("📁 Photo / Upload"):
        with gr.Row():
            with gr.Column():
                img_in = gr.Image(
                    sources=["upload", "webcam"],
                    type="numpy",
                    label="Input Image",
                )
                detect_btn = gr.Button("🔍 Detect Food", variant="primary")
            with gr.Column():
                img_out = gr.Image(type="numpy", label="Detection Result")
                label_out = gr.Label(label="Best Detection")
                df_out = gr.Dataframe(
                    headers=["Class", "Confidence"],
                    label="Detected Objects",
                )
        detect_btn.click(
            fn=detect,
            inputs=[img_in, mode],
            outputs=[img_out, label_out, df_out],
        )

    with gr.Tab("🎥 Live Streaming"):
        gr.Markdown("Start the webcam below — the annotated video streams "
                    "back in real time.")
        with gr.Row():
            stream_in = gr.Image(
                sources=["webcam"],
                streaming=True,
                interactive=True,
                type="numpy",
                label="Webcam Input",
            )
            stream_out = gr.Image(
                type="numpy",
                label="Annotated Stream",
            )
        stream_in.stream(
            fn=stream_frame,
            inputs=[stream_in, mode],
            outputs=stream_out,
            stream_every=0.25,
            show_progress="hidden",
        )

demo.queue()

if __name__ == "__main__":
    demo.launch()
