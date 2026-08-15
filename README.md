---
title: Bahraini Food Explorer
emoji: 🍽️
colorFrom: yellow
colorTo: green
sdk: gradio
sdk_version: "6.24.0"
app_file: gradio_app.py
pinned: false
---

# Bahraini Food Explorer

Detect and classify traditional Bahraini dishes in photos and live video with a
two-stage YOLO + Vision Transformer pipeline.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1UKzkRAD-gbk1ZHPQIh5-sA63TEj_qK3N?usp=sharing)
[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bahrain-food-classifier.streamlit.app/)

## Try It

| Link | Description |
|---|---|
| [Streamlit App](https://bahrain-food-classifier.streamlit.app/) | Upload, camera, and live-streaming demo |
| [Colab Notebook](https://colab.research.google.com/drive/1UKzkRAD-gbk1ZHPQIh5-sA63TEj_qK3N?usp=sharing) | Interactive training and detection notebook |
| Gradio (this repo) | Also deployed as a Hugging Face Space — run `gradio_app.py` |

## How It Works

1. **YOLO** (`models/best.pt`) locates food objects and draws boxes.
2. **ViT classifier** (`models/big_model.pt`, ViT-B/16 fine-tuned with a linear
   probe) identifies the dish inside each detection.
3. Two modes: YOLO-only detection, or two-stage detection + classification.

## Classes

balaleet · egg_tomato · fish · gaimat · halwa · karak · liver · ma3krona ·
nakhaj · samboosa · tikka

## Sample Photos

| | | | |
|---|---|---|---|
| balaleet<br><img src="data/balaleet/Balaleet_4.jpeg" width="150"> | egg_tomato<br><img src="data/eggs_tomato/IMG_5944.jpeg" width="150"> | fish<br><img src="data/fish/IMG_6011.jpeg" width="150"> | gaimat<br><img src="data/gaimat/images%20(15).jpg" width="150"> |
| halwa<br><img src="data/halwa/images%20(25).jpg" width="150"> | karak<br><img src="data/karak/Karak_14.jpeg" width="150"> | liver<br><img src="data/liver/hqdefault.jpg" width="150"> | ma3krona<br><img src="data/ma3krona/makarona5.jpg" width="150"> |
| nakhaj<br><img src="data/nakhaj/Nakhaj19.jpg" width="150"> | samboosa<br><img src="data/samboosa/images%20(3).jpg" width="150"> | tikka<br><img src="data/tikka/Tikka_30.jpeg" width="150"> | |

## Run Locally

```bash
pip install -r requirements.txt
```

```bash
# Streamlit app (camera, upload, live streaming)
streamlit run Bahfood_app.py

# Gradio app (Hugging Face Space)
python gradio_app.py
```

Place `models/best.pt` (YOLO) and optionally `models/big_model.pt` (ViT
classifier) in `models/`. The dataset lives in `data/<class>/`.

## Training

`train_big_model.py` fine-tunes the classifier: it loads a ViT-B/16 backbone,
replaces the head with an 11-class linear layer, trains a linear probe, then
optionally unfreezes the last blocks. See
[MODEL_REPORT.md](MODEL_REPORT.md) for the full model investigation.

## License

[MIT](LICENSE)
