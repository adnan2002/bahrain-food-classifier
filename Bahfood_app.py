import streamlit as st
import datetime
import pandas as pd
import random
import json
from pathlib import Path
import numpy as np
from PIL import Image
import os
from ultralytics import YOLO

st.set_page_config(
    page_title="Bahraini Food Explorer",
    page_icon="🍽️",
    layout="wide"
)

# Title
st.title("🍽️ Bahraini Food Explorer")
st.write( "Take a photo or upload an image to detect "
         "Bahraini food."
)

st.divider()

CLASSES = [
    "balaleet", "egg_tomato", "fish", "gaimat", "halwa",
    "karak", "liver", "ma3krona", "nakhaj", "samboosa", "tikka"
]

MODEL_PATH = Path("models/best.pt")


@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


@st.cache_resource
def get_model_info(model):
    if hasattr(model, "names") and model.names:
        names = [model.names[i] for i in sorted(model.names)]
        return names
    return CLASSES


# -----------------------------------------------------
# SELECT IMAGE SOURCE
# -----------------------------------------------------

st.subheader("Choose Image Source")

image_source = st.radio( "Select an option:", [ "📷 Camera",  "📁 Upload Image"],  horizontal=True )

# =====================================================
# CAMERA
# =====================================================

if image_source == "📷 Camera":
    st.subheader("📷 Take a Photo")

    camera_image = st.camera_input( "Take a picture of the food" )
    
    if camera_image is not None:
        st.success("Image captured successfully!")
        
        col1, col2 = st.columns(2)

            # -------------------------------------------------
            # IMAGE
            # -------------------------------------------------
        with col1:
            st.subheader("📷 Captured Image")
            st.image(camera_image, use_container_width=True)

            # -------------------------------------------------
            # DETECTION RESULT
            # -------------------------------------------------

        with col2:
            st.subheader("🎯 Detection Result")
            
            if st.button( "🔍 Detect Food", use_container_width=True):
                model = load_model()
                model_names = get_model_info(model)

                img = Image.open(camera_image).convert("RGB")
                results = model(img)

                boxes = results[0].boxes
                if len(boxes) == 0:
                    st.warning("No food detected in the image.")
                    st.write("Food Class: —")
                    st.write("Confidence: —")
                    st.write("Objects Detected: 0")
                else:
                    labels = [model_names[int(cls)] for cls in boxes.cls]
                    confs = [float(conf) for conf in boxes.conf]

                    best_idx = int(np.argmax(confs))

                    st.image(results[0].plot(), use_container_width=True,
                             caption="Detection Result")

                    st.write(f"Food Class: **{labels[best_idx]}**")
                    st.write(f"Confidence: **{confs[best_idx]:.2%}**")
                    st.write(f"Objects Detected: **{len(boxes)}**")

                    st.dataframe(
                        pd.DataFrame({
                            "Class": labels,
                            "Confidence": [f"{c:.2%}" for c in confs],
                        })
                    )

    # =====================================================
    # UPLOAD IMAGE
    # =====================================================

else:
    st.subheader("📁 Upload an Image")
    uploaded_image = st.file_uploader( "Choose a food image", type=["jpg", "jpeg", "png" ])
    
    if uploaded_image is not None:
        st.success("Image uploaded successfully!")
        
        col1, col2 = st.columns(2)
# -------------------------------------------------
# IMAGE
# -------------------------------------------------
        
        with col1:
            st.subheader("📷 Uploaded Image")
            st.image(uploaded_image, use_container_width=True )
            
# -------------------------------------------------
# DETECTION
# -------------------------------------------------
        
        with col2:
            st.subheader("🎯 Detection Result")
            
            if st.button("🔍 Detect Food", use_container_width=True):
                model = load_model()
                model_names = get_model_info(model)

                img = Image.open(uploaded_image).convert("RGB")
                results = model(img)

                boxes = results[0].boxes
                if len(boxes) == 0:
                    st.warning("No food detected in the image.")
                    st.write("Food Class: —")
                    st.write("Confidence: —")
                    st.write("Objects Detected: 0")
                else:
                    labels = [model_names[int(cls)] for cls in boxes.cls]
                    confs = [float(conf) for conf in boxes.conf]

                    best_idx = int(np.argmax(confs))

                    st.image(results[0].plot(), use_container_width=True,
                             caption="Detection Result")

                    st.write(f"Food Class: **{labels[best_idx]}**")
                    st.write(f"Confidence: **{confs[best_idx]:.2%}**")
                    st.write(f"Objects Detected: **{len(boxes)}**")

                    st.dataframe(
                        pd.DataFrame({
                            "Class": labels,
                            "Confidence": [f"{c:.2%}" for c in confs],
                        })
                    )
