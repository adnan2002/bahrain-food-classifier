import streamlit as st
import datetime
import pandas as pd
import random
import json
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image
import os

st.set_page_config(
    page_title="Bahraini Food Explorer",
    page_icon="🍽️",
    layout="wide"
)

# Title
st.title("🍽️ Bahraini Food Explorer")

tab1, tab2 = st.tabs([
    "📷 Food Detection",
    "📊 Performance",
])

# =========================================================
# FOOD DETECTION PAGE
# =========================================================


with tab1:
    #st.header("Welcome to Bahraini Food Explorer")
    
    st.header("📷 Bahraini Food Detection")

    st.write(
        "Take a photo or upload an image to detect "
        "Bahraini food."
    )

    st.divider()

    # -----------------------------------------------------
    # SELECT IMAGE SOURCE
    # -----------------------------------------------------

    st.subheader("Choose Image Source")

    image_source = st.radio(
        "Select an option:",
        [
            "📷 Camera",
            "📁 Upload Image"
        ],
        horizontal=True
    )

    # =====================================================
    # CAMERA
    # =====================================================

    if image_source == "📷 Camera":

        st.subheader("📷 Take a Photo")

        camera_image = st.camera_input(
            "Take a picture of the food"
        )

        if camera_image is not None:

            st.success(
                "Image captured successfully!"
            )

            col1, col2 = st.columns(2)

            # -------------------------------------------------
            # IMAGE
            # -------------------------------------------------

            with col1:

                st.subheader("📷 Captured Image")

                st.image(
                    camera_image,
                    use_container_width=True
                )

            # -------------------------------------------------
            # DETECTION RESULT
            # -------------------------------------------------

            with col2:

                st.subheader("🎯 Detection Result")

                if st.button(
                    "🔍 Detect Food",
                    use_container_width=True
                ):

                    st.info(
                        "YOLO model is not connected yet."
                    )

                st.write("Food Class: —")

                st.write("Confidence: —")

                st.write("Objects Detected: —")


    # =====================================================
    # UPLOAD IMAGE
    # =====================================================

    else:

        st.subheader("📁 Upload an Image")

        uploaded_image = st.file_uploader(
            "Choose a food image",
            type=[
                "jpg",
                "jpeg",
                "png"
            ]
        )

        if uploaded_image is not None:

            st.success(
                "Image uploaded successfully!"
            )

            col1, col2 = st.columns(2)

            # -------------------------------------------------
            # IMAGE
            # -------------------------------------------------

            with col1:

                st.subheader("📷 Uploaded Image")

                st.image(
                    uploaded_image,
                    use_container_width=True
                )

            # -------------------------------------------------
            # DETECTION
            # -------------------------------------------------

            with col2:

                st.subheader("🎯 Detection Result")

                if st.button(
                    "🔍 Detect Food",
                    use_container_width=True
                ):

                    st.info(
                        "YOLO model is not connected yet."
                    )

                st.write("Food Class: —")

                st.write("Confidence: —")

                st.write("Objects Detected: —")





with tab2:
    st.header("📊 Performance")
