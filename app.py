import streamlit as st
from streamlit_drawable_canvas import st_canvas
import torch
import numpy as np
from PIL import Image, ImageOps
from model import load_model

# Set Streamlit app configuration
st.set_page_config(page_title="MNIST PyTorch Classifier", layout="centered")

# Load the trained model
model = load_model()

# Title
st.title("🧠 PyTorch MNIST Digit Recognizer")
st.markdown("Draw a digit (0–9) below:")

# Canvas for drawing
canvas = st_canvas(
    fill_color="#000000",
    stroke_width=12,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas"
)

# Predict button
if st.button("Predict"):
    if canvas.image_data is not None:
        # Preprocess the image
        img = Image.fromarray((canvas.image_data[:, :, 0]).astype(np.uint8))
        img = img.convert("L")  # Convert to grayscale
        img = ImageOps.invert(img)

        # Resize with compatible resampling method
        try:
            resample = Image.Resampling.LANCZOS  # Pillow ≥10
        except AttributeError:
            resample = Image.ANTIALIAS  # Pillow <10
        
        img = img.resize((28, 28), resample)

        # Convert to tensor
        img_tensor = torch.tensor(np.array(img), dtype=torch.float32).unsqueeze(0).unsqueeze(0) / 255.0

        # Predict
        with torch.no_grad():
            output = model(img_tensor)
            pred = torch.argmax(output, dim=1).item()

        st.success(f"Predicted Digit: **{pred}**")

# Clear canvas
if st.button("Clear"):
    st.rerun()
