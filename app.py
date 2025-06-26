# app.py

import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
from streamlit_drawable_canvas import st_canvas
import os

st.set_page_config(page_title="MNIST Digit Recognizer", layout="centered")

MODEL_PATH = "model/mnist_cnn.h5"

# Load or train model
@st.cache_resource
def load_or_train_model():
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
    else:
        # Load and preprocess data
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
        x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

        # Build model
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(10, activation='softmax')
        ])
        model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))
        os.makedirs("model", exist_ok=True)
        model.save(MODEL_PATH)

    return model

model = load_or_train_model()

# Title
st.title("🧠 MNIST Handwritten Digit Recognizer")
st.write("Draw a digit (0-9) below and let the model predict it!")

# Create canvas
canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=12,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key="canvas",
)

# Predict from drawing
if canvas_result.image_data is not None:
    img = Image.fromarray((canvas_result.image_data[:, :, 0]).astype('uint8'))
    img = img.resize((28, 28))
    img = ImageOps.invert(img)
    img = img.convert('L')
    img_array = np.array(img).reshape(1, 28, 28, 1) / 255.0

    prediction = model.predict(img_array)
    predicted_digit = np.argmax(prediction)

    st.subheader(f"Predicted Digit: {predicted_digit}")
    st.bar_chart(prediction[0])

