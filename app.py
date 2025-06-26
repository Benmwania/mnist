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
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
        x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
        x_test = x_test.reshape(-1, 28, 28, 1) / 255.0

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

st.title("🧠 MNIST Handwritten Digit Recognizer")
st.write("Draw a digit (0-9) and click Predict. Use Clear to erase and retry.")

# Clear button using canvas key reset
if "canvas_key" not in st.session_state:
    st.session_state.canvas_key = "canvas1"

if st.button("🧹 Clear Canvas"):
    st.session_state.canvas_key = "canvas" + str(np.random.randint(1000))

# Canvas
canvas_result = st_canvas(
    fill_color="#000000",
    stroke_width=12,
    stroke_color="#FFFFFF",
    background_color="#000000",
    height=280,
    width=280,
    drawing_mode="freedraw",
    key=st.session_state.canvas_key,
)

# Prediction
if canvas_result.image_data is not None:
    # Extract & preprocess image
    img = canvas_result.image_data[:, :, 0:3]  # Drop alpha if present
    img = Image.fromarray((img[:, :, 0]).astype(np.uint8)).convert("L")  # Convert to grayscale
    img = ImageOps.invert(img)  # Invert: white digit on black background
    img = img.resize((28, 28))  # Resize to MNIST size
    img_array = np.array(img).reshape(1, 28, 28, 1).astype("float32") / 255.0  # Normalize

    # Predict
    prediction = model.predict(img_array)
    predicted_digit = int(np.argmax(prediction))

    st.subheader(f"🔢 Predicted Digit: {predicted_digit}")
    st.bar_chart(prediction[0])
