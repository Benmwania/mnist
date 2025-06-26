# 🧠 MNIST Handwritten Digit Recognizer

This project is a Streamlit-based web app that allows users to draw a digit (0–9) and get real-time predictions using a Convolutional Neural Network (CNN) trained on the MNIST dataset.

---

## 📌 Assignment Objective

> **Practical Assignment 2**:  
> Build, train, and evaluate a machine learning or deep learning model for handwritten digit classification using the MNIST dataset.

---

## 🚀 Features

- Draw digits directly in the app
- Predicts digits 0–9 using a trained CNN
- "Clear Canvas" button to reset and try again
- Live probability chart of all digit classes
- Trains the model if not found (auto-saves it)

---

## 🧠 Model Details

- Framework: TensorFlow / Keras
- Model type: Deep Learning (CNN)
- Dataset: MNIST (60,000 training / 10,000 testing)
- Input shape: 28x28 grayscale
- Output: 10-class softmax (digits 0–9)

---

## 🛠️ Technologies Used

- Python
- Streamlit
- TensorFlow / Keras
- NumPy
- Pillow
- streamlit-drawable-canvas

---

## 📦 Installation

1. Clone the repository:

```bash
git clone https://github.com/your-username/mnist-streamlit.git
cd mnist-streamlit
