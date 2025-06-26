# MNIST Digit Recognizer (PyTorch + Streamlit)

A Streamlit app that lets users draw a digit (0–9) and predicts it using a trained PyTorch CNN model.

## 🔧 Tech Stack
- Python
- PyTorch
- Streamlit
- streamlit-drawable-canvas

## 🧠 How It Works
- A Convolutional Neural Network (CNN) is trained on MNIST dataset.
- The user draws a digit using a canvas.
- The digit is processed, resized, and passed to the CNN model.
- The model returns its prediction with confidence.

## 🚀 Run Locally

```bash
pip install -r requirements.txt
python model.py      # train and save the model
streamlit run app.py
