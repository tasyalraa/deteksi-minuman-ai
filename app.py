import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import json
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model("models/model_minuman.h5")

# =========================
# CHECK CLASS FILE
# =========================
if not os.path.exists("class_indices.json"):
    st.error("class_indices.json tidak ditemukan. Jalankan train.py dulu.")
    st.stop()

with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

class_names = [None] * len(class_indices)
for name, idx in class_indices.items():
    class_names[idx] = name

# =========================
# PREPROCESS (FIXED IMPORTANT)
# =========================
def preprocess(img):
    img = img.resize((224, 224))
    img = np.array(img)
    img = preprocess_input(img)   # 🔥 INI FIX UTAMA
    return np.expand_dims(img, axis=0)

# =========================
# UI
# =========================
st.title("🍹 Prediksi Minuman AI (FIXED VERSION)")

uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Input Image")

    # preprocess
    img_array = preprocess(img)

    # prediction
    predictions = model.predict(img_array)[0]

    st.subheader("Hasil Prediksi:")

    for i, cls in enumerate(class_names):
        st.write(f"{cls}: {predictions[i]*100:.2f}%")

    top_idx = np.argmax(predictions)
    st.success(f"Prediksi: {class_names[top_idx]} ({predictions[top_idx]*100:.2f}%)")