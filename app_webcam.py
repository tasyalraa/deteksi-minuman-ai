import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import json
import datetime

# =========================
# SAFETY CHECK FILE
# =========================
if not os.path.exists("models/model_minuman.h5"):
    st.error("Model tidak ditemukan")
    st.stop()

if not os.path.exists("class_indices.json"):
    st.error("class_indices.json tidak ditemukan")
    st.stop()

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model("models/model_minuman.h5")

# =========================
# LOAD CLASS LABEL
# =========================
with open("class_indices.json", "r") as f:
    class_indices = json.load(f)

class_names = [None] * len(class_indices)
for name, idx in class_indices.items():
    class_names[idx] = name

# =========================
# DATASET FOLDER
# =========================
BASE_DIR = "dataset_tambahan"

for cls in class_names:
    os.makedirs(os.path.join(BASE_DIR, cls), exist_ok=True)

# =========================
# PREPROCESS
# =========================
def preprocess(img):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    return np.expand_dims(img, axis=0)

# =========================
# UI
# =========================
st.title("🔥 Webcam AI Dataset Collector PRO")

img_file = st.camera_input("Ambil gambar")

if img_file is not None:

    # LOAD IMAGE
    img = Image.open(img_file).convert("RGB")

    # FIX MIRROR ISSUE
    img = img.transpose(Image.FLIP_LEFT_RIGHT)

    st.image(img, caption="Preview")

    # PREPROCESS
    img_array = preprocess(img)

    # PREDICTION
    predictions = model.predict(img_array, verbose=0)[0]

    pred_idx = np.argmax(predictions)
    pred_class = class_names[pred_idx]
    confidence = float(predictions[pred_idx]) * 100

    # =========================
    # SHOW RESULT
    # =========================
    st.subheader("Hasil Prediksi")
    st.write(f"Class: **{pred_class}**")
    st.write(f"Confidence: **{confidence:.2f}%**")

    # =========================
    # CONFIDENCE RULE SYSTEM
    # =========================
    THRESHOLD = 70  # batas aman

    if confidence >= THRESHOLD:
        st.success("Model cukup yakin")

        if st.button("💾 Simpan Auto (Prediksi Benar)"):
            label = pred_class

            filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
            save_path = os.path.join(BASE_DIR, label, filename)

            img.save(save_path)

            st.success(f"Tersimpan ke: {save_path}")

    else:
        st.warning("Model kurang yakin → WAJIB manual label")

        selected_label = st.selectbox("Pilih label yang benar", class_names)

        if st.button("💾 Simpan Manual"):
            filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
            save_path = os.path.join(BASE_DIR, selected_label, filename)

            img.save(save_path)

            st.success(f"Tersimpan ke: {save_path}")