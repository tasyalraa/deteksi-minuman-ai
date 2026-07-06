import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import json
import os
import datetime
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
# DATASET FOLDER
# =========================
BASE_DIR = "dataset_tambahan"

for cls in class_names:
    os.makedirs(os.path.join(BASE_DIR, cls), exist_ok=True)

# =========================
# PREPROCESS (FIXED IMPORTANT)
# =========================
def preprocess(img):
    img = img.resize((224, 224))
    img = np.array(img)
    img = preprocess_input(img)   # 🔥 INI FIX UTAMA
    return np.expand_dims(img, axis=0)

def preprocess_webcam(img):
    img = img.resize((224,224))
    img = np.array(img)/255.0
    return np.expand_dims(img,0)

# =========================
# UI
# =========================
# =========================
# UI UPGRADED (ONLY VISUAL)
# =========================
st.set_page_config(
    page_title="Beverage Detector",
    page_icon="☕",
    layout="wide"
)

st.markdown("""
<style>

/* Background utama */
.stApp{
    background:#EEF2F7;
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background:#27374D;
    border-right:none;
    padding-top: 0.5rem;
}

/* Tulisan sidebar */
section[data-testid="stSidebar"] *{
    color:white !important;
}

/* Area utama */
.main .block-container{
    padding-top:2rem;
}

/* Card */
.card{
    background:#FFFFFF;
    border-radius:16px;
    padding:22px;
    border:none;
    box-shadow:0 4px 12px rgba(0,0,0,.08);
}

/* Judul */
h1{
    color:#27374D;
}

h2,h3{
    color:#374151;
}

/* Caption */
.stCaption{
    color:#6B7280;
}

/* Header transparan */
[data-testid="stHeader"]{
    background: transparent;
}

/* Hilangkan shadow */
[data-testid="stHeader"]::before{
    background: transparent;
}

/* Geser konten sedikit ke atas */
.block-container{
    padding-top: 1rem;
}

/* Sidebar Button */

section[data-testid="stSidebar"] button{

    background:#374B63;

    color:white;

    border:none;

    border-radius:10px;

    margin-bottom:8px;

    height:45px;

    text-align:left;

    font-weight:600;

}

section[data-testid="stSidebar"] button:hover{

    background:#4B6382;

    color:white;

    transition:.2s;

}
                        
</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR MENU
# =========================

if "menu" not in st.session_state:
    st.session_state.menu = "Dashboard"

st.sidebar.markdown("""
# ☕ Beverage Detector
<small style="color:#d1d5db;">Deep Learning Classification</small>
""", unsafe_allow_html=True)

if st.sidebar.button("🏠 Dashboard", use_container_width=True):
    st.session_state.menu = "Dashboard"

if st.sidebar.button("📤 Upload Image", use_container_width=True):
    st.session_state.menu = "Upload"

if st.sidebar.button("📷 Webcam Detection", use_container_width=True):
    st.session_state.menu = "Webcam"


st.sidebar.divider()
st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.caption("Version 1.0")

menu = st.session_state.menu

if menu == "Dashboard":

    st.title("☕ Beverage Detector")

    st.write(
        """
Selamat datang di aplikasi klasifikasi jenis minuman.

Silakan pilih menu di sidebar untuk mulai menggunakan aplikasi.
"""
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
<div class="card">
<h3>📤 Upload Image</h3>

Upload gambar dari komputer
untuk dilakukan klasifikasi.

</div>
""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
<div class="card">
<h3>📷 Webcam Detection</h3>

Gunakan kamera secara langsung
untuk melakukan prediksi.

</div>
""", unsafe_allow_html=True)

    st.divider()

    c1,c2,c3 = st.columns(3)

    c1.metric("Model","MobileNetV2")

    c2.metric("Framework","TensorFlow")

    c3.metric("Interface","Streamlit")

elif menu == "Upload":

    st.title("📤 Upload Image")
    st.caption("Upload gambar untuk mendeteksi jenis minuman")
    st.divider()

    # =========================
    # UPLOAD
    # =========================
    uploaded_file = st.file_uploader("📤 Upload gambar", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:

        img = Image.open(uploaded_file).convert("RGB")

        # layout 2 kolom biar profesional
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("### 📷 Input Image")
            st.image(img)

        # =========================
        # PROCESSING (TIDAK DIUBAH)
        # =========================
        img_array = preprocess(img)
        predictions = model.predict(img_array)[0]

        top_idx = np.argmax(predictions)

        with col2:
            st.markdown("### 🎯 Hasil Prediksi")

            confidence = predictions[top_idx] * 100

            st.markdown(f"""
            <div class="card">
                <h3>{class_names[top_idx]}</h3>
                <p><b>Confidence</b></p>
            </div>
            """, unsafe_allow_html=True)

            st.progress(float(predictions[top_idx]))

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )

        st.divider()

        # =========================
        # FULL RESULT LIST
        # =========================
        st.subheader("📊 Semua Probabilitas")

        for i, cls in enumerate(class_names):

            st.write(cls)

            st.progress(float(predictions[i]))

            st.caption(f"{predictions[i]*100:.2f}%")

elif menu == "Webcam":

    st.header("📷 Webcam Detection")
    st.caption("Gunakan kamera untuk mendeteksi jenis minuman")
    st.divider()

    img_file = st.camera_input("Ambil gambar")

    if img_file is not None:

        # LOAD IMAGE
        img = Image.open(img_file).convert("RGB")

        # FIX MIRROR ISSUE
        img = img.transpose(Image.FLIP_LEFT_RIGHT)

        st.image(img, caption="Preview", use_container_width=True)

        # PREPROCESS
        img_array = preprocess_webcam(img)

        # PREDICTION
        predictions = model.predict(img_array, verbose=0)[0]

        pred_idx = np.argmax(predictions)
        pred_class = class_names[pred_idx]
        confidence = float(predictions[pred_idx]) * 100

        st.subheader("Hasil Prediksi")

        st.metric(
            "Jenis Minuman",
            pred_class
        )

        st.metric(
            "Confidence",
            f"{confidence:.2f}%"
        )

        THRESHOLD = 70

        if confidence >= THRESHOLD:

            st.success("Model cukup yakin")

            if st.button("💾 Simpan Auto"):

                filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"

                save_path = os.path.join(BASE_DIR, pred_class, filename)

                img.save(save_path)

                st.success(f"Tersimpan ke {save_path}")

        else:

            st.warning("Model kurang yakin")

            selected_label = st.selectbox(
                "Pilih label yang benar",
                class_names
            )

            if st.button("💾 Simpan Manual"):

                filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"

                save_path = os.path.join(BASE_DIR, selected_label, filename)

                img.save(save_path)

                st.success(f"Tersimpan ke {save_path}")