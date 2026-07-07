import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import json
import os
import datetime
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from streamlit_option_menu import option_menu

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/model_minuman.h5")

model = load_model()

# =========================
# CHECK CLASS FILE
# =========================
if not os.path.exists("class_indices.json"):
    st.error("class_indices.json tidak ditemukan. Jalankan train.py dulu.")
    st.stop()

@st.cache_data
def load_classes():
    with open("class_indices.json", "r") as f:
        return json.load(f)

class_indices = load_classes()

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
    
    page_title="Deteksi Jenis Minuman",
    page_icon="☕",
    layout="wide"
)

st.markdown("""
<link rel="stylesheet"
href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.7.2/css/all.min.css">
""", unsafe_allow_html=True)

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
    transition:0.25s ease;
}

.card:hover{
    transform:translateY(-4px);
    box-shadow:0 10px 22px rgba(0,0,0,.12);
}

/* STAT CARD */

.stat-card{
    background:#FFFFFF;
    border-radius:16px;
    padding:24px 20px;
    text-align:center;
    box-shadow:0 4px 12px rgba(0,0,0,.08);
    transition:.25s ease;
}

.stat-card:hover{
    transform:translateY(-4px);
    box-shadow:0 10px 22px rgba(0,0,0,.12);
}

.stat-card i{
    font-size:34px;
    color:#27374D;
    margin-bottom:15px;
}

.stat-title{
    color:#6B7280;
    font-size:14px;
    margin-bottom:8px;
}

.stat-value{
    color:#27374D;
    font-size:22px;
    font-weight:700;
}

/* Judul */
h1{
    color:#27374D;
    margin-bottom:20px;
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

/* Progress bar */
.stProgress > div > div > div > div{
    background:#2563EB;
}      

.stButton>button{
    width:100%;
    border-radius:10px;
    border:none;
    background:#27374D;
    color:white;
    font-weight:600;
    transition:.2s;
}

.stButton>button:hover{
    background:#1E293B;
    color:white;
}

[data-testid="stFileUploader"]{
    border:2px dashed #BFC8D6;
    border-radius:15px;
    padding:15px;
    background:white;
}             
</style>
""", unsafe_allow_html=True)

# =========================
# SIDEBAR MENU
# =========================

st.sidebar.markdown("""
<h2 style="margin-bottom:0;color:white;">
Deteksi Jenis Minuman
</h2>

<p style="color:#CBD5E1;font-size:14px;">
Aplikasi klasifikasi jenis minuman berbasis deep learning.
</p>
""", unsafe_allow_html=True)

with st.sidebar:
    selected = option_menu(
        menu_title=None,
        options=[
            "Dashboard",
            "Unggah Gambar",
            "Webcam Detection"
        ],
        icons=[
            "house-fill",
            "cloud-upload-fill",
            "camera-fill"
        ],
        default_index=0,
        styles={
            "container": {
                "padding": "0!important",
                "background-color": "#27374D",
            },
            "icon": {
                "color": "white",
                "font-size": "18px",
            },
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "4px 0",
                "padding": "10px",
                "border-radius": "10px",
                "color": "white",
            },
            "nav-link-selected": {
                "background-color": "#4B6382",
            },
        },
    )

st.sidebar.divider()
st.sidebar.caption("Versi 1.0")

# Mapping menu
if selected == "Dashboard":
    menu = "Dashboard"
elif selected == "Unggah Gambar":
    menu = "Upload"
else:
    menu = "Webcam"

if menu == "Dashboard":

    st.markdown("""
    <h1><i class="fa-solid fa-mug-hot"></i> Deteksi Jenis Minuman</h1>
    """, unsafe_allow_html=True)

    st.write("")

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
<h3><i class="fa-solid fa-upload"></i> Masukkan Gambar Minuman</h3>

Masukkan gambar dari komputer
untuk dilakukan klasifikasi.

</div>
""", unsafe_allow_html=True)

    with col2:
        st.markdown("""
<div class="card">
<h3><i class="fa-solid fa-camera"></i> Deteksi Minuman via Webcam </h3>

Gunakan kamera secara langsung
untuk melakukan prediksi.

</div>
""", unsafe_allow_html=True)

    st.divider()

    c1, c2, c3, c4 = st.columns(4)

    stats = [
        ("fa-solid fa-microchip", "Model", "MobileNetV2"),
        ("fa-solid fa-gears", "Framework", "TensorFlow"),
        ("fa-solid fa-display", "Interface", "Streamlit"),
        ("fa-solid fa-layer-group", "Kelas", str(len(class_names))),
    ]

    for col, (icon, title, value) in zip([c1, c2, c3, c4], stats):
        with col:
            st.markdown(f"""
            <div class="stat-card">
                <i class="{icon}"></i>
                <div class="stat-title">{title}</div>
                <div class="stat-value">{value}</div>
            </div>
            """, unsafe_allow_html=True)

elif menu == "Upload":

    st.markdown("""
    <h1><i class="fa-solid fa-cloud-arrow-up"></i> Unggah Gambar</h1>
    """, unsafe_allow_html=True)

    st.caption("Unggah gambar minuman untuk dilakukan klasifikasi menggunakan model AI.")
    st.divider()

    # =========================
    # UPLOAD
    # =========================
    uploaded_file = st.file_uploader(
        "Pilih file gambar",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:

        img = Image.open(uploaded_file).convert("RGB")

        # layout 2 kolom biar profesional
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("""
            <h3><i class="fa-solid fa-image"></i> Gambar Masukan</h3>
            """, unsafe_allow_html=True)
            st.image(
            img,
            caption="Gambar yang diunggah",
            use_container_width=True
        )

        # =========================
        # PROCESSING (TIDAK DIUBAH)
        # =========================
        img_array = preprocess(img)
        with st.spinner("Sedang mendeteksi gambar..."):
            predictions = model.predict(img_array)[0]

        top_indices = np.argsort(predictions)[::-1]
        top_idx = top_indices[0]

        with col2:
            st.markdown("""
            <h3><i class="fa-solid fa-crosshairs"></i> Hasil Prediksi</h3>
            """, unsafe_allow_html=True)

            confidence = predictions[top_idx] * 100

            st.markdown(f"""
            <div class="card">

            <h2 style="margin-bottom:10px;">
            {class_names[top_idx]}
            </h2>

            <p style="color:#6B7280;">
            Prediction Confidence
            </p>

            <h3 style="color:#2563EB;">
            {confidence:.2f}%
            </h3>

            </div>
            """, unsafe_allow_html=True)

            st.progress(float(predictions[top_idx]))

        st.divider()

        # =========================
        # FULL RESULT LIST
        # =========================
        st.markdown("""
        <h3><i class="fa-solid fa-ranking-star"></i> Tiga Prediksi Teratas</h3>
        """, unsafe_allow_html=True)

        medals = ["🥇", "🥈", "🥉"]

        for rank, idx in enumerate(top_indices[:3]):

            confidence = float(predictions[idx]) * 100

            st.write(f"{medals[rank]} {class_names[idx]}")

            st.progress(float(predictions[idx]))

            st.caption(f"{confidence:.2f}%")

elif menu == "Webcam":

    st.markdown("""
    <h1><i class="fa-solid fa-camera"></i> Deteksi Melalui Webcam</h1>
    """, unsafe_allow_html=True)

    st.caption("Ambil gambar menggunakan kamera untuk mendeteksi jenis minuman.")
    st.divider()

    img_file = st.camera_input("Ambil gambar")

    if img_file is not None:

        # LOAD IMAGE
        img = Image.open(img_file).convert("RGB")

        st.image(
            img,
            caption="Gambar dari webcam",
            use_container_width=True
        )

        # PREPROCESS
        img_array = preprocess(img)

        # PREDICTION
        with st.spinner("Sedang mendeteksi gambar..."):
            predictions = model.predict(img_array, verbose=0)[0]

        pred_idx = np.argmax(predictions)
        pred_class = class_names[pred_idx]
        confidence = float(predictions[pred_idx]) * 100

        st.markdown("""
        <h3><i class="fa-solid fa-crosshairs"></i> Hasil Prediksi</h3>
        """, unsafe_allow_html=True)

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

            st.success("Model memiliki tingkat keyakinan yang tinggi.")

            if st.button("Simpan Otomatis"):

                filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"

                save_path = os.path.join(BASE_DIR, pred_class, filename)

                img.save(save_path)

                st.success(f"Tersimpan ke {save_path}")

        else:

            st.warning("Tingkat keyakinan model masih rendah. Silakan pilih label yang benar.")

            selected_label = st.selectbox(
                "Pilih label yang benar",
                class_names
            )

            if st.button("Simpan Manual"):

                filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"

                save_path = os.path.join(BASE_DIR, selected_label, filename)

                img.save(save_path)

                st.success(f"Tersimpan ke {save_path}")