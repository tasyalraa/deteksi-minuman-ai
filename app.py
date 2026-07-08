import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
import json
import os
import datetime
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# =========================================================
# LOAD MODEL  (LOGIKA TIDAK DIUBAH)
# =========================================================
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("models/model_minuman.h5")

model = load_model()

# =========================================================
# CHECK CLASS FILE  (LOGIKA TIDAK DIUBAH)
# =========================================================
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

# =========================================================
# DATASET FOLDER  (LOGIKA TIDAK DIUBAH)
# =========================================================
BASE_DIR = "dataset_tambahan"

for cls in class_names:
    os.makedirs(os.path.join(BASE_DIR, cls), exist_ok=True)

# =========================================================
# PREPROCESS  (LOGIKA TIDAK DIUBAH)
# =========================================================
def preprocess(img):
    img = img.resize((224, 224))
    img = np.array(img)
    img = preprocess_input(img)   # FIX UTAMA - JANGAN DIUBAH
    return np.expand_dims(img, axis=0)

def preprocess_webcam(img):
    img = img.resize((224, 224))
    img = np.array(img) / 255.0
    return np.expand_dims(img, 0)

# =========================================================
# KONFIGURASI HALAMAN
# =========================================================
st.set_page_config(
    page_title="AI Beverage Detector",
    page_icon="☕",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =========================================================
# PALET WARNA (dipakai berulang di CSS & komponen)
# TEMA BARU: Latar utama CREAM HANGAT (serasi dgn beige sidebar,
# tapi TIDAK menyatu dengan sidebar), aksen COFFEE/CARAMEL/GOLD
# (tidak ada hijau maupun teal sama sekali)
# =========================================================
SIDEBAR_BG        = "#F1EAD8"   # light beige (tetap, sesuai referensi)
SIDEBAR_BG_HOVER  = "#E7DEC6"
BG_MAIN           = "#FAF5EA"   # cream hangat - serasi dgn beige tapi lebih terang & beda saturasi dr sidebar
PRIMARY           = "#8B5E34"   # coffee brown (pengganti teal)
PRIMARY_DARK      = "#6E4827"
SECONDARY         = "#C08552"   # caramel terang
SUCCESS           = "#B8860B"   # dark goldenrod (pengganti hijau utk status "high confidence")
WARNING           = "#C2661C"   # burnt orange
DANGER            = "#B3271D"   # brick red
CARD_BG           = "#FFFFFF"   # kartu tetap putih agar menonjol di atas cream
BORDER            = "#E8E0CF"   # border lembut senada beige
TEXT_MUTED        = "#6B6455"
TEXT_DARK         = "#2E2A22"

# warna khusus judul hero (di atas gradient coklat gelap -> harus kontras tinggi)
HERO_TITLE_COLOR  = "#FFE9C7"   # krem-emas hangat, kontras tinggi di atas gradient coklat

# warna khusus sidebar (karena latarnya terang, teks harus gelap)
SIDEBAR_TEXT       = "#3B3625"
SIDEBAR_TEXT_MUTED = "#8A8065"
SIDEBAR_BORDER     = "#DCD3B8"

# =========================================================
# EXTERNAL ASSETS: FONTS + ICON LIBRARIES
# =========================================================
st.markdown("""
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&family=Nunito:wght@400;600;700&display=swap" rel="stylesheet">
<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.7.2/css/all.min.css">
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap-icons@1.11.3/font/bootstrap-icons.min.css">
""", unsafe_allow_html=True)

# =========================================================
# GLOBAL CSS - DESAIN DASHBOARD AI MODERN (TEMA TERANG, COFFEE)
# =========================================================
st.markdown(f"""
<style>

/* ---------- RESET & BASE ---------- */
html, body {{
    font-family: 'Poppins', 'Inter', 'Nunito', sans-serif;
}}

.stApp, .stApp p, .stApp span, .stApp label, .stApp div, .stApp button {{
    font-family: 'Poppins', 'Inter', 'Nunito', sans-serif;
}}

/* Jangan timpa font ikon bawaan Streamlit (panah collapse sidebar, dll) */
[data-testid="stIconMaterial"] {{
    font-family: 'Material Symbols Rounded' !important;
}}

.stApp {{
    background: {BG_MAIN};
}}

.main .block-container {{
    padding-top: 1.6rem;
    padding-bottom: 3rem;
    max-width: 1250px;
}}

[data-testid="stHeader"] {{
    background: transparent;
}}

#MainMenu, footer {{ visibility: hidden; }}

h1, h2, h3, h4 {{
    font-family: 'Poppins', sans-serif;
    color: {TEXT_DARK};
    font-weight: 700;
}}

p, span, label, li {{
    color: {TEXT_MUTED};
}}

hr {{
    border-color: {BORDER};
}}

/* ---------- ANIMATIONS ---------- */
@keyframes fadeIn {{
    from {{ opacity: 0; }}
    to   {{ opacity: 1; }}
}}
@keyframes slideUp {{
    from {{ opacity: 0; transform: translateY(24px); }}
    to   {{ opacity: 1; transform: translateY(0); }}
}}
.anim-fade  {{ animation: fadeIn .6s ease both; }}
.anim-slide {{ animation: slideUp .55s cubic-bezier(.4,0,.2,1) both; }}
.d1 {{ animation-delay: .05s; }}
.d2 {{ animation-delay: .12s; }}
.d3 {{ animation-delay: .19s; }}
.d4 {{ animation-delay: .26s; }}

/* ---------- SIDEBAR ---------- */
section[data-testid="stSidebar"] {{
    background: {SIDEBAR_BG};
    border-right: 1px solid {SIDEBAR_BORDER};
}}

section[data-testid="stSidebar"] * {{
    color: {SIDEBAR_TEXT} !important;
    font-family: 'Poppins', sans-serif;
}}

section[data-testid="stSidebar"] .block-container {{
    padding-top: 1.6rem;
}}

.sidebar-brand {{
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 0 4px 18px 4px;
}}

.sidebar-brand .brand-icon {{
    width: 46px;
    height: 46px;
    border-radius: 14px;
    background: linear-gradient(135deg, {PRIMARY}, {SECONDARY});
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 20px;
    color: white !important;
    box-shadow: 0 6px 14px rgba(139,94,52,.25);
}}

.sidebar-brand .brand-text h3 {{
    margin: 0;
    font-size: 17px;
    font-weight: 700;
    color: {SIDEBAR_TEXT} !important;
    line-height: 1.2;
}}

.sidebar-brand .brand-text span {{
    font-size: 12px;
    color: {SIDEBAR_TEXT_MUTED} !important;
}}

.sidebar-divider {{
    border: none;
    border-top: 1px solid {SIDEBAR_BORDER};
    margin: 4px 0 16px 0;
}}

.sidebar-label {{
    font-size: 11px;
    letter-spacing: .08em;
    text-transform: uppercase;
    color: {SIDEBAR_TEXT_MUTED} !important;
    margin: 4px 0 8px 4px;
    font-weight: 600;
}}

/* ---------- SEMUA TOMBOL (SIDEBAR & KONTEN) DISAMAKAN WARNANYA ---------- */
/* Tombol nav di sidebar sekarang solid (tidak transparan lagi) agar
   konsisten dengan tombol-tombol lain di konten utama */
section[data-testid="stSidebar"] div[data-testid="stButton"] {{
    margin-bottom: 8px;
}}

section[data-testid="stSidebar"] div[data-testid="stButton"] button {{
    background: {PRIMARY} !important;
    color: #FFFFFF !important;
    border: none;
    border-left: 4px solid transparent;
    border-radius: 10px;
    height: 46px;
    width: 100%;
    text-align: left;
    font-size: 15px;
    font-weight: 600;
    padding-left: 16px;
    box-shadow: 0 4px 10px rgba(139,94,52,.20);
    transition: all .2s ease;
}}

section[data-testid="stSidebar"] div[data-testid="stButton"] button p {{
    color: #FFFFFF !important;
}}

section[data-testid="stSidebar"] div[data-testid="stButton"] button:hover {{
    background: {PRIMARY_DARK} !important;
    transform: translateX(3px);
    color: #FFFFFF !important;
    box-shadow: 0 8px 16px rgba(110,72,39,.28);
}}

section[data-testid="stSidebar"] div[data-testid="stButton"] button:focus:not(:active) {{
    box-shadow: 0 4px 10px rgba(139,94,52,.20);
}}

.sidebar-footer {{
    position: relative;
    margin-top: 22px;
    font-size: 12px;
    color: {SIDEBAR_TEXT_MUTED} !important;
}}

/* ---------- CARD ---------- */
.ai-card {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 20px;
    padding: 24px;
    box-shadow: 0 2px 10px rgba(46,42,34,.06);
    transition: all .25s ease;
}}

.ai-card:hover {{
    transform: translateY(-4px);
    box-shadow: 0 14px 28px rgba(46,42,34,.10);
}}

/* ---------- HERO SECTION ---------- */
.hero-section {{
    background: linear-gradient(120deg, #4A3320 0%, #6E4827 58%, #8B5E34 145%);
    border: 1px solid {BORDER};
    border-radius: 24px;
    padding: 46px 44px;
    color: white;
    position: relative;
    overflow: hidden;
    box-shadow: 0 16px 34px rgba(139,94,52,.20);
    margin-bottom: 26px;
}}

.hero-section::after {{
    content:"";
    position:absolute;
    right:-60px; top:-60px;
    width:220px; height:220px;
    background: radial-gradient(circle, rgba(255,255,255,.14) 0%, rgba(255,255,255,0) 70%);
    border-radius: 50%;
}}

.hero-badge {{
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(255,255,255,.14);
    border: 1px solid rgba(255,255,255,.28);
    padding: 6px 14px;
    border-radius: 999px;
    font-size: 13px;
    margin-bottom: 16px;
    color: #F5EFE4 !important;
}}

/* FIX: judul hero sekarang eksplisit krem-emas + text-shadow agar kontras
   tinggi & konsisten di atas gradient coklat gelap (sebelumnya terlihat
   pudar/keabu-abuan karena warnanya tertimpa) */
.hero-section h1,
.hero-section h1 * {{
    color: {HERO_TITLE_COLOR} !important;
    font-size: 34px;
    font-weight: 800;
    margin-bottom: 10px;
    text-shadow: 0 2px 10px rgba(0,0,0,.35);
    -webkit-text-fill-color: {HERO_TITLE_COLOR};
}}

.hero-section p {{
    color: #F1E7D8 !important;
    font-size: 15.5px;
    max-width: 560px;
    line-height: 1.6;
}}

/* ---------- STAT CARD ---------- */
/* FIX: box-sizing + flex-shrink + overflow supaya konten tidak "keluar
   jalur" (meluber/terpotong) ketika lebar kolom mengecil / teks panjang */
.stat-card {{
    box-sizing: border-box;
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 20px;
    padding: 20px;
    text-align: left;
    box-shadow: 0 2px 10px rgba(46,42,34,.06);
    transition: all .25s ease;
    display: flex;
    align-items: center;
    gap: 14px;
    height: 100%;
    min-height: 92px;
    overflow: hidden;
}}

.stat-card:hover {{
    transform: translateY(-4px);
    box-shadow: 0 14px 28px rgba(46,42,34,.10);
}}

.stat-icon {{
    box-sizing: border-box;
    width: 48px;
    height: 48px;
    min-width: 48px;
    flex-shrink: 0;
    border-radius: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    line-height: 1;
    color: white;
}}

.stat-text {{
    min-width: 0;
    flex: 1;
}}

.stat-title {{
    color: {TEXT_MUTED};
    font-size: 13px;
    margin-bottom: 4px;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}}

.stat-value {{
    color: {TEXT_DARK};
    font-size: 18px;
    font-weight: 700;
    line-height: 1.25;
    word-break: break-word;
}}

/* Menjaga tinggi kolom stat card tetap seragam di semua breakpoint */
div[data-testid="column"] {{
    display: flex;
    flex-direction: column;
}}
div[data-testid="column"] > div {{
    height: 100%;
}}

/* ---------- HOW-TO STEPS ---------- */
.step-row {{
    display: flex;
    align-items: center;
    gap: 6px;
    margin-top: 6px;
}}

.step-card {{
    flex: 1;
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 20px;
    padding: 26px 20px;
    text-align: center;
    box-shadow: 0 2px 10px rgba(46,42,34,.06);
    transition: all .25s ease;
}}

.step-card:hover {{
    transform: translateY(-4px);
    box-shadow: 0 14px 28px rgba(46,42,34,.10);
}}

.step-circle {{
    width: 54px;
    height: 54px;
    margin: 0 auto 14px auto;
    border-radius: 50%;
    background: linear-gradient(135deg, {PRIMARY}, {SECONDARY});
    color: white;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 22px;
    box-shadow: 0 8px 16px rgba(139,94,52,.22);
}}

.step-connector {{
    flex: 0 0 46px;
    height: 3px;
    background: repeating-linear-gradient(90deg, {SECONDARY} 0 8px, transparent 8px 14px);
    margin-top: -60px;
}}

.step-title {{
    font-weight: 700;
    color: {TEXT_DARK};
    margin-bottom: 6px;
    font-size: 15.5px;
}}

.step-desc {{
    font-size: 13px;
    color: {TEXT_MUTED};
}}

/* ---------- BADGE ---------- */
.badge {{
    display: inline-block;
    padding: 6px 14px;
    border-radius: 999px;
    font-size: 13px;
    font-weight: 600;
}}
.badge-success {{ background: rgba(184,134,11,.14); color: {SUCCESS}; }}
.badge-warning {{ background: rgba(194,102,28,.14); color: {WARNING}; }}
.badge-danger  {{ background: rgba(179,39,29,.14);  color: {DANGER}; }}

/* ---------- PREDICTION CARD ---------- */
.pred-emoji {{
    font-size: 46px;
    line-height: 1;
    margin-bottom: 6px;
}}
.pred-name {{
    font-size: 26px;
    font-weight: 700;
    color: {TEXT_DARK};
    margin-bottom: 2px;
}}
.pred-conf-label {{
    font-size: 13px;
    color: {TEXT_MUTED};
    margin-top: 14px;
}}
.pred-conf-value {{
    font-size: 30px;
    font-weight: 800;
    color: {PRIMARY};
    margin-bottom: 10px;
}}

/* ---------- CUSTOM PROGRESS BARS ---------- */
.cprog-item {{
    margin-bottom: 14px;
}}
.cprog-top {{
    display: flex;
    justify-content: space-between;
    font-size: 13.5px;
    font-weight: 600;
    color: {TEXT_DARK};
    margin-bottom: 6px;
}}
.cprog-track {{
    width: 100%;
    height: 12px;
    background: #F0EBDD;
    border: 1px solid {BORDER};
    border-radius: 999px;
    overflow: hidden;
}}
.cprog-fill {{
    height: 100%;
    border-radius: 999px;
    transition: width .6s ease;
}}

/* ---------- STATUS BOX (webcam) ---------- */
.status-box {{
    display:flex;
    align-items:center;
    gap:10px;
    padding: 12px 16px;
    border-radius: 14px;
    font-weight: 600;
    font-size: 14px;
    margin-bottom: 14px;
}}
.status-ready  {{ background: rgba(184,134,11,.10);  color: {SUCCESS}; }}
.status-wait   {{ background: rgba(179,39,29,.08);  color: {DANGER}; }}
.status-accept {{ background: rgba(184,134,11,.10);  color: {SUCCESS}; border:1px solid rgba(184,134,11,.30); }}
.status-uncertain {{ background: rgba(194,102,28,.10); color: {WARNING}; border:1px solid rgba(194,102,28,.30); }}

/* ---------- BUTTONS (Konten Utama) - warna disamakan dgn tombol sidebar ---------- */
.stButton > button {{
    width: 100%;
    border-radius: 12px;
    border: none;
    background: {PRIMARY} !important;
    color: #FFFFFF !important;
    font-weight: 600;
    padding: 0.55rem 1rem;
    transition: all .2s ease;
    box-shadow: 0 4px 10px rgba(139,94,52,.20);
}}

.stButton > button:hover {{
    background: {PRIMARY_DARK} !important;
    color: #FFFFFF !important;
    transform: translateY(-2px);
    box-shadow: 0 8px 16px rgba(110,72,39,.28);
}}

.stButton > button p {{
    color: #FFFFFF !important;
}}

/* ---------- FILE UPLOADER / CAMERA ---------- */
[data-testid="stFileUploader"] {{
    border: 2px dashed {BORDER};
    border-radius: 18px;
    padding: 18px;
    background: #FDFBF6;
    transition: all .2s ease;
}}

[data-testid="stFileUploader"]:hover {{
    border-color: {PRIMARY};
}}

[data-testid="stCameraInput"] {{
    border-radius: 18px;
    overflow: hidden;
}}

/* ---------- PROGRESS (native, fallback) ---------- */
.stProgress > div > div > div > div {{
    background: {PRIMARY};
}}

/* ---------- SECTION TITLE ---------- */
.section-title {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 30px 0 14px 0;
}}
.section-title i {{
    color: {PRIMARY};
    font-size: 20px;
}}
.section-title h3 {{
    margin: 0;
    font-size: 19px;
}}

</style>
""", unsafe_allow_html=True)

# =========================================================
# SESSION STATE - NAVIGASI (LOGIKA INTI TIDAK DIUBAH)
# =========================================================
if "menu" not in st.session_state:
    st.session_state.menu = "Dashboard"

# state tambahan murni untuk UX (reset uploader) - tidak memengaruhi logika prediksi
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# state tambahan opsional untuk histori prediksi webcam (murni tampilan)
if "webcam_history" not in st.session_state:
    st.session_state.webcam_history = []

MENU_ITEMS = ["Dashboard", "Upload", "Webcam"]

# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <div class="brand-icon"><i class="bi bi-cup-hot-fill"></i></div>
        <div class="brand-text">
            <h3>AI Beverage Detector</h3>
            <span>Deep Learning Classification</span>
        </div>
    </div>
    <hr class="sidebar-divider">
    <div class="sidebar-label">Menu</div>
    """, unsafe_allow_html=True)

    if st.button("📊  Dashboard", use_container_width=True, key="nav_dashboard"):
        st.session_state.menu = "Dashboard"
        st.rerun()

    if st.button("🖼️  Unggah Gambar", use_container_width=True, key="nav_upload"):
        st.session_state.menu = "Upload"
        st.rerun()

    if st.button("🎥  Webcam Detection", use_container_width=True, key="nav_webcam"):
        st.session_state.menu = "Webcam"
        st.rerun()

    st.markdown(f"""
    <div class="sidebar-footer">
        Model aktif: MobileNetV2<br>
        Total kelas: {len(class_names)}
    </div>
    """, unsafe_allow_html=True)

# Highlight tombol menu yang sedang aktif (murni CSS, tidak mengubah logika)
# Tombol aktif tetap memakai keluarga warna yang sama (PRIMARY_DARK),
# hanya ditambah border-left aksen agar tetap terlihat konsisten & tidak transparan
active_index = MENU_ITEMS.index(st.session_state.menu) + 1
st.markdown(f"""
<style>
section[data-testid="stSidebar"] div[data-testid="stButton"]:nth-of-type({active_index}) button {{
    background: {PRIMARY_DARK} !important;
    color: #FFFFFF !important;
    border-left: 4px solid {SECONDARY} !important;
    box-shadow: 0 4px 10px rgba(110,72,39,.30);
}}
section[data-testid="stSidebar"] div[data-testid="stButton"]:nth-of-type({active_index}) button p {{
    color: #FFFFFF !important;
}}
</style>
""", unsafe_allow_html=True)

menu = st.session_state.menu

# =========================================================
# HALAMAN: DASHBOARD
# =========================================================
if menu == "Dashboard":

    st.markdown(f"""
    <div class="hero-section anim-slide">
        <div class="hero-badge"><i class="fa-solid fa-bolt"></i> Powered by Deep Learning</div>
        <h1 style="color:{HERO_TITLE_COLOR} !important; -webkit-text-fill-color:{HERO_TITLE_COLOR}; text-shadow:0 2px 10px rgba(0,0,0,.35);">☕ AI Beverage Detection</h1>
        <p>Deteksi jenis minuman secara otomatis menggunakan Deep Learning MobileNetV2.
        Unggah gambar atau gunakan webcam untuk melihat AI bekerja secara real-time.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-title anim-slide"><i class="fa-solid fa-chart-simple"></i><h3>Ringkasan Sistem</h3></div>', unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    stats = [
        ("fa-solid fa-brain", PRIMARY, "Model", "MobileNetV2"),
        ("fa-solid fa-gears", SECONDARY, "Framework", "TensorFlow"),
        ("fa-solid fa-layer-group", SUCCESS, "Jumlah Kelas", str(len(class_names))),
        ("fa-solid fa-bullseye", WARNING, "Akurasi Model", "96%"),
    ]
    for col, (icon, color, title, value) in zip([c1, c2, c3, c4], stats):
        with col:
            st.markdown(f"""
            <div class="stat-card anim-slide">
                <div class="stat-icon" style="background:{color};"><i class="{icon}"></i></div>
                <div class="stat-text">
                    <div class="stat-title">{title}</div>
                    <div class="stat-value">{value}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown('<div class="section-title anim-slide"><i class="fa-solid fa-diagram-project"></i><h3>Cara Menggunakan</h3></div>', unsafe_allow_html=True)

    s1, arrow1, s2, arrow2, s3 = st.columns([3, 0.6, 3, 0.6, 3])
    with s1:
        st.markdown("""
        <div class="step-card anim-slide d1">
            <div class="step-circle">①</div>
            <div class="step-title"><i class="fa-solid fa-upload"></i> Upload Gambar</div>
            <div class="step-desc">Pilih gambar minuman dari galeri atau ambil langsung dari webcam.</div>
        </div>
        """, unsafe_allow_html=True)
    with arrow1:
        st.markdown('<div style="text-align:center;font-size:22px;color:#B7AC90;padding-top:40px;">→</div>', unsafe_allow_html=True)
    with s2:
        st.markdown("""
        <div class="step-card anim-slide d2">
            <div class="step-circle">②</div>
            <div class="step-title"><i class="fa-solid fa-microchip"></i> AI Memproses</div>
            <div class="step-desc">Model MobileNetV2 menganalisis gambar dan mengekstrak fitur visual.</div>
        </div>
        """, unsafe_allow_html=True)
    with arrow2:
        st.markdown('<div style="text-align:center;font-size:22px;color:#B7AC90;padding-top:40px;">→</div>', unsafe_allow_html=True)
    with s3:
        st.markdown("""
        <div class="step-card anim-slide d3">
            <div class="step-circle">③</div>
            <div class="step-title"><i class="fa-solid fa-circle-check"></i> Hasil Prediksi</div>
            <div class="step-desc">Sistem menampilkan jenis minuman beserta tingkat keyakinannya.</div>
        </div>
        """, unsafe_allow_html=True)

# =========================================================
# HALAMAN: UPLOAD GAMBAR
# =========================================================
elif menu == "Upload":

    st.markdown("""
    <div class="section-title anim-slide">
        <i class="fa-solid fa-cloud-arrow-up"></i>
        <h3 style="font-size:26px;">Unggah Gambar Minuman</h3>
    </div>
    <p style="color:#8A8065;margin-top:-8px;">Unggah gambar untuk diklasifikasikan oleh model AI secara otomatis.</p>
    """, unsafe_allow_html=True)

    st.write("")
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="ai-card anim-slide d1">', unsafe_allow_html=True)
        st.markdown("#### 📤 Drag & Drop Gambar")
        uploaded_file = st.file_uploader(
            "Atau klik untuk browse file",
            type=["jpg", "jpeg", "png"],
            key=f"uploader_{st.session_state.uploader_key}",
        )

        img = None
        if uploaded_file is not None:
            img = Image.open(uploaded_file).convert("RGB")
            st.markdown("**Preview Gambar**")
            st.image(img, use_container_width=True)

            if st.button("🔄  Upload Gambar Baru", key="reset_uploader"):
                st.session_state.uploader_key += 1
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        if uploaded_file is not None and img is not None:

            # =========================
            # PREDIKSI (LOGIKA TIDAK DIUBAH)
            # =========================
            img_array = preprocess(img)
            with st.spinner("AI sedang menganalisis gambar..."):
                predictions = model.predict(img_array)[0]

            top_indices = np.argsort(predictions)[::-1]
            top_idx = top_indices[0]
            confidence = float(predictions[top_idx]) * 100

            if confidence >= 80:
                badge_label, badge_class, bar_color = "High Confidence", "badge-success", SUCCESS
            elif confidence >= 50:
                badge_label, badge_class, bar_color = "Medium Confidence", "badge-warning", WARNING
            else:
                badge_label, badge_class, bar_color = "Low Confidence", "badge-danger", DANGER

            st.markdown(f"""
            <div class="ai-card anim-slide d2" style="text-align:center;">
                <div style="color:{TEXT_MUTED};font-size:13px;font-weight:600;letter-spacing:.05em;text-transform:uppercase;">Prediction</div>
                <div class="pred-emoji">☕</div>
                <div class="pred-name">{class_names[top_idx]}</div>
                <span class="badge {badge_class}">{badge_label}</span>
                <div class="pred-conf-label">Confidence</div>
                <div class="pred-conf-value">{confidence:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown('<div class="ai-card anim-slide d3" style="margin-top:18px;">', unsafe_allow_html=True)
            st.markdown('<div class="section-title" style="margin-top:0;"><i class="fa-solid fa-ranking-star"></i><h3>Top Prediksi</h3></div>', unsafe_allow_html=True)

            medals = ["🥇", "🥈", "🥉"]
            for rank, idx in enumerate(top_indices[:3]):
                conf_i = float(predictions[idx]) * 100
                st.markdown(f"""
                <div class="cprog-item">
                    <div class="cprog-top"><span>{medals[rank]} {class_names[idx]}</span><span>{conf_i:.0f}%</span></div>
                    <div class="cprog-track">
                        <div class="cprog-fill" style="width:{conf_i}%;background:{bar_color if rank==0 else SECONDARY};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="ai-card anim-slide d2" style="text-align:center;color:#B7AC90;padding:60px 20px;">
                <i class="fa-solid fa-image" style="font-size:40px;margin-bottom:12px;display:block;"></i>
                Hasil prediksi akan muncul di sini setelah gambar diunggah.
            </div>
            """, unsafe_allow_html=True)

# =========================================================
# HALAMAN: WEBCAM DETECTION
# =========================================================
elif menu == "Webcam":

    st.markdown("""
    <div class="section-title anim-slide">
        <i class="fa-solid fa-camera"></i>
        <h3 style="font-size:26px;">Deteksi Melalui Webcam</h3>
    </div>
    <p style="color:#8A8065;margin-top:-8px;">Ambil gambar langsung dari kamera untuk deteksi minuman secara real-time.</p>
    """, unsafe_allow_html=True)

    st.write("")
    col_left, col_right = st.columns([1, 1], gap="large")

    with col_left:
        st.markdown('<div class="ai-card anim-slide d1">', unsafe_allow_html=True)
        img_file = st.camera_input("Ambil gambar")

        if img_file is None:
            st.markdown('<div class="status-box status-wait">🔴 &nbsp; Waiting Camera</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-box status-ready">🟢 &nbsp; AI Ready</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_right:
        if img_file is not None:

            # LOAD IMAGE (LOGIKA TIDAK DIUBAH)
            img = Image.open(img_file).convert("RGB")

            # PREPROCESS (LOGIKA TIDAK DIUBAH)
            img_array = preprocess(img)

            # PREDICTION (LOGIKA TIDAK DIUBAH)
            with st.spinner("AI sedang menganalisis gambar..."):
                predictions = model.predict(img_array, verbose=0)[0]

            pred_idx = np.argmax(predictions)
            pred_class = class_names[pred_idx]
            confidence = float(predictions[pred_idx]) * 100

            if confidence >= 80:
                badge_label, badge_class = "High Confidence", "badge-success"
            elif confidence >= 50:
                badge_label, badge_class = "Medium Confidence", "badge-warning"
            else:
                badge_label, badge_class = "Low Confidence", "badge-danger"

            st.markdown(f"""
            <div class="ai-card anim-slide d2" style="text-align:center;">
                <div style="color:{TEXT_MUTED};font-size:13px;font-weight:600;letter-spacing:.05em;text-transform:uppercase;">Prediction</div>
                <div class="pred-emoji">☕</div>
                <div class="pred-name">{pred_class}</div>
                <span class="badge {badge_class}">{badge_label}</span>
                <div class="pred-conf-label">Confidence</div>
                <div class="pred-conf-value">{confidence:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

            st.write("")

            THRESHOLD = 70  # LOGIKA TIDAK DIUBAH

            if confidence >= THRESHOLD:
                st.markdown("""
                <div class="status-box status-accept anim-slide d3">
                    ✅ &nbsp; Prediction Accepted — Model memiliki tingkat keyakinan yang tinggi.
                </div>
                """, unsafe_allow_html=True)

                if st.button("💾  Simpan Otomatis", key="save_auto"):
                    filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
                    save_path = os.path.join(BASE_DIR, pred_class, filename)
                    img.save(save_path)
                    st.session_state.webcam_history.insert(0, f"{pred_class} • {confidence:.2f}% • {filename}")
                    st.success(f"Tersimpan ke {save_path}")

            else:
                st.markdown("""
                <div class="status-box status-uncertain anim-slide d3">
                    ⚠️ &nbsp; Prediction Uncertain — Silakan pilih label manual.
                </div>
                """, unsafe_allow_html=True)

                selected_label = st.selectbox("Pilih label yang benar", class_names, key="manual_label")

                if st.button("💾  Simpan Manual", key="save_manual"):
                    filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".jpg"
                    save_path = os.path.join(BASE_DIR, selected_label, filename)
                    img.save(save_path)
                    st.session_state.webcam_history.insert(0, f"{selected_label} • manual • {filename}")
                    st.success(f"Tersimpan ke {save_path}")
        else:
            st.markdown("""
            <div class="ai-card anim-slide d2" style="text-align:center;color:#B7AC90;padding:60px 20px;">
                <i class="fa-solid fa-video" style="font-size:40px;margin-bottom:12px;display:block;"></i>
                Aktifkan kamera untuk memulai deteksi.
            </div>
            """, unsafe_allow_html=True)

    # History Prediction (opsional, murni tampilan tambahan)
    if st.session_state.webcam_history:
        st.markdown('<div class="section-title anim-slide"><i class="fa-solid fa-clock-rotate-left"></i><h3>History Prediction</h3></div>', unsafe_allow_html=True)
        st.markdown('<div class="ai-card anim-slide">', unsafe_allow_html=True)
        for item in st.session_state.webcam_history[:5]:
            st.markdown(f"- {item}")
        st.markdown('</div>', unsafe_allow_html=True)