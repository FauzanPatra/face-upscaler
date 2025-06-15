import os
import gdown
import streamlit as st
import cv2
import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import visualize_object_predictions
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from gfpgan import GFPGANer
import numpy as np
from PIL import Image
from io import BytesIO

# ====================== #
#  KONFIGURASI AWAL      #
# ====================== #
# Cek dan install dependensi yang diperlukan
try:
    import cv2
    import torch
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    from sahi.utils.cv import visualize_object_predictions
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from realesrgan import RealESRGANer
    from gfpgan import GFPGANer
except ImportError as e:
    st.error(f"Error: {e}")
    st.info("Memasang dependensi yang diperlukan...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
    st.experimental_rerun()

# ====================== #
#  PENGATURAN APLIKASI   #
# ====================== #
st.set_page_config(
    page_title="✨ Face Enhancer Pro",
    page_icon=":camera_flash:",
    layout="wide"
)

# ====================== #
#  FUNGSI UTAMA         #
# ====================== #
def unduh_model():
    """Mengunduh model jika belum ada"""
    os.makedirs("models", exist_ok=True)
    os.makedirs("weights", exist_ok=True)
    
    model_dict = {
        "models/yolov8n-face.pt": "https://drive.google.com/uc?id=1wgm1JlhtBJl_WlOPCQdovn4CctImpCsG",
        "weights/RealESRGAN_x4plus.pth": "https://drive.google.com/uc?id=1d-wlgMp2NZPnLfSkcCgRNqN_yd83Lew8", 
        "weights/GFPGANv1.3.pth": "https://drive.google.com/uc?id=1H3m1BquK3wWb2XM9EPa4i_w7AsOiH546"
    }

    for path, url in model_dict.items():
        if not os.path.exists(path):
            with st.spinner(f"Mengunduh {os.path.basename(path)}..."):
                try:
                    gdown.download(url, path, quiet=True)
                except Exception as e:
                    st.error(f"Gagal mengunduh {path}: {str(e)}")
                    raise e

@st.cache_resource
def muat_model():
    """Memuat semua model dengan caching"""
    unduh_model()  # Pastikan model sudah ada
    
    # Detektor Wajah
    model_wajah = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path="models/yolov8n-face.pt",
        confidence_threshold=0.4,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Peningkat Resolusi
    model_upscale = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64, 
        num_block=23, num_grow_ch=32, scale=4
    )
    upsampler = RealESRGANer(
        scale=4,
        model_path="weights/RealESRGAN_x4plus.pth",
        model=model_upscale,
        tile=400,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Peningkat Kualitas Wajah
    enhancer_wajah = GFPGANer(
        model_path="weights/GFPGANv1.3.pth",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    return model_wajah, upsampler, enhancer_wajah

# ====================== #
#  TAMPILAN STREAMLIT    #
# ====================== #
st.title("✨ Face Enhancer Pro")
st.markdown("""
Meningkatkan kualitas gambar dengan:
- **Deteksi Wajah**: YOLOv8 + SAHI
- **Peningkatan Resolusi**: RealESRGAN
- **Perbaikan Wajah**: GFPGAN
""")

with st.sidebar:
    st.header("⚙️ Pengaturan")
    threshold_kepercayaan = st.slider("Threshold Deteksi", 0.1, 0.9, 0.4, 0.05)
    faktor_upscale = st.slider("Faktor Upscale", 2, 4, 2)
    aktifkan_perbaikan = st.toggle("Aktifkan Perbaikan Wajah", True)
    
    st.divider()
    st.markdown("**ℹ️ Informasi Sistem**")
    st.write(f"Device: {'GPU 🚀' if torch.cuda.is_available() else 'CPU 🐢'}")

# ====================== #
#  PROSES UTAMA         #
# ====================== #
file_upload = st.file_uploader(
    "Unggah gambar (JPG/PNG)", 
    type=["jpg", "png", "jpeg"],
    help="Maksimal 5MB"
)

if file_upload:
    # Baca gambar
    gambar_asli = Image.open(file_upload).convert("RGB")
    array_gambar = np.array(gambar_asli)
    
    # Tampilkan gambar asli
    col1, col2 = st.columns(2)
    with col1:
        st.image(gambar_asli, caption="Gambar Asli", use_column_width=True)

    # Tombol proses
    if st.button("🚀 Proses Sekarang", type="primary"):
        with st.spinner("Memuat model..."):
            model_wajah, upsampler, enhancer_wajah = muat_model()

        with st.spinner("Meningkatkan resolusi..."):
            try:
                # Upscale gambar
                gambar_upscale, _ = upsampler.enhance(
                    array_gambar, 
                    outscale=faktor_upscale
                )
                
                # Perbaikan wajah (opsional)
                if aktifkan_perbaikan:
                    _, _, gambar_upscale = enhancer_wajah.enhance(
                        gambar_upscale,
                        has_aligned=False,
                        only_center_face=False,
                        paste_back=True
                    )

                # Deteksi wajah
                hasil_deteksi = get_sliced_prediction(
                    gambar_upscale,
                    model_wajah,
                    slice_height=640,
                    slice_width=640,
                    overlap_height_ratio=0.3
                )

                # Visualisasi hasil
                if hasil_deteksi.object_prediction_list:
                    gambar_hasil = visualize_object_predictions(
                        gambar_upscale,
                        hasil_deteksi.object_prediction_list,
                        output_dir=None
                    )["image"]
                    jumlah_wajah = len(hasil_deteksi.object_prediction_list)
                else:
                    gambar_hasil = gambar_upscale
                    jumlah_wajah = 0
                    st.warning("Tidak ada wajah yang terdeteksi!")

                # Tampilkan hasil
                with col2:
                    st.image(
                        gambar_hasil, 
                        caption=f"Hasil ({jumlah_wajah} wajah terdeteksi)", 
                        use_column_width=True
                    )
                    
                    # Tombol download
                    buffer = BytesIO()
                    Image.fromarray(cv2.cvtColor(gambar_hasil, cv2.COLOR_RGB2BGR)).save(
                        buffer, format="JPEG", quality=95
                    )
                    st.download_button(
                        "💾 Download Hasil",
                        buffer.getvalue(),
                        file_name=f"hasil_{file_upload.name}",
                        mime="image/jpeg"
                    )

                    # Informasi teknis
                    with st.expander("🔍 Detail Teknis"):
                        st.metric("Ukuran Asli", f"{gambar_asli.width}×{gambar_asli.height} piksel")
                        st.metric("Ukuran Hasil", f"{gambar_hasil.shape[1]}×{gambar_hasil.shape[0]} piksel")
                        st.metric("Model", "YOLOv8n + RealESRGAN_x4plus + GFPGANv1.3")

            except Exception as e:
                st.error(f"Terjadi kesalahan: {str(e)}")
                st.stop()

# Catatan kaki
st.divider()
st.caption("© 2023 Face Enhancer Pro | Dibuat dengan Streamlit")