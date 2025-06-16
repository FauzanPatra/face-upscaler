import os
import sys
import subprocess
import pkg_resources
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
#  VERIFIKASI DEPENDENSI #
# ====================== #
def verifikasi_dependensi():
    """Memastikan semua dependensi terinstall dengan benar"""
    required = {
        'opencv-python': '4.8.0.76',
        'streamlit': '1.28.0',
        'torch': '2.0.1',
        'sahi': '0.11.4',
        'realesrgan': '0.3.0',
        'gfpgan': '1.3.8',
        'numpy': '1.24.3',
        'Pillow': '10.0.0',
        'basicsr': '1.4.2',
        'gdown': '4.7.1'
    }
    
    # Coba import OpenCV terlebih dahulu karena paling kritis
    try:
        import cv2
    except ImportError:
        print("OpenCV tidak ditemukan, melakukan instalasi...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python==4.8.0.76"])
        import cv2
    
    # Verifikasi sisanya
    installed = {pkg.key: pkg.version for pkg in pkg_resources.working_set}
    missing = [pkg for pkg, ver in required.items() if pkg not in installed]
    
    if missing:
        print(f"Memasang dependensi yang kurang: {', '.join(missing)}")
        subprocess.check_call([sys.executable, "-m", "pip", "install"] + [f"{pkg}=={ver}" for pkg, ver in required.items()])
        print("Instalasi selesai. Silakan restart aplikasi.")
        sys.exit(1)

# Verifikasi sebelum import streamlit
verifikasi_dependensi()

# ====================== #
#  IMPORT LIBRARY UTAMA  #
# ====================== #
import streamlit as st
import cv2
import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import visualize_object_predictions
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer
from gfpgan import GFPGANer

# ====================== #
#  KONFIGURASI APLIKASI  #
# ====================== #
st.set_page_config(
    page_title="✨ Face Enhancer Pro",
    page_icon=":camera_flash:",
    layout="wide",
    initial_sidebar_state="expanded"
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
                    st.stop()

@st.cache_resource(ttl=3600)
def muat_model():
    """Memuat semua model dengan caching"""
    unduh_model()
    
    try:
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
        
    except Exception as e:
        st.error(f"Gagal memuat model: {str(e)}")
        st.stop()

# ====================== #
#  TAMPILAN UTAMA       #
# ====================== #
def main():
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
        st.write(f"PyTorch: {torch.__version__}")
        st.write(f"OpenCV: {cv2.__version__}")

    # Proses Upload Gambar
    file_upload = st.file_uploader(
        "Unggah gambar (JPG/PNG)", 
        type=["jpg", "png", "jpeg"],
        help="Maksimal 5MB"
    )

    if file_upload:
        try:
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
                        # Konversi ke BGR untuk RealESRGAN
                        array_gambar_bgr = cv2.cvtColor(array_gambar, cv2.COLOR_RGB2BGR)
                        
                        # Upscale gambar
                        gambar_upscale, _ = upsampler.enhance(
                            array_gambar_bgr, 
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

                        # Konversi kembali ke RGB untuk visualisasi
                        gambar_upscale_rgb = cv2.cvtColor(gambar_upscale, cv2.COLOR_BGR2RGB)

                        # Deteksi wajah
                        hasil_deteksi = get_sliced_prediction(
                            gambar_upscale_rgb,
                            model_wajah,
                            slice_height=640,
                            slice_width=640,
                            overlap_height_ratio=0.3
                        )

                        # Visualisasi hasil
                        if hasil_deteksi.object_prediction_list:
                            gambar_hasil = visualize_object_predictions(
                                gambar_upscale_rgb,
                                hasil_deteksi.object_prediction_list,
                                output_dir=None
                            )["image"]
                            jumlah_wajah = len(hasil_deteksi.object_prediction_list)
                        else:
                            gambar_hasil = gambar_upscale_rgb
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
                            Image.fromarray(gambar_hasil).save(
                                buffer, format="JPEG", quality=95
                            )
                            st.download_button(
                                "💾 Download Hasil",
                                buffer.getvalue(),
                                file_name=f"hasil_{file_upload.name}",
                                mime="image/jpeg"
                            )

                    except Exception as e:
                        st.error(f"Terjadi kesalahan saat memproses gambar: {str(e)}")
                        st.stop()
                        
        except Exception as e:
            st.error(f"Gagal memproses file: {str(e)}")

    # Catatan kaki
    st.divider()
    st.caption("© 2025 Face Enhancer Pro | Dibuat dengan Streamlit")

if __name__ == "__main__":
    main()