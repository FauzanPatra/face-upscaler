import os
import sys
import subprocess
import pkg_resources
import gdown
import numpy as np
from io import BytesIO
from PIL import Image
import streamlit as st
import cv2
import torch
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from sahi.utils.cv import visualize_object_predictions
from basicsr.archs.rrdbnet_arch import RRDBNet
from ultralytics import YOLO
from realesrgan import RealESRGANer
from gfpgan import GFPGANer

# ====================== #
#  FUNGSI BAGIAN ESRGAN  #
# ====================== #
class ESRGAN_Enhancer:
    def __init__(self):
        self.upsampler = None
        self.face_enhancer = None
        
    def load_models(self):
        """Memuat model ESRGAN dan GFPGAN"""
        os.makedirs("weights", exist_ok=True)
        
        model_dict = {
            "weights/RealESRGAN_x4plus.pth": "https://drive.google.com/uc?id=1d-wlgMp2NZPnLfSkcCgRNqN_yd83Lew8",
            "weights/GFPGANv1.3.pth": "https://drive.google.com/uc?id=1H3m1BquK3wWb2XM9EPa4i_w7AsOiH546"
        }

        for path, url in model_dict.items():
            if not os.path.exists(path):
                with st.spinner(f"Mengunduh {os.path.basename(path)}..."):
                    gdown.download(url, path, quiet=True)

        model = RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=64, 
            num_block=23, num_grow_ch=32, scale=4
        )
        
        self.upsampler = RealESRGANer(
            scale=4,
            model_path="weights/RealESRGAN_x4plus.pth",
            model=model,
            tile=400,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        
        self.face_enhancer = GFPGANer(
            model_path="weights/GFPGANv1.3.pth",
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
    
    def enhance_image(self, image, upscale_factor=2, enhance_face=True):
        """Meningkatkan resolusi gambar"""
        try:
            img_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            upscaled_img, _ = self.upsampler.enhance(img_bgr, outscale=upscale_factor)
            
            if enhance_face:
                _, _, upscaled_img = self.face_enhancer.enhance(
                    upscaled_img,
                    has_aligned=False,
                    only_center_face=False,
                    paste_back=True
                )
            
            return cv2.cvtColor(upscaled_img, cv2.COLOR_BGR2RGB)
            
        except Exception as e:
            st.error(f"Gagal meningkatkan resolusi: {str(e)}")
            return None

# ====================== #
#  FUNGSI FACE DETECTION #
# ====================== #
class FaceDetector:
    def __init__(self):
        self.model = None
        
    def load_model(self):
        """Memuat model deteksi wajah"""
        os.makedirs("models", exist_ok=True)
        
        model_path = "models/yolov8n-face.pt"
        if not os.path.exists(model_path):
            with st.spinner("Mengunduh model deteksi wajah..."):
                gdown.download(
                    "https://drive.google.com/uc?id=1wgm1JlhtBJl_WlOPCQdovn4CctImpCsG",
                    model_path,
                    quiet=True
                )

        # Gunakan YOLO langsung dari Ultralytics
        self.model = YOLO(model_path)
    
    def detect_faces(self, image, slice_size=640, overlap_ratio=0.3):
        """Mendeteksi wajah dalam gambar"""
        try:
            img_array = np.array(image)
            img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            
            # Deteksi dengan YOLOv8
            results = self.model.predict(
                img_bgr,
                conf=0.4,
                imgsz=slice_size
            )
            
            # Visualisasi hasil
            if len(results[0]) > 0:
                visualized_image = results[0].plot()
                visualized_image = cv2.cvtColor(visualized_image, cv2.COLOR_BGR2RGB)
                return visualized_image, len(results[0])
            else:
                return image, 0
                
        except Exception as e:
            st.error(f"Gagal mendeteksi wajah: {str(e)}")
            return image, 0

# ====================== #
#  ANTARMUKA UTAMA      #
# ====================== #
def main():
    st.set_page_config(
        page_title="✨ Face Enhancer Pro",
        page_icon=":camera_flash:",
        layout="wide"
    )
    
    st.title("✨ Face Enhancer Pro")
    
    with st.sidebar:
        st.header("⚙️ Pengaturan")
        tab1, tab2 = st.tabs(["ESRGAN", "Face Detection"])
        
        with tab1:
            upscale_factor = st.slider("Faktor Upscale", 2, 4, 2)
            enhance_faces = st.toggle("Perbaikan Wajah", True)
            
        with tab2:
            detection_thresh = st.slider("Threshold Deteksi", 0.1, 0.9, 0.4, 0.05)
            slice_size = st.slider("Ukuran Slice", 320, 800, 640, 32)
        
        st.divider()
        st.markdown("**ℹ️ Informasi Sistem**")
        st.write(f"Device: {'GPU 🚀' if torch.cuda.is_available() else 'CPU 🐢'}")
    
    uploaded_file = st.file_uploader(
        "Unggah gambar (JPG/PNG)", 
        type=["jpg", "png", "jpeg"]
    )
    
    if uploaded_file:
        original_image = Image.open(uploaded_file).convert("RGB")
        col1, col2 = st.columns(2)
        
        with col1:
            st.image(original_image, caption="Gambar Asli", use_column_width=True)
        
        if st.button("🚀 Proses Sekarang", type="primary"):
            with st.spinner("Memuat model..."):
                esrgan = ESRGAN_Enhancer()
                face_detector = FaceDetector()
                esrgan.load_models()
                face_detector.load_model()
            
            with st.spinner("Meningkatkan resolusi..."):
                enhanced_img = esrgan.enhance_image(
                    original_image, 
                    upscale_factor=upscale_factor,
                    enhance_face=enhance_faces
                )
            
            if enhanced_img is not None:
                with st.spinner("Mendeteksi wajah..."):
                    final_img, face_count = face_detector.detect_faces(
                        enhanced_img,
                        slice_size=slice_size
                    )
                
                with col2:
                    st.image(
                        final_img, 
                        caption=f"Hasil ({face_count} wajah terdeteksi)", 
                        use_column_width=True
                    )
                    
                    buf = BytesIO()
                    Image.fromarray(final_img).save(buf, format="JPEG", quality=95)
                    st.download_button(
                        "💾 Download Hasil",
                        buf.getvalue(),
                        file_name=f"enhanced_{uploaded_file.name}",
                        mime="image/jpeg"
                    )

if __name__ == "__main__":
    main()