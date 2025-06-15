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
#  MODEL DOWNLOADER      #
# ====================== #
def download_models():
    """Download models from Google Drive if not exists"""
    os.makedirs("models", exist_ok=True)
    os.makedirs("weights", exist_ok=True)
    
    models = {
        "models/yolov8n-face.pt": "https://drive.google.com/file/d/1wgm1JlhtBJl_WlOPCQdovn4CctImpCsG/view?usp=sharing",
        "weights/RealESRGAN_x4plus.pth": "https://drive.google.com/file/d/1d-wlgMp2NZPnLfSkcCgRNqN_yd83Lew8/view?usp=sharing", 
        "weights/GFPGANv1.3.pth": "https://drive.google.com/file/d/1H3m1BquK3wWb2XM9EPa4i_w7AsOiH546/view?usp=sharing"
    }

    for path, url in models.items():
        if not os.path.exists(path):
            try:
                gdown.download(url, path, quiet=False)
                st.success(f"Downloaded: {os.path.basename(path)}")
            except Exception as e:
                st.error(f"Failed to download {path}: {str(e)}")
                raise e

# ====================== #
#  INITIALIZE MODELS     #
# ====================== #
@st.cache_resource
def load_models():
    """Load all models with caching"""
    download_models()  # Pastikan model sudah ada
    
    # Load Face Detection Model
    face_model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",
        model_path="models/yolov8n-face.pt",
        confidence_threshold=0.4,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load Upscaler
    esrgan_model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4,
        model_path="weights/RealESRGAN_x4plus.pth",
        model=esrgan_model,
        tile=400,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    # Load Face Enhancer
    face_enhancer = GFPGANer(
        model_path="weights/GFPGANv1.3.pth",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    return face_model, upsampler, face_enhancer

# ====================== #
#  STREAMLIT UI          #
# ====================== #
st.title("🚀 Face Detection & Image Upscaling")
st.write("Upscale gambar lalu deteksi wajah dengan YOLOv8 + SAHI")

uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "png", "jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_array = np.array(image)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption="Gambar Asli", use_column_width=True)

    if st.button("Proses"):
        with st.spinner("Loading models..."):
            face_model, upsampler, face_enhancer = load_models()

        with st.spinner("Upscaling image..."):
            # Upscale dengan ESRGAN
            upscaled, _ = upsampler.enhance(img_array, outscale=2)
            
            # Face Enhancement dengan GFPGAN
            _, _, upscaled = face_enhancer.enhance(
                upscaled,
                has_aligned=False,
                only_center_face=False,
                paste_back=True
            )

        with st.spinner("Detecting faces..."):
            # Deteksi wajah dengan SAHI
            result = get_sliced_prediction(
                upscaled,
                face_model,
                slice_height=640,
                slice_width=640,
                overlap_height_ratio=0.3
            )

            # Visualisasi hasil
            if len(result.object_prediction_list) > 0:
                vis_image = visualize_object_predictions(
                    upscaled,
                    result.object_prediction_list,
                    output_dir=None
                )
                result_img = vis_image["image"]
            else:
                result_img = upscaled
                st.warning("Tidak ada wajah yang terdeteksi!")

        with col2:
            st.image(result_img, caption="Hasil", use_column_width=True)

            # Download button
            buf = BytesIO()
            result_pil = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
            result_pil.save(buf, format="JPEG")
            st.download_button(
                "💾 Download Hasil",
                buf.getvalue(),
                file_name=f"hasil_{uploaded_file.name}",
                mime="image/jpeg"
            )

# Tampilkan info sistem
st.sidebar.markdown("### ℹ️ System Info")
st.sidebar.write(f"Device: {'GPU 🚀' if torch.cuda.is_available() else 'CPU 🐢'}")