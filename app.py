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
#  Konfigurasi Aplikasi  #
# ====================== #
st.set_page_config(
    page_title="🔍 Face Upscaler Pro",
    page_icon="✨",
    layout="wide"
)

# ====================== #
#  Sidebar Kontrol       #
# ====================== #
with st.sidebar:
    st.header("Pengaturan")
    confidence_threshold = st.slider("Confidence Threshold", 0.1, 0.9, 0.4, 0.05)
    upscale_factor = st.slider("Upscale Factor", 2, 4, 2)
    enable_face_enhance = st.toggle("Face Enhancement", True)

# ====================== #
#  Fungsi Utama          #
# ====================== #
def main():
    st.title("🔍 Face Upscaler Pro")
    st.markdown("""
    Upscale gambar + deteksi wajah dengan:
    - **YOLOv8** (Deteksi) + **SAHI** (Slice Inference)
    - **RealESRGAN** (Upscale)
    - **GFPGAN** (Perbaikan Wajah)
    """)

    uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        with st.spinner("Memproses gambar..."):
            try:
                # Load image
                image = Image.open(uploaded_file).convert("RGB")
                img_array = np.array(image)

                # Display original
                col1, col2 = st.columns(2)
                with col1:
                    st.image(image, caption="Original", use_column_width=True)

                # Proses hanya ketika tombol ditekan
                if st.button("✨ Enhance Image"):
                    # Load models
                    face_model, upsampler, face_enhancer = load_models(confidence_threshold)

                    # Upscale
                    upscaled, _ = upsampler.enhance(img_array, outscale=upscale_factor)

                    # Face Enhancement (opsional)
                    if enable_face_enhance:
                        _, _, upscaled = face_enhancer.enhance(
                            upscaled,
                            has_aligned=False,
                            only_center_face=False,
                            paste_back=True
                        )

                    # Face Detection
                    result = get_sliced_prediction(
                        upscaled,
                        face_model,
                        slice_height=640,
                        slice_width=640,
                        overlap_height_ratio=0.3
                    )

                    # Visualisasi hasil
                    if result.object_prediction_list:
                        vis_image = visualize_object_predictions(
                            upscaled,
                            result.object_prediction_list,
                            output_dir=None
                        )
                        result_img = vis_image["image"]
                        detected_faces = len(result.object_prediction_list)
                    else:
                        result_img = upscaled
                        detected_faces = 0

                    # Tampilkan hasil
                    with col2:
                        st.image(result_img, caption=f"Hasil (Detected: {detected_faces} faces)", use_column_width=True)
                        
                        # Download button
                        buf = BytesIO()
                        result_pil = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
                        result_pil.save(buf, format="JPEG", quality=95)
                        st.download_button(
                            "⬇️ Download Hasil",
                            buf.getvalue(),
                            file_name=f"enhanced_{uploaded_file.name}",
                            mime="image/jpeg"
                        )

                        # Metrics
                        st.metric("Original Size", f"{image.width}x{image.height}")
                        st.metric("Upscaled Size", f"{result_img.shape[1]}x{result_img.shape[0]}")

            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.stop()

# Jalankan aplikasi
if __name__ == "__main__":
    main()