import streamlit as st
import numpy as np
import os
import cv2
import io
from PIL import Image
#from models.real_esrgan import RealESRGAN

st.set_page_config(page_title="Restauration d'image IA", layout="centered")

st.title("🧠✨ RevivIA")

# Menu de sélection du modèle
model_type = st.selectbox(
    "Type de modèle",
    ("Real-ESRGAN", "SRGAN", "SwinIR")
)

if model_type=="Real-ESRGAN":
    model_name = st.selectbox(
        "Nom du modèle",
        (
            "RealESRGAN_x4plus",
            "RealESRNet_x4plus",
            "RealESRGAN_x4plus_anime_6B",
            "RealESRGAN_x2plus",
            "realesr-animevideov3",
            "realesr-general-x4v3",
        )
    )

st.markdown("Téléversez votre image ")
uploaded_file = st.file_uploader("📤 Choisissez une image à restaurer :", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
    img_name, extension = os.path.splitext(os.path.basename(uploaded_file.name))
    input_image = Image.open(uploaded_file)
    st.image(input_image, caption="Image originale", use_container_width=True)

    if st.button("Restaurer l’image"):
        with st.spinner("Restauration en cours..."):

            if model_type=="Real-ESRGAN":
                model = RealESRGAN()
                image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
                output_image = model.inference(input_image=image, model_name=model_name)
                st.success("Image restaurée avec succès ✅")
                st.image(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB), caption="Image restaurée", use_container_width=True)

                is_success, buffer = cv2.imencode(extension, output_image)
                if is_success:
                    st.download_button(
                        label="📥 Télécharger l'image restaurée",
                        data=io.BytesIO(buffer),
                        file_name=f"{img_name}_upscaled{extension}",
                        mime="image/png"
                    )
            

            elif model_type=="SRGAN":

                file_bytes = np.frombuffer(uploaded_file.read(), np.uint8)
                img_name, extension = os.path.splitext(os.path.basename(uploaded_file.name))
                input_image = Image.open(uploaded_file)
                st.image(input_image, caption="Image originale", use_container_width=True)
                print(model_type)

            elif model_type=="SwinIR":
                print(model_type) 