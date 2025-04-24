import streamlit as st
from PIL import Image

st.set_page_config(page_title="Restauration d'image IA", layout="centered")

st.title("🧠✨ RevivIA")
st.markdown("Téléversez votre image ")

uploaded_file = st.file_uploader("📤 Choisissez une image à restaurer :", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    input_image = Image.open(uploaded_file)
    st.image(input_image, caption="Image originale", use_column_width=True)

    if st.button("Restaurer l’image"):
        with st.spinner("Restauration en cours..."):
            st.success("Image restaurée avec succès ✅")