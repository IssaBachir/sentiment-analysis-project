import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from fpdf import FPDF
import os
# Configuration de la page Streamlit
st.set_page_config(
    page_title="Application de Classification  ",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)
# Définir les classes
classes = {
    0: 'Cyst',
    1: 'Normal',
    2: 'Stone',
    3: 'Tumor'
}

# Charger le modèle sauvegardé
model_path = "kidney_model.h5"
model = load_model(model_path)

# Définir la taille des images attendues
image_size = (64, 64)

# Titre de l'application avec mise en forme
st.markdown(
    """
    <style>
        .main {
            background-color: #f0f2f6;
            font-family: 'Arial', sans-serif;
        }
        .title {
            font-size: 36px;
            font-weight: bold;
            color: #4A90E2;
            text-align: center;
            margin-bottom: 10px;
        }
        .sub-title {
            font-size: 18px;
            color: #7D7D7D;
            text-align: center;
            margin-bottom: 20px;
        }
    </style>
    <div class="title">🌟 Application de Classification des Maladies Rénales 🌟</div>
    <div class="sub-title">Utilisez cette application pour classifier les maladies rénales à partir d'images CT</div>
    """,
    unsafe_allow_html=True
)

# Étape 1 : Importation de l'image
st.markdown("### 1⃣ Importer une Image CT du Rein")
uploaded_file = st.file_uploader("📂 Téléchargez une image CT du rein", type=["jpg", "jpeg", "png"], label_visibility="visible")
image = None
if uploaded_file is not None:
    image = load_img(uploaded_file, target_size=image_size)
    st.image(image, caption='🖼 Image téléchargée', use_container_width=True)

# Étape 2 : Remplissage des informations du patient
st.markdown("### 2⃣ Remplir les Informations du Patient")
with st.form("patient_form"):
    patient_name = st.text_input("✍️ Nom du Patient", "")
    patient_age = st.number_input("🎂 Âge du Patient", min_value=0, step=1)
    patient_gender = st.selectbox("🚼 Sexe", ["Homme", "Femme"])
    submit_button = st.form_submit_button("🔍 Soumettre et Prédire")

# Étape 3 : Prédiction et affichage des résultats
if submit_button:
    if not uploaded_file:
        st.warning("⚠️ Veuillez importer une image avant de soumettre.")
    elif not patient_name or patient_age == 0:
        st.warning("⚠️ Veuillez remplir toutes les informations du patient avant de prédire.")
    else:
        # Prétraiter l'image
        test_image = img_to_array(image)
        test_image = np.expand_dims(test_image, axis=0) / 255.0

        # Prédire avec le modèle
        prediction = model.predict(test_image)
        predicted_class = np.argmax(prediction, axis=1)[0]
        confidence_scores = prediction[0]

        # Afficher les résultats
        st.markdown("""<hr style='border: 1px solid #4A90E2;'>""", unsafe_allow_html=True)
        st.markdown("### 3⃣ Résultats de la Prédiction")
        predicted_disease = classes[predicted_class]
        st.markdown(f"**🧬 Maladie Prédite :** :blue[{predicted_disease}]")

        st.subheader("🔢 Scores de Confiance")
        st.progress(int(max(confidence_scores) * 100))
        for idx, score in enumerate(confidence_scores):
            st.write(f"{classes[idx]}: {score * 100:.2f}%")

        # Générer un rapport PDF
        pdf = FPDF()
        pdf.add_page()

        # Ajouter le logo
        logo_path = "issb.png"  # Chemin vers votre logo par défaut
        if os.path.exists(logo_path):
            pdf.image(logo_path, x=10, y=8, w=33)  # Ajuster la position et la taille du logo

        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Rapport de Classification des Maladies Rénales", ln=True, align='C')
        pdf.ln(20)

        pdf.cell(200, 10, txt=f"Nom du Patient : {patient_name}", ln=True)
        pdf.cell(200, 10, txt=f"Âge du Patient : {patient_age}", ln=True)
        pdf.cell(200, 10, txt=f"Sexe du Patient : {patient_gender}", ln=True)
        pdf.cell(200, 10, txt=f"Maladie Prédite : {predicted_disease}", ln=True)
        pdf.ln(10)
        pdf.cell(200, 10, txt="Scores de Confiance :", ln=True)

        for idx, score in enumerate(confidence_scores):
            pdf.cell(200, 10, txt=f"{classes[idx]}: {score * 100:.2f}%", ln=True)

        # Sauvegarder le fichier PDF
        pdf_path = "rapport_patient.pdf"
        pdf.output(pdf_path)

        # Ajouter un bouton de téléchargement
        st.markdown("""<hr style='border: 1px solid #4A90E2;'>""", unsafe_allow_html=True)
        st.markdown("### 📥 Télécharger le Rapport")
        with open(pdf_path, "rb") as pdf_file:
            st.download_button(
                label="📄 Télécharger le Rapport en PDF",
                data=pdf_file,
                file_name=f"{patient_name}_rapport.pdf",
                mime="application/pdf"
            )
