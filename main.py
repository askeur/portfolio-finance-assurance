import streamlit as st
import os
import sys
import subprocess
import time
import importlib.util

# Configuration page (doit être premier Streamlit call)
st.set_page_config(page_title="Portfolio Finance & Assurance", layout="wide")

# Authentification simple
VALID_USERNAME = "admin"
VALID_PASSWORD = "1234"

if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔐 Connexion requise")
    username = st.text_input("Nom d'utilisateur")
    password = st.text_input("Mot de passe", type="password")
    if st.button("Se connecter"):
        if username == VALID_USERNAME and password == VALID_PASSWORD:
           
            st.session_state.authenticated = True
            st.query_params["reload"] = "1"  # ⚠️ Force un reload léger via URL
            
        else:
            st.error("Nom d'utilisateur ou mot de passe incorrect.")
    st.stop()

# Chemins
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
base_path = os.path.join(BASE_DIR, "01-credit-default-prediction", "credit-default-prediction")
api_path = os.path.join(base_path, "api")
app_path = os.path.join(base_path, "app", "credit_app.py")

if not os.path.exists(app_path):
    st.error(f"❌ Fichier Streamlit introuvable : {app_path}")
    st.stop()

def launch_api_once():
    if not getattr(st.session_state, "api_launched", False):
        subprocess.Popen(
            ["uvicorn", "main:app", "--reload", "--port", "8000"],
            cwd=api_path,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        st.session_state.api_launched = True
        time.sleep(2)

if "project" not in st.session_state:
    st.session_state.project = None

# Titre dynamique
if st.session_state.project is None:
    st.title("🏦 Portfolio Finance & Assurance")
else:
    titles = {
        "credit_default": "📈 Credit Default Prediction",
        "fraud": "💳 Fraude Transactionnelle",
        "insurance": "📋 Insurance Risk",
        "churn": "📉 Client Churn"
    }
    st.title(titles.get(st.session_state.project, "📊 Portfolio Finance & Assurance"))

# Interface principale
if st.session_state.project is None:
    st.markdown("Bienvenue ! Choisissez un projet à lancer :")
    col1, col2, col3 = st.columns([1, 1, 2])
    with col3:
        st.markdown("### ⚙️ Projets disponibles")
        if st.button("📈 01 - Credit Default Prediction"):
            st.session_state.project = "credit_default"
            st.query_params["reload"] = "1"  # ⚠️ Force un reload léger via URL
        if st.button("💳 02 - Fraude Transactionnelle"):
            st.session_state.project = "fraud"
            st.query_params["reload"] = "1"  # ⚠️ Force un reload léger via URL
        if st.button("📋 03 - Insurance Risk"):
            st.session_state.project = "insurance"
            st.query_params["reload"] = "1"  # ⚠️ Force un reload léger via URL
        if st.button("📉 04 - Client Churn"):
            st.session_state.project = "churn"
            st.query_params["reload"] = "1"  # ⚠️ Force un reload léger via URL

else:
    with st.sidebar:
        st.markdown("### 📂 Projets")
        if st.button("🏠 Revenir à l’accueil"):
            st.session_state.project = None
            st.query_params["reload"] = "1"  # ⚠️ Force un reload léger via URL

    if st.session_state.project == "credit_default":
        launch_api_once()

        spec = importlib.util.spec_from_file_location("credit_app", app_path)
        credit_app = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(credit_app)
        credit_app.run_credit_app()

    else:
        st.markdown(f"🛠️ Projet **{st.session_state.project}** en cours de développement.")
