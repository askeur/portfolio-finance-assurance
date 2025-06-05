import streamlit as st
import os
import sys
import subprocess
import time
import importlib.util


# Configuration de la page
st.set_page_config(
    page_title="🏥 Portfolio Finance & Assurance",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)



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
credit_api_path = os.path.join(base_path, "api")
credit_app_path = os.path.join(base_path, "app", "credit_app.py")

base_path = os.path.join(BASE_DIR, "02-insurance-claims-analysis", "insurance-claims-analysis")
insurance_api_path = os.path.join(base_path, "api")
insurance_app_path = os.path.join(base_path, "app", "insurance_app.py")

if not os.path.exists(credit_app_path):
    st.error(f"❌ Fichier Streamlit introuvable : {credit_app_path}")
    st.stop()

def run_credit_api():
    if not getattr(st.session_state, "api_launched", False):
        subprocess.Popen(
            ["uvicorn", "main:app", "--reload", "--port", "8000"],
            cwd=credit_api_path,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        st.session_state.api_launched = True
        time.sleep(2)

def run_insurance_api():
    if not getattr(st.session_state, "insurance_api_launched", False):
        subprocess.Popen(
            ["python", "api/insurance_api.py"],  # ⚠️ ajuste le chemin si nécessaire
            cwd=base_path,
            creationflags=subprocess.CREATE_NO_WINDOW
        )
        st.session_state.insurance_api_launched = True
        time.sleep(3)  # Laisse le temps à l’API de démarrer

        
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
            st.query_params["reload"] = "1"  

    if st.session_state.project == "credit_default":
        run_credit_api()

        spec = importlib.util.spec_from_file_location("credit_app", credit_app_path)
        credit_app = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(credit_app)
        credit_app.run_credit_app()
    
    elif st.session_state.project == "insurance":
        if not os.path.exists(insurance_app_path):
            st.error(f"❌ Fichier Streamlit introuvable : {insurance_app_path}")
            st.stop()
        run_insurance_api()
        spec = importlib.util.spec_from_file_location("insurance_app", insurance_app_path)
        insurance_app = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(insurance_app)
        insurance_app.run_insurance_app()  
    else:
        st.markdown(f"🛠️ Le projet **{st.session_state.project}** est en cours de déploiement.")