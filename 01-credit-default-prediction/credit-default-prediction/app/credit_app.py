import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests
import json
import os
import subprocess
import joblib
import shap
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc


def run_credit_app():
    # 🔧 Sidebar
    st.sidebar.header("🧭 Navigation")

    menu = st.sidebar.radio("Aller vers :", [
        "📋 Présentation",
        "📈 EDA",
        "🔁 Entraînement",
        "📡 Analyse API",
        "🔍 SHAP",
        "🧪 Prédire un client"
    ])

    # 📦 Choix du modèle
    models = {
        "XGBoost": "xgboost",
        "Random Forest": "random_forest",
        "Logistic Regression": "logistic_regression",
        "LightGBM": "lightgbm",
        "CatBoost": "catboost"
    }
    selected_label = st.sidebar.selectbox("🧠 Modèle", list(models.keys()))
    selected_model = models[selected_label]

    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    base_path = os.path.join(BASE_DIR, "01-credit-default-prediction", "credit-default-prediction")  
    
    json_path = os.path.join(base_path, "data", "processed", f"{selected_model}_eval.json") 
    model_path = os.path.join(base_path, "models", f"{selected_model}.pkl")
    fig_path = os.path.join(base_path, "reports", "figures")
    model_training_script = os.path.join(base_path, "src", "model_training.py")

    if os.path.exists(json_path):
        with open(json_path) as f:
            st.sidebar.download_button("📄 Télécharger rapport", f, file_name="rapport_modele.json")

    # 📥 Modèle + données
    @st.cache_resource
    def load_model(path):
        return joblib.load(path)

    @st.cache_data
    def load_data():
        
        try:
            train_path = os.path.join(base_path, "data", "processed", "train_clean.csv")
            df = pd.read_csv(train_path, encoding="utf-8")
        except UnicodeDecodeError:
            # Repli automatique sur l'encodage ISO-8859-1 (souvent utilisé pour les fichiers Windows)
            df = pd.read_csv(train_path, encoding="ISO-8859-1")

        X = df.drop(columns=["SeriousDlqin2yrs"])
        y = df["SeriousDlqin2yrs"]
        return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


    if menu == "📋 Présentation":
        st.markdown("""
        Cette application vous permet d'entraîner et d'analyser des modèles de scoring crédit.
        
        **Fonctionnalités** :
        - Entraînement automatique de modèles
        - Analyse via API
        - Visualisations EDA
        - Interprétabilité SHAP
        """)

    # 📈 Exploration des données
    elif menu == "📈 EDA":
        st.subheader("📈 Exploration des données")
       
        if os.path.exists(f"{fig_path}/target_distribution.png"):
            st.image(f"{fig_path}/target_distribution.png", caption="Distribution de la cible")
            st.image(f"{fig_path}/correlation_matrix.png", caption="Matrice de corrélation")
        else:
            st.info("📌 Lancez le script `src/visualize_data.py` pour générer les figures.")

    # 🔁 Entraînement
    elif menu == "🔁 Entraînement":
        st.subheader("🚀 Entraînement du modèle")
        if os.path.exists(model_path):
            st.success("✅ Modèle déjà entraîné.")
            if st.button("🔁 Réentraîner"):
                subprocess.run(["python", model_training_script])
                st.success("Modèle réentraîné.")
                st.rerun()
        else:
            if st.button("🚀 Lancer l'entraînement"):
                subprocess.run(["python", model_training_script])
                st.success("Entraînement terminé.")
                st.rerun()

    # 📡 Analyse API
    elif menu == "📡 Analyse API":
        st.subheader("📡 Analyse via API")
        print(json_path)
        if os.path.exists(json_path):
            with open(json_path) as f:
                eval_data = json.load(f)
            st.json(eval_data)
            if st.button("📨 Envoyer à l'API"):
                try:
                    response = requests.post("http://127.0.0.1:8000/analyze", json=eval_data)
                    if response.status_code == 200:
                        st.success("Analyse API :")
                        for line in response.json().get("analysis", []):
                            st.write("🔹", line)
                    else:
                        st.error(f"Erreur API : {response.status_code}")
                except Exception as e:
                    st.warning(f"API non disponible : {e}")
        else:
            st.info("⚠️ Entraînez un modèle d'abord.")

    # 🔍 Interprétation SHAP
    elif menu == "🔍 SHAP":
        st.subheader("🔍 Interprétabilité SHAP")
        
        if os.path.exists(model_path):
            model = load_model(model_path)
            X_train, X_val, _, _ = load_data()

            try:
                sample_X = X_val.sample(min(100, len(X_val)), random_state=42)

                # Utilisation du TreeExplainer pour les modèles d'arbres
                explainer = shap.TreeExplainer(model)
                shap_values = explainer.shap_values(sample_X)

                st.markdown("#### 📊 Importance globale")
                shap.summary_plot(shap_values, sample_X, plot_type="bar", show=False)
                st.pyplot(plt.gcf())

                st.markdown("#### 🧬 Explication locale (1er individu)")
                shap.plots._waterfall.waterfall_legacy(
                    explainer.expected_value, shap_values[0], feature_names=sample_X.columns, show=False
                )
                st.pyplot(plt.gcf())

            except Exception as e:
                st.error(f"Erreur SHAP : {e}")

    # 🧪 Prédire un client
    elif menu == "🧪 Prédire un client":
        st.subheader("🧪 Tester un nouveau client")
        uploaded = st.file_uploader("Charger un fichier CSV (1 ligne)", type="csv")
        if uploaded and os.path.exists(model_path):
            df_input = pd.read_csv(uploaded)
            model = load_model(model_path)
            try:
                prediction = model.predict_proba(df_input)[:, 1]
                st.success(f"📌 Probabilité de défaut : {prediction[0]:.2f}")
            except Exception as e:
                st.error(f"Erreur de prédiction : {e}")
        else:
            st.info("📂 Veuillez charger un fichier CSV et avoir un modèle entraîné.")
