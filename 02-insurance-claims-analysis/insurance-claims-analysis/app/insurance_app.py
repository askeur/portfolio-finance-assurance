#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Application Streamlit pour l'Analyse d'Assurance
Interface utilisateur pour le preprocessing, modélisation et visualisation
"""

import streamlit as st
import pandas as pd
import numpy as np
import requests
import json
import os
import sys
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import time
from datetime import datetime


# Configuration de l'API
API_BASE_URL = "http://localhost:5000"
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
base_path = os.path.join(BASE_DIR, "02-insurance-claims-analysis", "insurance-claims-analysis")


# CSS personnalisé
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .success-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
    }
    .error-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
    }
    .info-box {
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        color: #0c5460;
    }
</style>
""", unsafe_allow_html=True)


def check_api_status():
    """Vérifier l'état de l'API"""
    try:
        response = requests.get(f"{API_BASE_URL}/status", timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except:
        return False, None

def display_header():
    # Vérifier l'état de l'API
    api_status, status_data = check_api_status()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if api_status:
            st.success("✅ API Connectée")
        else:
            st.error("❌ API Déconnectée")
    
    with col2:
        st.info(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    
    with col3:
        if status_data:
            models_count = status_data.get('components', {}).get('models_count', 0)
            st.info(f"🤖 {models_count} Modèles")

def sidebar_navigation():
    """Barre latérale de navigation"""
    st.sidebar.title("🧭 Navigation")
    
    pages = {
        "📋 Présentation": "home",
        "📊 État du Système": "status", 
        "📤 Upload de Données": "upload",
        "🔧 Preprocessing": "preprocess",
        "🤖 Entraînement": "training",
        "🔮 Prédictions": "predictions",
        "📈 Visualisations": "visualizations",
        "📋 Modèles": "models"
    }
    
    selected = st.sidebar.selectbox("Choisir une page", list(pages.keys()))
    return pages[selected]

def home_page():
    """Page d'accueil"""
    st.title("🏥 Analyse d'Assurance Santé")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📋 À propos")
        st.write("""
        Cette plateforme permet d'analyser les données d'assurance santé avec :
        - **Preprocessing** automatique des données
        - **Modélisation** ML (classification et clustering)
        - **Visualisations** interactives
        - **Prédictions** en temps réel
        """)
        
        st.subheader("🚀 Pour commencer")
        st.write("""
        1. **Upload** vos fichiers CSV (bénéficiaires + réclamations)
        2. **Preprocess** les données pour les nettoyer
        3. **Entraîner** les modèles ML
        4. **Visualiser** les résultats et faire des prédictions
        """)
    
    with col2:
        st.subheader("📊 Statistiques Rapides")
        
        # Récupérer les stats de l'API
        api_status, status_data = check_api_status()
        
        if status_data:
            components = status_data.get('components', {})
            
            # Métriques
            col2_1, col2_2 = st.columns(2)
            with col2_1:
                st.metric("Données Preprocessées", 
                         "✅" if components.get('processed_data_available') else "❌")
                st.metric("Modèles Disponibles", components.get('models_count', 0))
            
            with col2_2:
                st.metric("Preprocessor", 
                         "✅" if components.get('preprocessor_loaded') else "❌")
                st.metric("Trainer", 
                         "✅" if components.get('trainer_loaded') else "❌")
        
        # Graphique factice pour l'accueil
        chart_data = pd.DataFrame({
            'Étapes': ['Upload', 'Preprocess', 'Training', 'Prediction'],
            'Statut': [1, 0.8, 0.6, 0.4]
        })
        
        fig = px.bar(chart_data, x='Étapes', y='Statut', 
                    title="Progression du Workflow",
                    color='Statut', color_continuous_scale='viridis')
        st.plotly_chart(fig, use_container_width=True)

def status_page():
    """Page d'état du système"""
    st.header("📊 État du Système")
    
    api_status, status_data = check_api_status()
    
    if not api_status:
        st.error("❌ Impossible de se connecter à l'API. Vérifiez que l'API Flask est démarrée.")
        return
    
    if status_data:
        # Informations générales
        st.subheader("ℹ️ Informations Générales")
        col1, col2, col3, col4 = st.columns(4)
        
        components = status_data.get('components', {})
        
        with col1:
            st.metric("API Status", "🟢 En ligne")
        with col2:
            st.metric("Preprocessor", "✅" if components.get('preprocessor_loaded') else "❌")
        with col3:
            st.metric("Trainer", "✅" if components.get('trainer_loaded') else "❌")
        with col4:
            st.metric("Modèles", components.get('models_count', 0))
        
        # Détails des composants
        st.subheader("🔧 Détails des Composants")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**État des Composants:**")
            for component, status in components.items():
                icon = "✅" if status else "❌"
                st.write(f"{icon} {component.replace('_', ' ').title()}")
        
        with col2:
            st.write("**Modèles Disponibles:**")
            available_models = status_data.get('available_models', [])
            if available_models:
                for model in available_models:
                    st.write(f"🤖 {model}")
            else:
                st.write("Aucun modèle disponible")
        
        # JSON complet
        with st.expander("📄 Réponse API Complète"):
            st.json(status_data)

def upload_page():
    """Page d'upload de données"""
    st.header("📤 Upload de Données")
    
    st.write("Uploadez vos fichiers CSV de données d'assurance :")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("👥 Fichier Bénéficiaires")
        bene_file = st.file_uploader(
            "Sélectionner le fichier des bénéficiaires", 
            type=['csv'],
            key="bene_file"
        )
        
        if bene_file:
            st.success(f"✅ Fichier chargé: {bene_file.name}")
            # Aperçu du fichier
            try:
                df_preview = pd.read_csv(bene_file, nrows=5)
                st.write("**Aperçu (5 premières lignes):**")
                st.dataframe(df_preview)
                st.write(f"**Colonnes ({len(df_preview.columns)}):** {', '.join(df_preview.columns)}")
            except Exception as e:
                st.error(f"Erreur lecture fichier: {e}")
    
    with col2:
        st.subheader("🏥 Fichier Réclamations")
        claims_file = st.file_uploader(
            "Sélectionner le fichier des réclamations",
            type=['csv'],
            key="claims_file"
        )
        
        if claims_file:
            st.success(f"✅ Fichier chargé: {claims_file.name}")
            # Aperçu du fichier
            try:
                df_preview = pd.read_csv(claims_file, nrows=5)
                st.write("**Aperçu (5 premières lignes):**")
                st.dataframe(df_preview)
                st.write(f"**Colonnes ({len(df_preview.columns)}):** {', '.join(df_preview.columns)}")
            except Exception as e:
                st.error(f"Erreur lecture fichier: {e}")
    
    # Bouton d'upload
    if st.button("🚀 Uploader vers l'API", type="primary"):
        if bene_file and claims_file:
            with st.spinner("Upload en cours..."):
                try:
                    files = {
                        'bene_file': (bene_file.name, bene_file.getvalue(), 'text/csv'),
                        'claims_file': (claims_file.name, claims_file.getvalue(), 'text/csv')
                    }

                    response = requests.post(f"{API_BASE_URL}/upload", files=files)

                    if response.status_code == 200:
                        result = response.json()
                        st.success("✅ Upload réussi !")

                        # Afficher les détails
                        st.subheader("📊 Détails des Fichiers Uploadés")

                        col1, col2 = st.columns(2)

                        with col1:
                            bene_info = result['files']['bene_file']
                            st.write("**Bénéficiaires:**")
                            st.write(f"- Shape: {bene_info['shape']}")
                            st.write(f"- Colonnes: {len(bene_info['columns'])}")

                        with col2:
                            claims_info = result['files']['claims_file']
                            st.write("**Réclamations:**")
                            st.write(f"- Shape: {claims_info['shape']}")
                            st.write(f"- Colonnes: {len(claims_info['columns'])}")

                    else:
                        st.error(f"❌ Erreur upload: {response.status_code}")
                        
                except Exception as e:
                    st.error(f"❌ Une erreur s'est produite pendant l'upload : {e}")
        else:
            st.warning("⚠️ Veuillez sélectionner les deux fichiers avant d'uploader.")

def preprocess_page():
    """Page de preprocessing"""
    st.header("🔧 Preprocessing des Données")
    st.info("Page de preprocessing en cours de développement...")
    
    if st.button("🚀 Lancer le Preprocessing"):
        with st.spinner("Preprocessing en cours..."):
            try:
                response = requests.post(f"{API_BASE_URL}/preprocess")
                if response.status_code == 200:
                    st.success("✅ Preprocessing réussi !")
                else:
                    st.error(f"❌ Erreur preprocessing: {response.status_code}")
            except Exception as e:
                st.error(f"❌ Erreur: {e}")

def training_page():
    """Page d'entraînement"""
    st.header("🤖 Entraînement des Modèles")
    st.markdown("Cliquez sur le bouton ci-dessous pour entraîner les modèles de classification et clustering.")

    model_types = st.multiselect(
        "Types de modèles à entraîner",
        ["classification", "clustering"],
        default=["classification", "clustering"]
    )

    optimize = st.checkbox("🔍 Optimisation automatique des modèles", value=True)

    if st.button("🚀 Entraîner les Modèles"):
        with st.spinner("⏳ Entraînement en cours..."):
            try:
                payload = {
                    "model_types": model_types,
                    "optimize": optimize
                }
                response = requests.post(f"{API_BASE_URL}/train", json=payload)
                
                if response.status_code == 200:
                    data = response.json()
                    st.success("✅ Entraînement terminé avec succès !")
                    
                    # Résumé des résultats
                    st.subheader("📊 Résultats d'Entraînement")
                    results = data.get("results", {})

                    if "classification" in results:
                        st.markdown("### 📘 Classification")
                        st.write(f"- Modèles entraînés : {results['classification']['models_trained']}")
                        st.write(f"- Liste : {', '.join(results['classification']['models'])}")

                    if "clustering" in results:
                        st.markdown("### 📗 Clustering")
                        st.write(f"- Modèles entraînés : {results['clustering']['models_trained']}")
                        st.write(f"- Liste : {', '.join(results['clustering']['models'])}")

                    # Meilleur modèle
                    best_model = results.get("best_model")
                    if best_model:
                        st.success(f"🏆 Meilleur modèle : **{best_model}**")

                    # Fichiers générés
                    st.markdown("### 📁 Fichiers Générés")
                    st.write(data.get("files_generated"))

                else:
                    error = response.json().get("error", "Erreur inconnue.")
                    st.error(f"❌ Erreur entraînement : {error}")

            except Exception as e:
                st.error(f"❌ Exception : {e}")


def predictions_page():
    """Page de prédictions"""
    st.header("🔮 Prédictions")
    st.info("Page de prédictions en cours de développement...")

def visualizations_page():
    """Page de visualisations"""
    st.header("📈 Visualisations Automatiques")

    if st.button("🖼️ Générer les visualisations"):
        with st.spinner("⏳ Génération des visualisations en cours..."):
            try:
                response = requests.get(f"{API_BASE_URL}/visualize")
                if response.status_code == 200:
                    data = response.json()
                    st.success(data.get("message", "✅ Visualisations générées."))

                    image_files = data.get("files", [])
                    image_dir = os.path.join(base_path, "reports", "figures")

                    if not image_files:
                        st.warning("⚠️ Aucune image générée.")
                        return

                    st.subheader("📂 Visualisations Générées")
                    for img_file in image_files:
                        img_path = os.path.join(image_dir, img_file)
                        try:
                            with open(img_path, "rb") as f:
                                img_bytes = f.read()
                            
                            st.image(img_bytes, caption=img_file, use_container_width=True)

                        except Exception as e:
                            st.error(f"❌ Impossible d'afficher {img_file}: {e}")

                else:
                    error = response.json().get("error", "Erreur inconnue.")
                    st.error(f"❌ Échec: {error}")

            except Exception as e:
                st.error(f"❌ Exception lors de la requête: {e}")


def models_page():
    """Page de gestion des modèles"""
    st.header("📋 Gestion des Modèles")
    st.info("Page de gestion des modèles en cours de développement...")

def run_insurance_app():
    """Fonction principale pour lancer l'application Streamlit"""
    
    display_header()
    page = sidebar_navigation()

    if page == "home":
        home_page()
    elif page == "status":
        status_page()
    elif page == "upload":
        upload_page()
    elif page == "preprocess":
        preprocess_page()
    elif page == "training":
        training_page()
    elif page == "predictions":
        predictions_page()
    elif page == "visualizations":
        visualizations_page()
    elif page == "models":
        models_page()

# Point d'entrée si le script est exécuté directement
if __name__ == "__main__":
    run_insurance_app()