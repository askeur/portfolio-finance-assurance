#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
API Flask pour l'Analyse d'Assurance
Intègre le preprocessing, modélisation et prédictions
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import joblib
import io
import json
from datetime import datetime
import logging
from werkzeug.utils import secure_filename
import tempfile

# Importer nos modules personnalisés
from data_preparation import InsuranceDataPreprocessor
from model_training import InsuranceModelTrainer
from visualization import InsuranceDataVisualizer

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['RESULTS_FOLDER'] = 'results'

# Créer les dossiers nécessaires
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('data/processed', exist_ok=True)
os.makedirs('reports/figures', exist_ok=True)

# Variables globales pour stocker les instances
preprocessor = None
trainer = None
visualizer = InsuranceDataVisualizer()

@app.route('/')
def home():
    """Page d'accueil de l'API"""
    return jsonify({
        "message": "🏥 API d'Analyse d'Assurance",
        "version": "1.0.0",
        "endpoints": {
            "upload": "/upload - POST - Uploader des fichiers CSV",
            "preprocess": "/preprocess - POST - Préprocesser les données",
            "train": "/train - POST - Entraîner les modèles",
            "predict": "/predict - POST - Faire des prédictions",
            "visualize": "/visualize - GET - Générer des visualisations",
            "status": "/status - GET - État du système",
            "models": "/models - GET - Liste des modèles disponibles"
        }
    })

@app.route('/status')
def status():
    """État du système"""
    global preprocessor, trainer
    
    models_available = []
    if os.path.exists('models'):
        models_available = [f for f in os.listdir('models') if f.endswith('.joblib')]
    
    processed_data_exists = os.path.exists('data/processed/processed_data.csv')
    
    return jsonify({
        "status": "✅ Opérationnel",
        "timestamp": datetime.now().isoformat(),
        "components": {
            "preprocessor_loaded": preprocessor is not None,
            "trainer_loaded": trainer is not None,
            "processed_data_available": processed_data_exists,
            "models_count": len(models_available)
        },
        "available_models": models_available,
        "data_files": {
            "processed_data": processed_data_exists,
            "upload_folder": os.path.exists(app.config['UPLOAD_FOLDER'])
        }
    })

@app.route('/upload', methods=['POST'])
def upload_files():
    """Uploader des fichiers CSV"""
    try:
        if 'bene_file' not in request.files or 'claims_file' not in request.files:
            return jsonify({
                "error": "❌ Fichiers manquants. Requis: bene_file, claims_file"
            }), 400
        
        bene_file = request.files['bene_file']
        claims_file = request.files['claims_file']
        
        if bene_file.filename == '' or claims_file.filename == '':
            return jsonify({"error": "❌ Noms de fichiers vides"}), 400
        
        # Sauvegarder les fichiers
        bene_filename = secure_filename(bene_file.filename)
        claims_filename = secure_filename(claims_file.filename)
        
        bene_path = os.path.join(app.config['UPLOAD_FOLDER'], bene_filename)
        claims_path = os.path.join(app.config['UPLOAD_FOLDER'], claims_filename)
        
        bene_file.save(bene_path)
        claims_file.save(claims_path)
        
        # Vérifier les fichiers
        try:
            bene_df = pd.read_csv(bene_path)
            claims_df = pd.read_csv(claims_path)
            
            return jsonify({
                "message": "✅ Fichiers uploadés avec succès",
                "files": {
                    "bene_file": {
                        "filename": bene_filename,
                        "shape": bene_df.shape,
                        "columns": list(bene_df.columns)
                    },
                    "claims_file": {
                        "filename": claims_filename,
                        "shape": claims_df.shape,
                        "columns": list(claims_df.columns)
                    }
                }
            })
            
        except Exception as e:
            return jsonify({
                "error": f"❌ Erreur lors de la lecture des CSV: {str(e)}"
            }), 400
            
    except Exception as e:
        logger.error(f"Erreur upload: {str(e)}")
        return jsonify({"error": f"❌ Erreur upload: {str(e)}"}), 500

@app.route('/preprocess', methods=['POST'])
def preprocess_data():
    """Préprocesser les données"""
    global preprocessor
    
    try:
        # Récupérer les paramètres
        data = request.get_json() or {}
        bene_filename = data.get('bene_filename', 'bene_file.csv')
        claims_filename = data.get('claims_filename', 'Inpatient_Claim.csv')
        
        # Chemins des fichiers
        bene_path = os.path.join(app.config['UPLOAD_FOLDER'], bene_filename)
        claims_path = os.path.join(app.config['UPLOAD_FOLDER'], claims_filename)
        
        # Vérifier que les fichiers existent
        if not os.path.exists(bene_path) or not os.path.exists(claims_path):
            return jsonify({
                "error": "❌ Fichiers non trouvés. Uploadez d'abord les fichiers."
            }), 400
        
        # Initialiser le preprocessor
        preprocessor = InsuranceDataPreprocessor()
        
        # Pipeline de preprocessing
        logger.info("Début du preprocessing...")
        
        # 1. Charger les données
        bene_df, claims_df = preprocessor.load_data(bene_path, claims_path)
        if bene_df is None or claims_df is None:
            return jsonify({"error": "❌ Erreur lors du chargement des données"}), 500
        
        # 2. Explorer les données
        exploration_stats = {
            "bene_shape": bene_df.shape,
            "claims_shape": claims_df.shape,
            "bene_columns": list(bene_df.columns),
            "claims_columns": list(claims_df.columns)
        }
        
        # 3. Fusionner les données
        merged_df = preprocessor.merge_data()
        
        # 4. Nettoyer les données
        cleaned_df = preprocessor.clean_data()
        
        # 5. Ingénierie des features
        feature_df = preprocessor.feature_engineering()
        
        # 6. Sauvegarder les données traitées
        preprocessor.save_processed_data()
        
        return jsonify({
            "message": "✅ Preprocessing terminé avec succès",
            "results": {
                "original_data": exploration_stats,
                "merged_shape": merged_df.shape,
                "cleaned_shape": cleaned_df.shape,
                "final_shape": feature_df.shape,
                "final_columns": list(feature_df.columns),
                "processed_file": "data/processed/processed_data.csv"
            }
        })
        
    except Exception as e:
        logger.error(f"Erreur preprocessing: {str(e)}")
        return jsonify({
            "error": f"❌ Erreur preprocessing: {str(e)}"
        }), 500

@app.route('/train', methods=['POST'])
def train_models():
    """Entraîner les modèles"""
    global trainer
    
    try:
        # Paramètres
        data = request.get_json() or {}
        model_types = data.get('model_types', ['classification', 'clustering'])
        optimize = data.get('optimize', True)
        
        # Vérifier que les données préprocessées existent
        processed_path = "data/processed/processed_data.csv"
        if not os.path.exists(processed_path):
            return jsonify({
                "error": "❌ Données préprocessées non trouvées. Lancez d'abord le preprocessing."
            }), 400
        
        # Initialiser le trainer
        trainer = InsuranceModelTrainer()
        
        # Charger les données
        df = trainer.load_processed_data(processed_path)
        if df is None:
            return jsonify({"error": "❌ Erreur lors du chargement des données"}), 500
        
        results = {}
        
        # Préparer les données pour l'entraînement
        # Simuler une target pour la classification (à adapter selon vos données)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            # Créer une target factice basée sur des seuils
            target_col = numeric_cols[0]  # Première colonne numérique
            threshold = df[target_col].median()
            y = (df[target_col] > threshold).astype(int)
            X = df[numeric_cols].fillna(0)
            
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        
        # Classification
        if 'classification' in model_types:
            trainer.setup_classification_models()
            classification_results = trainer.train_classification_models(
                X_train, X_test, y_train, y_test, optimize=optimize
            )
            results['classification'] = {
                "models_trained": len(classification_results),
                "models": list(classification_results.keys())
            }
        
        # Clustering
        if 'clustering' in model_types:
            trainer.setup_clustering_models()
            clustering_results = trainer.train_clustering_models(X, optimize=optimize)
            results['clustering'] = {
                "models_trained": len(clustering_results),
                "models": list(clustering_results.keys())
            }
        
        # Évaluation et sauvegarde
        if 'classification' in model_types:
            comparison_df = trainer.evaluate_models(X_test, y_test)
            trainer.generate_detailed_report(X_test, y_test)
            results['best_model'] = trainer.best_model_name if hasattr(trainer, 'best_model_name') else None
        
        trainer.save_models()
        
        return jsonify({
            "message": "✅ Entraînement terminé avec succès",
            "results": results,
            "files_generated": {
                "models": "models/",
                "reports": "reports/"
            }
        })
        
    except Exception as e:
        logger.error(f"Erreur entraînement: {str(e)}")
        return jsonify({
            "error": f"❌ Erreur entraînement: {str(e)}"
        }), 500

@app.route('/predict', methods=['POST'])
def predict():
    """Faire des prédictions"""
    global trainer
    
    try:
        # Récupérer les données
        data = request.get_json()
        if not data or 'features' not in data:
            return jsonify({
                "error": "❌ Données manquantes. Format requis: {'features': [[...]]}"
            }), 400
        
        # Charger le meilleur modèle si pas déjà chargé
        if trainer is None or not hasattr(trainer, 'best_model'):
            model_files = [f for f in os.listdir('models') if f.startswith('best_model_')]
            if not model_files:
                return jsonify({
                    "error": "❌ Aucun modèle entraîné trouvé. Entraînez d'abord un modèle."
                }), 400
            
            model_path = os.path.join('models', model_files[0])
            model = joblib.load(model_path)
            
            # Simuler un trainer pour les prédictions
            trainer = InsuranceModelTrainer()
            trainer.best_model = model
        
        # Faire les prédictions
        features = np.array(data['features'])
        predictions, probabilities = trainer.predict_new_data(features)
        
        result = {
            "predictions": predictions.tolist() if predictions is not None else None,
            "probabilities": probabilities.tolist() if probabilities is not None else None,
            "model_used": getattr(trainer, 'best_model_name', 'unknown'),
            "num_predictions": len(predictions) if predictions is not None else 0
        }
        
        return jsonify({
            "message": "✅ Prédictions générées",
            "results": result
        })
        
    except Exception as e:
        logger.error(f"Erreur prédiction: {str(e)}")
        return jsonify({
            "error": f"❌ Erreur prédiction: {str(e)}"
        }), 500

@app.route('/visualize', methods=['GET'])
def generate_visualizations():
    """Générer les visualisations"""
    try:
        # Vérifier que les données existent
        processed_path = "data/processed/processed_data.csv"
        if not os.path.exists(processed_path):
            return jsonify({
                "error": "❌ Données préprocessées non trouvées"
            }), 400
        
        # Générer les visualisations
        visualizer.visualize_insurance_data(
            processed_data_path=processed_path,
            output_dir="reports/figures"
        )
        
        # Lister les fichiers générés
        figures_dir = "reports/figures"
        generated_files = []
        if os.path.exists(figures_dir):
            generated_files = [f for f in os.listdir(figures_dir) if f.endswith('.png')]
        
        return jsonify({
            "message": "✅ Visualisations générées",
            "files": generated_files,
            "location": figures_dir
        })
        
    except Exception as e:
        logger.error(f"Erreur visualisation: {str(e)}")
        return jsonify({
            "error": f"❌ Erreur visualisation: {str(e)}"
        }), 500

@app.route('/models')
def list_models():
    """Lister les modèles disponibles"""
    try:
        models_dir = 'models'
        model_files = []
        
        if os.path.exists(models_dir):
            for filename in os.listdir(models_dir):
                if filename.endswith('.joblib'):
                    filepath = os.path.join(models_dir, filename)
                    model_info = {
                        "filename": filename,
                        "size_mb": round(os.path.getsize(filepath) / (1024*1024), 2),
                        "modified": datetime.fromtimestamp(
                            os.path.getmtime(filepath)
                        ).isoformat()
                    }
                    model_files.append(model_info)
        
        return jsonify({
            "models": model_files,
            "count": len(model_files)
        })
        
    except Exception as e:
        return jsonify({
            "error": f"❌ Erreur listage modèles: {str(e)}"
        }), 500

@app.route('/download/<path:filename>')
def download_file(filename):
    """Télécharger un fichier généré"""
    try:
        # Sécuriser le chemin
        filename = secure_filename(filename)
        
        # Chercher dans les différents dossiers
        possible_paths = [
            os.path.join('reports/figures', filename),
            os.path.join('reports', filename),
            os.path.join('data/processed', filename),
            os.path.join('models', filename)
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return send_file(path, as_attachment=True)
        
        return jsonify({"error": "❌ Fichier non trouvé"}), 404
        
    except Exception as e:
        return jsonify({
            "error": f"❌ Erreur téléchargement: {str(e)}"
        }), 500

@app.errorhandler(413)
def too_large(e):
    return jsonify({"error": "❌ Fichier trop volumineux (max 100MB)"}), 413

@app.errorhandler(500)
def internal_error(e):
    return jsonify({"error": "❌ Erreur interne du serveur"}), 500

if __name__ == '__main__':
    print("🚀 Démarrage de l'API d'Analyse d'Assurance...")
    app.run(host='0.0.0.0', port=5000, debug=True)