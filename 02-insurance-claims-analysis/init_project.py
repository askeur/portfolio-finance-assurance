import os

# Dossiers à créer
folders = [
    "insurance-claims-analysis/app",
    "insurance-claims-analysis/api",
    "insurance-claims-analysis/models",
    "insurance-claims-analysis/src",
    "insurance-claims-analysis/data/raw",
    "insurance-claims-analysis/data/processed",
    "insurance-claims-analysis/notebooks",
    "insurance-claims-analysis/tests",
]

# Fichiers à créer
files = [
    "insurance-claims-analysis/README.md",
    "insurance-claims-analysis/requirements.txt",
    "insurance-claims-analysis/Dockerfile",
    "insurance-claims-analysis/docker-compose.yml",
    "insurance-claims-analysis/.dockerignore",
    "insurance-claims-analysis/app/insurance_app.py",
    "insurance-claims-analysis/api/main.py",
    "insurance-claims-analysis/src/model_training.py",
    "insurance-claims-analysis/src/data_preparation.py",
    "insurance-claims-analysis/src/visualization.py",
    "insurance-claims-analysis/tests/test_api.py",
    "insurance-claims-analysis/tests/test_evaluation.py",
    "insurance-claims-analysis/tests/test_preprocessing.py",
    "insurance-claims-analysis/tests/test_training.py",
]

# Création des dossiers
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Création des fichiers
for file in files:
    with open(file, "w") as f:
        pass  # fichier vide pour le moment

print("✅ Structure de projet créée dans 'insurance-claims-analysis'")
