import os

# Dossiers à créer
folders = [
    "credit-default-prediction/app",
    "credit-default-prediction/api",
    "credit-default-prediction/models",
    "credit-default-prediction/src",
    "credit-default-prediction/data/raw",
    "credit-default-prediction/data/processed",
    "credit-default-prediction/notebooks",
    "credit-default-prediction/tests",
]

# Fichiers à créer
files = [
    "credit-default-prediction/README.md",
    "credit-default-prediction/requirements.txt",
    "credit-default-prediction/Dockerfile",
    "credit-default-prediction/docker-compose.yml",
    "credit-default-prediction/.dockerignore",
    "credit-default-prediction/app/credit_app.py",
    "credit-default-prediction/api/main.py",
    "credit-default-prediction/src/model_training.py",
    "credit-default-prediction/src/data_preparation.py",
    "credit-default-prediction/src/visualization.py",
    "credit-default-prediction/tests/test_api.py",
    "credit-default-prediction/tests/test_evaluation.py",
    "credit-default-prediction/tests/test_preprocessing.py",
    "credit-default-prediction/tests/test_training.py",
]

# Création des dossiers
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Création des fichiers
for file in files:
    with open(file, "w") as f:
        pass  # fichier vide pour le moment

print("✅ Structure de projet créée dans 'credit-default-prediction'")
