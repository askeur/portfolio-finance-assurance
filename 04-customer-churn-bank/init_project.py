import os

# Dossiers à créer
folders = [
    "customer-churn-bank/app",
    "customer-churn-bank/api",
    "customer-churn-bank/models",
    "customer-churn-bank/src",
    "customer-churn-bank/data/raw",
    "customer-churn-bank/data/processed",
    "customer-churn-bank/notebooks",
    "customer-churn-bank/tests",
   
]

# Fichiers à créer
files = [
    "customer-churn-bank/README.md",
    "customer-churn-bank/requirements.txt",
    "customer-churn-bank/Dockerfile",
    "customer-churn-bank/docker-compose.yml",
    "customer-churn-bank/.dockerignore",
    "customer-churn-bank/app/churn_app.py",
    "customer-churn-bank/api/main.py",
    "customer-churn-bank/src/model_training.py",
    "customer-churn-bank/src/data_preparation.py",
    "customer-churn-bank/src/visualization.py",
    "customer-churn-bank/tests/test_api.py",
    "customer-churn-bank/tests/test_evaluation.py",
    "customer-churn-bank/tests/test_preprocessing.py",
    "customer-churn-bank/tests/test_training.py",
]

# Création des dossiers
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# Création des fichiers
for file in files:
    with open(file, "w") as f:
        pass  # fichier vide pour le moment

print("✅ Structure de projet créée dans 'customer-churn-bank'")
