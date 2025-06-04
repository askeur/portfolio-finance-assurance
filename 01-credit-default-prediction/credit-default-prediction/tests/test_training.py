import os
import sys
import pandas as pd
import pytest


# chemin vers le dossier 'src'
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
base_path = os.path.join(BASE_DIR, "01-credit-default-prediction", "credit-default-prediction") 
SRC_PATH = os.path.join(base_path, "src")
sys.path.insert(0, SRC_PATH)
from src.model_training import train_random_forest, train_logistic_regression, train_xgboost


MODELS = {
    "random_forest": train_random_forest,
    "logistic_regression": train_logistic_regression,
    "xgboost": train_xgboost
}

def test_train_all_models():
    # Load des données
    data_path = os.path.join(base_path, "data", "processed", "train_clean.csv")
    assert os.path.exists(data_path), f"⚠️ Le fichier {data_path} est introuvable."

    df = pd.read_csv(data_path)
    X = df.drop(columns=["SeriousDlqin2yrs"])
    y = df["SeriousDlqin2yrs"]

    for name, trainer in MODELS.items():
        model, _, _, _ = trainer(X, X, y, y)  #  X=X pour test rapide
        preds = model.predict(X)
        assert len(preds) == len(y), f"❌ Le modèle {name} retourne {len(preds)} prédictions pour {len(y)} échantillons"
