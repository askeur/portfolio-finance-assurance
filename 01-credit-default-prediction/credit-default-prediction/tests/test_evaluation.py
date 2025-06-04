import os
import sys
import pandas as pd


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
print(f"Base directory: {BASE_DIR}")
base_path = os.path.join(BASE_DIR, "01-credit-default-prediction", "credit-default-prediction") 
print(f"Base path: {base_path}")    
SRC_PATH = os.path.join(base_path, "src")
print(f"Source path: {SRC_PATH}")
sys.path.insert(0, SRC_PATH)

from src.model_training import train_random_forest, evaluate_model

def test_evaluate_model_score():
    data_path = os.path.join(base_path, "data", "processed", "train_clean.csv")
    assert os.path.exists(data_path), f"Fichier introuvable : {data_path}"

    df = pd.read_csv(data_path)
    X = df.drop(columns=["SeriousDlqin2yrs"])
    y = df["SeriousDlqin2yrs"]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model, _, _, roc_auc = train_random_forest(X_train, X_test, y_train, y_test)
    
    assert 0.0 <= roc_auc <= 1.0, f"ROC AUC invalide : {roc_auc}"
    assert roc_auc > 0.5, f"Modèle non performant (ROC AUC = {roc_auc})"
    print(f"✅ ROC AUC obtenu : {roc_auc:.4f}")
