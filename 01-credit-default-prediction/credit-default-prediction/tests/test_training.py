import pandas as pd
from src.model_training import train_model

MODELS = [
    "xgboost",
    "random_forest",
    "logistic_regression",
    "lightgbm",
    "catboost"
]

def test_train_all_models():
    df = pd.read_csv("data/processed/train_clean.csv")
    X = df.drop(columns=["SeriousDlqin2yrs"])
    y = df["SeriousDlqin2yrs"]
    for model_name in MODELS:
        model = train_model(model_name, X, y)
        preds = model.predict(X)
        assert len(preds) == len(y), f"Model {model_name} failed: prediction length mismatch"