import pandas as pd
from src.model_training import train_model, evaluate_model

def test_evaluate_model_score():
    df = pd.read_csv("data/processed/train_clean.csv")
    X = df.drop(columns=["SeriousDlqin2yrs"])
    y = df["SeriousDlqin2yrs"]
    model = train_model("random_forest", X, y)
    score = evaluate_model(model, X, y)
    assert 0.0 <= score <= 1.0