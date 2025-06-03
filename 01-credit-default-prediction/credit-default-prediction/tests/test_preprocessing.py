import pandas as pd
from src.model_training import preprocess_data

def test_preprocess_data_removes_nulls():
    df = pd.read_csv("data/raw/cs-training.csv")
    processed = preprocess_data(df)
    assert processed.isnull().sum().sum() == 0