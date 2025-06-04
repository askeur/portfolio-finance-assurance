import pandas as pd
import os
import sys


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
base_path = os.path.join(BASE_DIR, "01-credit-default-prediction", "credit-default-prediction") 
SRC_PATH = os.path.join(base_path, "src")
sys.path.insert(0, SRC_PATH)

from src.data_preparation import prepare_data 


def test_preprocess_data_removes_nulls():
    training_path = os.path.join(base_path, "data", "raw", "cs-training.csv")
    output_path = os.path.join(base_path, "data", "processed", "test_output.csv")

    processed = prepare_data(training_path, output_path)  # on passe le chemin ici

    assert processed.isnull().sum().sum() == 0
