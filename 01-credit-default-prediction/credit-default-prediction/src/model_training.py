import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

def evaluate_model(model, X_train, X_val, y_train, y_val):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]

    report = classification_report(y_val, y_pred)
    matrix = confusion_matrix(y_val, y_pred).tolist()
    roc_auc = roc_auc_score(y_val, y_proba)

    return model, report, matrix, roc_auc


def load_data(data_path):
    df = pd.read_csv(data_path)
    X = df.drop(columns=["SeriousDlqin2yrs"])
    y = df["SeriousDlqin2yrs"]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def train_random_forest(X_train, X_val, y_train, y_val):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    return evaluate_model(model, X_train, X_val, y_train, y_val)


def train_logistic_regression(X_train, X_val, y_train, y_val):
    model = LogisticRegression(max_iter=1000, class_weight='balanced')
    return evaluate_model(model, X_train, X_val, y_train, y_val)


def train_xgboost(X_train, X_val, y_train, y_val):
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    return evaluate_model(model, X_train, X_val, y_train, y_val)


def save_results(model, model_name, report, matrix, roc_auc, model_path, json_path):
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)

    results = {
        "model": model_name,
        "report_text": report,
        "confusion_matrix": {"matrix": matrix},
        "roc_auc": roc_auc
    }

    os.makedirs(os.path.dirname(json_path), exist_ok=True)
    with open(json_path, "w") as f:
        import json
        json.dump(results, f, indent=4)

    print(f"✅ Modèle {model_name} sauvegardé à {model_path}")
    print(f"✅ Résultats sauvegardés à {json_path}")


if __name__ == "__main__":
    X_train, X_val, y_train, y_val = load_data("data/processed/train_clean.csv")

    models = {
        "random_forest": train_random_forest,
        "logistic_regression": train_logistic_regression,
        "xgboost": train_xgboost
    }

    for name, trainer in models.items():
        model, report, matrix, roc_auc = trainer(X_train, X_val, y_train, y_val)
        save_results(
            model, name, report, matrix, roc_auc,
            
            BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))  # Remonter à la racine du projet 
            print("BASE_DIR",BASE_DIR)
            base_path = os.path.join(BASE_DIR, "01-credit-default-prediction", "credit-default-prediction")
            print("base_path",base_path)
            json_path = os.path.join(base_path, "data", "processed", f"{name}_eval.json")
            model_path = os.path.join(base_path, "models", f"{selected_model}.pkl")
            
        )
