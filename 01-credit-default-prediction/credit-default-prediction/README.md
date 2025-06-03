# 🏦 Credit Default Prediction – Streamlit App

Application complète de scoring crédit, inspirée des pratiques en entreprise : entraînement automatique de modèles, interprétabilité (SHAP), API d’analyse, visualisation EDA — le tout packagé avec Docker.

---

## 📁 Structure du projet

```bash
credit-default-prediction/
├── app/                   # Application Streamlit principale
│   └── app.py
├── api/                   # API FastAPI pour l’analyse automatique
│   └── main.py
├── src/                   # Scripts de traitement et d'entraînement
│   └── model_training.py
├── models/                # Modèles entraînés (.pkl)
├── data/
│   ├── raw/               # Données brutes Kaggle
│   └── processed/         # Données nettoyées
├── notebooks/             # Analyses exploratoires (optionnel)
├── reports/figures/       # Graphiques EDA générés
├── tests/                 # Tests unitaires
├── Dockerfile             # Image Docker
├── docker-compose.yml     # Orchestration de l'app + API
├── requirements.txt       # Dépendances Python
├── .dockerignore          # Fichiers à ignorer dans Docker
└── README.md              # Ce fichier
```

---

## 🔍 Description

Cette application propose un pipeline complet de scoring crédit :

- 📥 Préparation des données à partir des fichiers bruts Kaggle
- 🤖 Entraînement automatique de modèles (Random Forest, XGBoost, etc.)
- 📈 Évaluation avec métriques et courbes ROC
- 🧠 Interprétabilité des prédictions via SHAP (globale & locale)
- 🔌 Analyse automatique via API FastAPI
- 📊 Exploration visuelle avec figures EDA (distribution, boxplots, corrélation)

---

## 🗃️ Données utilisées

- **Source** : Compétition Kaggle [Give Me Some Credit](https://www.kaggle.com/c/GiveMeSomeCredit)
- **Emplacement** : Placez les fichiers bruts dans `data/raw/`

---

## 🚀 Lancer l’application

### Avec Docker

1. Construisez et lancez l’ensemble :
    ```bash
    cd credit-default-prediction
    docker-compose up --build
    ```

2. Accédez à l’application :
    - 🖥️ Streamlit : [http://localhost:8501](http://localhost:8501)
    - ⚙️ API FastAPI : [http://localhost:8000](http://localhost:8000)

3. Pour arrêter :
    ```bash
    docker-compose down
    ```

### Sans Docker

- **Terminal 1** – Lancer l’API FastAPI :
    ```bash
    uvicorn api.main:app --reload --port 8000
    ```
- **Terminal 2** – Lancer Streamlit :
    ```bash
    streamlit run app/app.py
    ```

---

## 🧰 Technologies

| Composant      | Stack                  |
|----------------|------------------------|
| Frontend       | Streamlit              |
| Backend API    | FastAPI                |
| Modélisation   | Scikit-learn, XGBoost  |
| Interprétation | SHAP                   |
| Visualisation  | Matplotlib, Seaborn    |
| Orchestration  | Docker, Docker Compose |

---

## 📊 Visualisation des données (EDA)

Le script `src/visualize_data.py` génère automatiquement :

- Distribution de la variable cible
- Histogrammes des variables numériques
- Boxplots par rapport à la cible
- Matrice de corrélation

Les figures sont sauvegardées dans `reports/figures/`.

---

## 🧪 Tests unitaires

Les tests sont à développer dans le dossier `tests/`. Exemples attendus :

- Prétraitement des données
- Entraînement de modèles
- Résultats d’évaluation
- Réponses de l’API

---

*Pour toute question ou contribution, n’hésitez pas à ouvrir une issue ou une pull request !*