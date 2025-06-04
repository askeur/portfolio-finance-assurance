# Portfolio Finance & Assurance

Cette application Streamlit propose un portail interactif pour explorer plusieurs projets de data science dans le domaine de la finance et de l'assurance.

## Fonctionnalités

- **Authentification** : Accès sécurisé par identifiant et mot de passe.
- **Navigation par projet** : Sélectionnez parmi plusieurs projets :
  - 📈 Credit Default Prediction
  - 💳 Fraude Transactionnelle
  - 📋 Insurance Risk
  - 📉 Client Churn
- **Chargement dynamique** : L'interface et les modules changent selon le projet sélectionné.
- **Lancement automatique d'API** (pour le projet Credit Default Prediction).

## Prérequis

- Python 3.8+
- [Streamlit](https://streamlit.io/)
- [Uvicorn](https://www.uvicorn.org/) (pour l'API du projet 01)
- Autres dépendances spécifiques à chaque projet

## Installation

1. Clonez ce dépôt :
    ```sh
    git clone <url-du-repo>
    cd portfolio-finance-assurance
    ```

2. Installez les dépendances :
    ```sh
    pip install -r requirements.txt
    ```

## Utilisation

Lancez l'application principale :

```sh
streamlit run main.py
```

Connectez-vous avec :
- **Nom d'utilisateur** : `admin`
- **Mot de passe** : `1234`

Sélectionnez un projet dans l'interface pour démarrer.

## Structure des dossiers

- `main.py` : Application principale Streamlit.
- `01-credit-default-prediction/` : Projet prédiction de défaut de crédit (avec API FastAPI).
- `02-insurance-claims-analysis/` : Analyse de sinistres d'assurance.
- `03-fraud-detection-creditcard/` : Détection de fraude sur cartes bancaires.
- `04-customer-churn-bank/` : Prédiction de churn client bancaire.

## Notes

- Seul le projet **Credit Default Prediction** est entièrement intégré pour le moment.
- Les autres projets affichent un message "est en cours de déploiement".

---

*Pour toute question, contactez askeurnabila@gmail.com.*