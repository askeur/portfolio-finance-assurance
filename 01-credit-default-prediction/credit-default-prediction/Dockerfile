FROM python:3.10-slim

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers
COPY . .

# Installer les dépendances Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Exposer les ports
EXPOSE 8000 8501

# Lancer les deux services : FastAPI et Streamlit
CMD ["bash", "-c", "uvicorn api.main:app --host 0.0.0.0 --port 8000 & streamlit run app.py --server.port 8501 --server.address 0.0.0.0"]
