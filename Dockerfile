# Image de base Python
FROM python:3.9-slim

# Dossier de travail dans le conteneur
WORKDIR /app

# Copier les dépendances
COPY requirements.txt .

# Installer les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copier le reste de l'application
COPY . .

# Exposer le port Flask
EXPOSE 5000

# Lancer l'API
CMD ["python", "app.py"]
