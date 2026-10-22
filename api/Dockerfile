# Image Python

FROM python:3.11-slim

# Variables Python

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1


# Répertoire de travail

WORKDIR /app


# Installation des dépendances

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt


# Copie du projet

COPY . .


# Port FastAPI

EXPOSE 8000


# Lancement de l'API

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]