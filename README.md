# 🌾 Crop Yield Prediction — Agritech

Application de prédiction du rendement des cultures basée sur le Machine Learning.

Le projet utilise un modèle **XGBoost** entraîné sur des données agricoles et expose les prédictions via une API **FastAPI**. Une interface **Streamlit** permet également d'interagir avec l'application.

Le projet intègre une chaîne **CI/CD avec GitHub Actions et Docker Hub** afin d'automatiser les tests, la construction et la publication de l'image Docker.

---

## 📌 Présentation du projet

L'objectif est de prédire le rendement agricole à partir de différentes variables liées :

- 🌍 au pays / à la zone géographique ;
- 🌱 à la culture ;
- 📅 à l'année ;
- 🌧️ aux précipitations ;
- 🌡️ à la température moyenne ;
- 🧪 à l'utilisation de pesticides.

La variable cible utilisée pour la prédiction est :

```text
Yield_crops_hg_per_ha
```

---

## 📊 Données

Le dataset final utilisé pour le projet est :

```text
outputs/dataset_final.csv
```

### Variables principales

| Variable | Description |
|---|---|
| `Area Code` | Code de la zone |
| `Area` | Pays / zone géographique |
| `Item Code` | Code de la culture |
| `Item` | Culture |
| `Year` | Année |
| `Yield_crops_hg_per_ha` | Rendement agricole |
| `average_rain_fall_mm_per_year` | Précipitations moyennes annuelles |
| `avg_temp` | Température moyenne |
| `Pesticide_totale_use_tonnes` | Utilisation totale de pesticides |

Le dataset couvre la période **1990–2013** et contient **101 pays/zones**.

---

# 🤖 Machine Learning

Plusieurs modèles ont été étudiés dans le cadre du projet :

- Régression linéaire
- Random Forest
- XGBoost

Le modèle utilisé pour l'API est :

```text
XGBoost
```

Le modèle entraîné est sauvegardé dans :

```text
model/model_xgb.pkl
```

Le modèle est chargé automatiquement au démarrage de l'API.

---

## 🔬 Prétraitement

Le pipeline de Machine Learning utilise notamment :

- traitement des variables catégorielles ;
- `OneHotEncoder` ;
- normalisation lorsque nécessaire ;
- pipeline Scikit-learn ;
- modèle XGBoost.

Le pipeline complet est sauvegardé avec le modèle afin de reproduire le même prétraitement lors des prédictions.

---

# 🧠 Analyse en composantes principales (PCA)

Une analyse PCA a également été réalisée afin d'étudier la structure des variables numériques.

Les quatre composantes expliquent :

| Composante | Variance expliquée |
|---|---:|
| PC1 | 34.38 % |
| PC2 | 28.17 % |
| PC3 | 22.54 % |
| PC4 | 14.91 % |

Les trois premières composantes expliquent cumulativement :

```text
85.09 %
```

Cela montre que trois composantes permettent de représenter plus de 80 % de la variance totale.

---

# 🚀 Architecture du projet

```text
agritech-prediction/
│
├── .github/
│   └── workflows/
│       └── ci-cd.yml
│
├── api/
│   ├── main.py
│   ├── requirements.txt
│   └── Dockerfile
│
├── model/
│   ├── model_xgb.pkl
│   └── Models.ipynb
│
├── outputs/
│   └── dataset_final.csv
│
├── mlops/
│   └── train_pipeline.py
│
├── streamlit/
│   ├── app.py
│   └── requirements.txt
│
├── tests/
│   └── test_api.py
│
├── Dockerfile
├── .gitignore
└── README.md
```

---

# ⚡ API FastAPI

L'API est développée avec **FastAPI**.

Elle charge automatiquement le modèle XGBoost :

```text
model/model_xgb.pkl
```

## Lancer l'API localement

Depuis la racine du projet :

```powershell
uvicorn api.main:app --reload
```

L'API est ensuite accessible à :

```text
http://localhost:8000
```

La documentation interactive Swagger est disponible à :

```text
http://localhost:8000/docs
```

---

# 🐳 Docker

Le projet contient un `Dockerfile` permettant de construire l'image complète de l'API.

## Construire l'image localement

Depuis la racine du projet :

```powershell
docker build -t crop-yield-api-test .
```

## Lancer le conteneur

```powershell
docker run --name crop-yield-api-test -p 8000:8000 crop-yield-api-test
```

L'API est alors disponible sur :

```text
http://localhost:8000/docs
```

---

# 🐳 Docker Hub

L'image Docker est publiée automatiquement sur Docker Hub.

Repository :

**amyhdev/crop-yield-api**

Image :

```text
amyhdev/crop-yield-api:latest
```

## Télécharger l'image depuis Docker Hub

```powershell
docker pull amyhdev/crop-yield-api:latest
```

## Lancer directement l'image Docker Hub

```powershell
docker run --name crop-yield-api-dockerhub -p 8000:8000 amyhdev/crop-yield-api:latest
```

Puis ouvrir :

```text
http://localhost:8000/docs
```

---

# 🔄 CI/CD avec GitHub Actions

Le projet utilise **GitHub Actions** pour automatiser le processus CI/CD.

Le workflow se trouve dans :

```text
.github/workflows/ci-cd.yml
```

À chaque `push` sur la branche `main`, le workflow réalise les étapes suivantes :

```text
Git Push
   │
   ▼
GitHub Actions
   │
   ├── Installation Python
   │
   ├── Installation des dépendances
   │
   ├── Vérification de la syntaxe
   │
   ├── Tests pytest
   │
   ▼
Docker Buildx
   │
   ▼
Docker Login
   │
   ▼
Docker Build
   │
   ▼
Docker Push
   │
   ▼
Docker Hub
```

### Tags Docker générés

L'image est publiée avec deux tags :

```text
latest
```

et le SHA du commit GitHub :

```text
<commit-sha>
```

Exemple :

```text
amyhdev/crop-yield-api:latest
amyhdev/crop-yield-api:151954abfff5c60638d9ed7f41f083ea8b19061e
```

---

# 🧪 Tests

Les tests sont exécutés automatiquement par GitHub Actions.

Pour lancer les tests localement :

```powershell
python -m pytest tests/ -v
```

La syntaxe Python peut également être vérifiée avec :

```powershell
python -m compileall -q api streamlit tests
```

---

# 🔐 Secrets GitHub

Le workflow utilise les informations d'authentification Docker Hub stockées dans GitHub Actions.

### Repository Variable

```text
DOCKERHUB_USERNAME
```

Valeur :

```text
amyhdev
```

### Repository Secret

```text
DOCKERHUB_TOKEN
```

Le token Docker Hub est stocké comme secret et n'est pas présent dans le code source.

---

# 📦 Technologies utilisées

| Technologie | Utilisation |
|---|---|
| Python 3.11 | Langage principal |
| Pandas | Manipulation des données |
| NumPy | Calcul numérique |
| Scikit-learn | Prétraitement et Machine Learning |
| XGBoost | Modèle de prédiction |
| FastAPI | API REST |
| Uvicorn | Serveur ASGI |
| Streamlit | Interface utilisateur |
| Joblib | Sauvegarde / chargement du modèle |
| Docker | Conteneurisation |
| Docker Hub | Registry d'images |
| GitHub Actions | CI/CD |
| Pytest | Tests |

---

# 📈 Workflow Machine Learning

```text
Données agricoles
       │
       ▼
Nettoyage des données
       │
       ▼
Analyse exploratoire
       │
       ▼
Prétraitement
       │
       ▼
OneHotEncoder
       │
       ▼
Entraînement des modèles
       │
       ├── Régression linéaire
       ├── Random Forest
       └── XGBoost
              │
              ▼
        Modèle sélectionné
              │
              ▼
       model_xgb.pkl
              │
              ▼
          FastAPI
              │
              ▼
        Prédiction
```

---

# 🎯 Objectif final

Le projet permet de mettre en place une solution complète de Machine Learning appliquée à l'agriculture :

**Données → Prétraitement → Modèle ML → API → Docker → CI/CD → Docker Hub**

Cette architecture permet de reproduire et déployer plus facilement le modèle de prédiction du rendement agricole.

---

# 👩‍💻 Auteur

**Amela Hadžiavdić**

Projet de Machine Learning / MLOps appliqué à la prédiction du rendement agricole.
```

