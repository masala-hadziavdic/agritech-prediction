import numpy as np
import joblib
from fastapi.testclient import TestClient
from api import main

# ============================================================
# CLIENT API
# ============================================================

client = TestClient(main.app)

# ============================================================
# CHARGEMENT DU VRAI MODELE XGBOOST
# ============================================================

# Chargement du modèle entraîné sauvegardé dans le projet
main.model = joblib.load("model/model_xgb.pkl")

# Liste des cultures utilisées par l'API
main.CROPS = [
    "Maize",
    "Cassava",
    "Rice",
]

# ============================================================
# TEST 1 : ENDPOINT PRINCIPAL
# ============================================================

def test_home():
    """Vérifie que l'API est accessible."""

    response = client.get("/")

    assert response.status_code == 200

# ============================================================
# TEST 2 : PREDICTION AVEC LE VRAI MODELE
# ============================================================

def test_predict_success():
    """Vérifie qu'une prédiction valide fonctionne avec le vrai modèle."""

    payload = {
        "Year": 2013,
        "Area": "France",
        "Item": "Maize",
        "average_rain_fall_mm_per_year": 1200,
        "avg_temp": 20.5,
        "Pesticide_totale_use_tonnes": 5000
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200

    result = response.json()

    # Vérification de la réponse API
    assert result["success"] is True
    assert result["culture"] == "Maize"

    # Vérification des prédictions
    assert result["predicted_yield_hg_per_ha"] >= 0
    assert result["predicted_yield_kg_per_ha"] >= 0

    # Vérification que les valeurs retournées sont numériques
    assert isinstance(
        result["predicted_yield_hg_per_ha"],
        (int, float)
    )

    assert isinstance(
        result["predicted_yield_kg_per_ha"],
        (int, float)
    )

# ============================================================
# TEST 3 : CULTURE INCONNUE
# ============================================================

def test_predict_unknown_crop():
    """Vérifie que l'API refuse une culture qui n'existe pas."""

    payload = {
        "Year": 2011,
        "Area": "Albania",
        "Item": "UnknownCrop",
        "average_rain_fall_mm_per_year": 900,
        "avg_temp": 18.5,
        "Pesticide_totale_use_tonnes": 500
    }

    response = client.post("/predict", json=payload)

    # L'API doit retourner une erreur HTTP 400
    assert response.status_code == 400

# ============================================================
# TEST 4 : RECOMMANDATIONS
# ============================================================

def test_recommend_success():
    """Vérifie que l'API génère des recommandations avec le vrai modèle."""

    payload = {
        "Year": 2012,
        "Area": "Afghanistan",
        "average_rain_fall_mm_per_year": 200,
        "avg_temp": 19.5,
        "Pesticide_totale_use_tonnes": 1000
    }

    response = client.post("/recommend", json=payload)

    assert response.status_code == 200

    result = response.json()

    # Vérification générale
    assert result["success"] is True

    # L'API doit retourner 3 recommandations
    assert len(result["recommendations"]) == 3

    # Vérification de la présence des cultures
    recommended_crops = [
        item["culture"]
        for item in result["recommendations"]
    ]

    assert len(recommended_crops) == 3

    # Vérification que les rendements sont numériques
    yields = [
        item["predicted_yield_kg_per_ha"]
        for item in result["recommendations"]
    ]

    assert all(isinstance(value, (int, float)) for value in yields)

    # Les recommandations doivent être classées
    # du rendement prédit le plus élevé au plus faible.
    assert yields == sorted(yields, reverse=True)