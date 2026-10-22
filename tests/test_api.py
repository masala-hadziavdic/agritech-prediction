import numpy as np
from fastapi.testclient import TestClient

from api import main


client = TestClient(main.app)


class FakeModel:
    """Remplace le vrai modèle pour tester l'API sans effectuer d'entraînement."""

    def predict(self, data):
        # Simule une prédiction sur l'échelle log1p.
        return np.array([np.log1p(45000.0)])


def setup_function():
    """Prépare un modèle simulé et une liste de cultures avant chaque test."""
    main.model = FakeModel()
    main.CROPS = ["Maize", "Cassava", "Rice"]


def test_home():
    response = client.get("/")
    assert response.status_code == 200


def test_predict_success():
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
    assert result["success"] is True
    assert result["culture"] == "Maize"
    assert result["predicted_yield_hg_per_ha"] >= 0
    assert result["predicted_yield_kg_per_ha"] >= 0


def test_predict_unknown_crop():
    payload = {
        "Year": 2011,
        "Area": "Albania",
        "Item": "UnknownCrop",
        "average_rain_fall_mm_per_year": 900,
        "avg_temp": 18.5,
        "Pesticide_totale_use_tonnes": 500
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 400


def test_recommend_success():
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
    assert result["success"] is True
    assert len(result["recommendations"]) == 3

    yields = [
        item["predicted_yield_kg_per_ha"]
        for item in result["recommendations"]
    ]
    assert yields == sorted(yields, reverse=True)
