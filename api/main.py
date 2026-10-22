from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from pathlib import Path
import pandas as pd
import numpy as np
import joblib


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = Path(r"C:\Users\amela\model\model_xgb.pkl")
DATASET_PATH = BASE_DIR / "outputs" / "dataset_final.csv"


# ============================================================
# CHARGEMENT DU MODÈLE
# ============================================================

try:
    model = joblib.load(MODEL_PATH)

    print("✅ Modèle XGBoost chargé avec succès !")
    print(f"Type : {type(model)}")
    print(f"Méthode predict disponible : {hasattr(model, 'predict')}")

except Exception as e:
    model = None
    print(f"❌ Erreur lors du chargement du modèle : {e}")


# ============================================================
# CHARGEMENT DU DATASET
# ============================================================

try:
    dataset = pd.read_csv(DATASET_PATH)

    # Liste de toutes les cultures connues du modèle/dataset
    CROPS = sorted(dataset["Item"].dropna().unique().tolist())

    print(f"✅ Dataset chargé : {dataset.shape}")
    print(f"✅ Nombre de cultures disponibles : {len(CROPS)}")
    print(f"🌱 Cultures : {CROPS}")

except Exception as e:
    dataset = None
    CROPS = []

    print(f"❌ Erreur lors du chargement du dataset : {e}")


# ============================================================
# APPLICATION FASTAPI
# ============================================================

app = FastAPI(
    title="Crop Yield Prediction API",
    description=(
        "API de prédiction et de recommandation de cultures "
        "basée sur un modèle XGBoost."
    ),
    version="1.0.0"
)


# ============================================================
# SCHEMA POUR /predict
# ============================================================

class PredictionInput(BaseModel):

    Year: int = Field(
        ...,
        description="Année de prédiction"
    )

    Area: str = Field(
        ...,
        description="Pays ou zone géographique"
    )

    Item: str = Field(
        ...,
        description="Culture à prédire"
    )

    average_rain_fall_mm_per_year: float = Field(
        ...,
        description="Précipitations moyennes annuelles en mm"
    )

    avg_temp: float = Field(
        ...,
        description="Température moyenne"
    )

    Pesticide_totale_use_tonnes: float = Field(
        ...,
        description="Utilisation totale de pesticides en tonnes"
    )


# ============================================================
# SCHEMA POUR /recommend
# ============================================================

class RecommendationInput(BaseModel):

    Year: int = Field(
        ...,
        description="Année de prédiction"
    )

    Area: str = Field(
        ...,
        description="Pays ou zone géographique"
    )

    average_rain_fall_mm_per_year: float = Field(
        ...,
        description="Précipitations moyennes annuelles en mm"
    )

    avg_temp: float = Field(
        ...,
        description="Température moyenne"
    )

    Pesticide_totale_use_tonnes: float = Field(
        ...,
        description="Utilisation totale de pesticides en tonnes"
    )


# ============================================================
# FONCTION DE PREDICTION
# ============================================================

def predict_yield(
    year,
    area,
    item,
    rainfall,
    temperature,
    pesticides
):

    input_data = pd.DataFrame([{
        "Year": year,
        "Area": area,
        "Item": item,
        "average_rain_fall_mm_per_year": rainfall,
        "avg_temp": temperature,
        "Pesticide_totale_use_tonnes": pesticides
    }])

    # Le pipeline réalise automatiquement :
    # - OneHotEncoder
    # - preprocessing
    # - XGBoost

    prediction_log = model.predict(input_data)

    # Le modèle a été entraîné sur log1p(y)
    prediction = np.expm1(prediction_log[0])

    # Sécurité
    prediction = max(0, float(prediction))

    return prediction


# ============================================================
# ENDPOINT RACINE
# ============================================================

@app.get("/")
def home():

    return {
        "message": "🌱 Crop Yield Prediction API",
        "status": "running",
        "model": "XGBoost",
        "version": "1.0.0",
        "endpoints": [
            "POST /predict",
            "POST /recommend"
        ]
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "model_loaded": model is not None,
        "dataset_loaded": dataset is not None,
        "number_of_crops": len(CROPS)
    }


# ============================================================
# POST /predict
# ============================================================

@app.post("/predict")
def predict(data: PredictionInput):

    # Vérification du modèle
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Le modèle XGBoost n'est pas disponible."
        )

    try:

        # Vérification de la culture
        if data.Item not in CROPS:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Culture inconnue : '{data.Item}'. "
                    f"Les cultures disponibles sont : {CROPS}"
                )
            )

        prediction = predict_yield(
            year=data.Year,
            area=data.Area,
            item=data.Item,
            rainfall=data.average_rain_fall_mm_per_year,
            temperature=data.avg_temp,
            pesticides=data.Pesticide_totale_use_tonnes
        )

        return {
            "success": True,
            "culture": data.Item,
            "zone": data.Area,
            "year": data.Year,
            "predicted_yield_hg_per_ha": round(prediction, 2),
            "predicted_yield_kg_per_ha": round(prediction / 10, 2)
        }

    except HTTPException:
        raise

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la prédiction : {str(e)}"
        )


# ============================================================
# POST /recommend
# ============================================================

@app.post("/recommend")
def recommend(data: RecommendationInput):

    # Vérification du modèle
    if model is None:
        raise HTTPException(
            status_code=500,
            detail="Le modèle XGBoost n'est pas disponible."
        )

    # Vérification du dataset
    if not CROPS:
        raise HTTPException(
            status_code=500,
            detail="La liste des cultures n'est pas disponible."
        )

    try:

        recommendations = []

        # ----------------------------------------------------
        # Tester chaque culture
        # ----------------------------------------------------

        for crop in CROPS:

            prediction = predict_yield(
                year=data.Year,
                area=data.Area,
                item=crop,
                rainfall=data.average_rain_fall_mm_per_year,
                temperature=data.avg_temp,
                pesticides=data.Pesticide_totale_use_tonnes
            )

            recommendations.append({
                "culture": crop,
                "predicted_yield_hg_per_ha": round(
                    prediction, 2
                ),
                "predicted_yield_kg_per_ha": round(
                    prediction / 10, 2
                )
            })

        # ----------------------------------------------------
        # Trier par rendement décroissant
        # ----------------------------------------------------

        recommendations = sorted(
            recommendations,
            key=lambda x: x["predicted_yield_hg_per_ha"],
            reverse=True
        )

        # Ajouter le classement
        for rank, recommendation in enumerate(
            recommendations,
            start=1
        ):
            recommendation["rank"] = rank

        # ----------------------------------------------------
        # Réponse
        # ----------------------------------------------------

        return {
            "success": True,
            "context": {
                "Year": data.Year,
                "Area": data.Area,
                "average_rain_fall_mm_per_year":
                    data.average_rain_fall_mm_per_year,
                "avg_temp": data.avg_temp,
                "Pesticide_totale_use_tonnes":
                    data.Pesticide_totale_use_tonnes
            },
            "number_of_crops": len(recommendations),
            "recommendations": recommendations
        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=f"Erreur lors de la recommandation : {str(e)}"
        )