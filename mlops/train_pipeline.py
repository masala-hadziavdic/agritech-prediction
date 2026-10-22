# ============================================================
# MLOPS PIPELINE — XGBOOST + CROSS-VALIDATION TEMPORELLE
# AGRITECH PREDICTION
# ============================================================

from pathlib import Path
import warnings
import joblib

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from xgboost import XGBRegressor

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from mlflow.models import infer_signature


# ============================================================
# 1. CONFIGURATION
# ============================================================

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
ANNEE_LIMITE = 2008
N_SPLITS = 5

TARGET = "Yield_crops_hg_per_ha"

MLFLOW_EXPERIMENT_NAME = "agritech-prediction"

# Chemins du projet
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "outputs" / "dataset_final.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "model_xgb.pkl"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 2. CONFIGURATION MLFLOW
# ============================================================

mlflow.sklearn.autolog(disable=True)

mlflow.set_tracking_uri(
    f"file:{BASE_DIR / 'mlruns'}"
)

mlflow.set_experiment(
    MLFLOW_EXPERIMENT_NAME
)


# ============================================================
# 3. CHARGEMENT DES DONNÉES
# ============================================================

print("=" * 80)
print("MLOPS PIPELINE — XGBOOST")
print("=" * 80)

print("\nChargement du dataset...")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Dataset introuvable : {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print(f"Dataset chargé : {df.shape}")
print(f"Fichier : {DATA_PATH}")


# ============================================================
# 4. VARIABLES
# ============================================================

num_features = [
    "Year",
    "average_rain_fall_mm_per_year",
    "avg_temp",
    "Pesticide_totale_use_tonnes"
]

cat_features = [
    "Area",
    "Item"
]

features = num_features + cat_features

colonnes_requises = features + [TARGET]


# ============================================================
# 5. VÉRIFICATION DES COLONNES
# ============================================================

colonnes_manquantes = [
    col
    for col in colonnes_requises
    if col not in df.columns
]

if colonnes_manquantes:
    raise ValueError(
        "Colonnes manquantes dans le dataset : "
        f"{colonnes_manquantes}"
    )


# ============================================================
# 6. NETTOYAGE DES DONNÉES
# ============================================================

df_model = df.copy()

# Conversion numérique
for col in num_features + [TARGET]:
    df_model[col] = pd.to_numeric(
        df_model[col],
        errors="coerce"
    )

# Remplacement des infinis
df_model = df_model.replace(
    [np.inf, -np.inf],
    np.nan
)

# Suppression des valeurs manquantes
df_model = df_model.dropna(
    subset=colonnes_requises
).copy()

# Rendement positif ou nul pour log1p
df_model = df_model[
    df_model[TARGET] >= 0
].copy()

# Année entière
df_model["Year"] = df_model["Year"].astype(int)

# Tri chronologique
df_model = df_model.sort_values(
    "Year"
).reset_index(drop=True)

if df_model.empty:
    raise ValueError(
        "Aucune donnée exploitable après nettoyage."
    )

print("\n" + "=" * 80)
print("PRÉPARATION DES DONNÉES")
print("=" * 80)

print(f"Nombre de lignes : {len(df_model)}")
print(
    f"Période : "
    f"{df_model['Year'].min()} - "
    f"{df_model['Year'].max()}"
)

print(
    f"Nombre de pays : "
    f"{df_model['Area'].nunique()}"
)

print(
    f"Nombre de cultures : "
    f"{df_model['Item'].nunique()}"
)


# ============================================================
# 7. SÉPARATION TEMPORELLE TRAIN / TEST
# ============================================================

train_df = df_model[
    df_model["Year"] <= ANNEE_LIMITE
].copy()

test_df = df_model[
    df_model["Year"] > ANNEE_LIMITE
].copy()

if train_df.empty:
    raise ValueError(
        "Le jeu Train est vide."
    )

if test_df.empty:
    raise ValueError(
        "Le jeu Test est vide."
    )

train_df = train_df.reset_index(drop=True)
test_df = test_df.reset_index(drop=True)

X_train = train_df[features].copy()
X_test = test_df[features].copy()

# Transformation logarithmique de la cible
y_train = np.log1p(
    train_df[TARGET].astype(float)
)

y_test = np.log1p(
    test_df[TARGET].astype(float)
)

print("\n" + "=" * 80)
print("SÉPARATION TEMPORELLE")
print("=" * 80)

print(
    f"TRAIN : "
    f"{train_df['Year'].min()} - "
    f"{train_df['Year'].max()}"
)

print(
    f"Nombre de lignes TRAIN : "
    f"{len(X_train)}"
)

print(
    f"TEST : "
    f"{test_df['Year'].min()} - "
    f"{test_df['Year'].max()}"
)

print(
    f"Nombre de lignes TEST : "
    f"{len(X_test)}"
)


# ============================================================
# 8. PRÉTRAITEMENT
# ============================================================

preprocessor_xgb = ColumnTransformer(
    transformers=[
        (
            "num",
            "passthrough",
            num_features
        ),
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore"
            ),
            cat_features
        )
    ]
)


# ============================================================
# 9. MODÈLE XGBOOST FINAL
# ============================================================

params = {
    "n_estimators": 500,
    "learning_rate": 0.05,
    "max_depth": 8,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
    "objective": "reg:squarederror"
}

xgb_pipeline = Pipeline(
    steps=[
        (
            "preprocessing",
            preprocessor_xgb
        ),
        (
            "model",
            XGBRegressor(**params)
        )
    ]
)


# ============================================================
# 10. FONCTIONS D'ÉVALUATION
# ============================================================

def calculer_metriques(y_true, y_pred):
    """
    Calcule RMSE, MAE et R².
    """

    y_true = np.asarray(
        y_true,
        dtype=float
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float
    )

    return {
        "RMSE": float(
            np.sqrt(
                mean_squared_error(
                    y_true,
                    y_pred
                )
            )
        ),
        "MAE": float(
            mean_absolute_error(
                y_true,
                y_pred
            )
        ),
        "R2": float(
            r2_score(
                y_true,
                y_pred
            )
        )
    }


def precision_10_pct(y_true, y_pred):
    """
    Pourcentage des prédictions ayant
    une erreur relative <= 10%.
    """

    y_true = np.asarray(
        y_true,
        dtype=float
    )

    y_pred = np.asarray(
        y_pred,
        dtype=float
    )

    masque = np.abs(y_true) > 1e-8

    if not np.any(masque):
        return float("nan")

    erreurs = (
        np.abs(
            y_true[masque]
            - y_pred[masque]
        )
        / np.abs(y_true[masque])
    )

    return float(
        100 * np.mean(
            erreurs <= 0.10
        )
    )


def evaluer_original(
    y_true_log,
    y_pred_log
):
    """
    Évaluation sur l'échelle originale hg/ha.
    """

    y_true_original = np.expm1(
        np.asarray(
            y_true_log,
            dtype=float
        )
    )

    y_pred_original = np.maximum(
        0,
        np.expm1(
            np.asarray(
                y_pred_log,
                dtype=float
            )
        )
    )

    metriques = calculer_metriques(
        y_true_original,
        y_pred_original
    )

    metriques[
        "Precision_10_pct"
    ] = precision_10_pct(
        y_true_original,
        y_pred_original
    )

    return metriques


# ============================================================
# 11. CONSTRUCTION DES FOLDS TEMPORELS
# ============================================================

annees_train = np.sort(
    train_df["Year"].unique()
)

if len(annees_train) <= N_SPLITS:
    raise ValueError(
        f"Il faut plus de {N_SPLITS} années "
        "distinctes dans le Train."
    )

# Première année = historique initial
blocs_validation = np.array_split(
    annees_train[1:],
    N_SPLITS
)

splits_temporels = []

for bloc in blocs_validation:

    if len(bloc) == 0:
        continue

    premiere_annee_validation = int(
        bloc[0]
    )

    idx_train = np.flatnonzero(
        train_df["Year"].to_numpy()
        < premiere_annee_validation
    )

    idx_val = np.flatnonzero(
        train_df["Year"]
        .isin(bloc)
        .to_numpy()
    )

    if (
        len(idx_train) == 0
        or len(idx_val) == 0
    ):
        continue

    splits_temporels.append(
        (
            idx_train,
            idx_val
        )
    )

if len(splits_temporels) < 2:
    raise ValueError(
        "Pas assez de folds temporels valides."
    )

print("\n" + "=" * 80)
print("CROSS-VALIDATION TEMPORELLE")
print("=" * 80)

for i, (
    idx_train,
    idx_val
) in enumerate(
    splits_temporels,
    start=1
):

    annees_fold_train = (
        train_df.iloc[idx_train]["Year"]
    )

    annees_fold_val = (
        train_df.iloc[idx_val]["Year"]
    )

    print(
        f"Fold {i} | "
        f"Train jusqu'à "
        f"{annees_fold_train.max()} | "
        f"Validation : "
        f"{annees_fold_val.min()}-"
        f"{annees_fold_val.max()} | "
        f"{len(idx_train)} / "
        f"{len(idx_val)} lignes"
    )


# ============================================================
# 12. MLFLOW RUN
# ============================================================

with mlflow.start_run(
    run_name=(
        f"XGBoost_TemporalCV_"
        f"{ANNEE_LIMITE}"
    )
):

    # ========================================================
    # A. CROSS-VALIDATION TEMPORELLE
    # ========================================================

    y_cv_true_liste = []
    y_cv_pred_liste = []

    for i, (
        idx_train,
        idx_val
    ) in enumerate(
        splits_temporels,
        start=1
    ):

        X_fold_train = (
            X_train.iloc[idx_train]
        )

        y_fold_train = (
            y_train.iloc[idx_train]
        )

        X_fold_val = (
            X_train.iloc[idx_val]
        )

        y_fold_val = (
            y_train.iloc[idx_val]
        )

        # Pipeline indépendant
        modele_fold = clone(
            xgb_pipeline
        )

        # Entraînement
        modele_fold.fit(
            X_fold_train,
            y_fold_train
        )

        # Prédiction
        y_fold_pred = (
            modele_fold.predict(
                X_fold_val
            )
        )

        y_cv_true_liste.extend(
            y_fold_val.to_numpy()
        )

        y_cv_pred_liste.extend(
            y_fold_pred
        )

        score_fold = r2_score(
            y_fold_val,
            y_fold_pred
        )

        print(
            f"Fold {i}/"
            f"{len(splits_temporels)} terminé | "
            f"R² log1p : "
            f"{score_fold:.4f}"
        )

    # ========================================================
    # B. MÉTRIQUES CROSS-VALIDATION
    # ========================================================

    y_cv_true = np.asarray(
        y_cv_true_liste,
        dtype=float
    )

    y_cv_pred = np.asarray(
        y_cv_pred_liste,
        dtype=float
    )

    if len(y_cv_true) == 0:
        raise ValueError(
            "Aucune prédiction CV produite."
        )

    cv_log = calculer_metriques(
        y_cv_true,
        y_cv_pred
    )

    cv_original = evaluer_original(
        y_cv_true,
        y_cv_pred
    )

    score_global_cv = cv_log["R2"]


    # ========================================================
    # C. ENTRAÎNEMENT FINAL
    # ========================================================

    print("\n" + "=" * 80)
    print("ENTRAÎNEMENT DU MODÈLE FINAL")
    print("=" * 80)

    xgb_pipeline.fit(
        X_train,
        y_train
    )

    y_pred_train = (
        xgb_pipeline.predict(
            X_train
        )
    )

    y_pred_test = (
        xgb_pipeline.predict(
            X_test
        )
    )


    # ========================================================
    # D. MÉTRIQUES TRAIN / TEST
    # ========================================================

    train_log = calculer_metriques(
        y_train,
        y_pred_train
    )

    test_log = calculer_metriques(
        y_test,
        y_pred_test
    )

    train_original = evaluer_original(
        y_train,
        y_pred_train
    )

    test_original = evaluer_original(
        y_test,
        y_pred_test
    )


    # ========================================================
    # E. AFFICHAGE
    # ========================================================

    print("\n" + "=" * 85)
    print(
        "RÉSULTATS XGBOOST "
        "— CROSS-VALIDATION TEMPORELLE"
    )
    print("=" * 85)

    print(
        f"\nR² global CV log1p : "
        f"{score_global_cv:.4f}"
    )

    print(
        f"R² Test log1p : "
        f"{test_log['R2']:.4f}"
    )

    print(
        f"RMSE Test hg/ha : "
        f"{test_original['RMSE']:.2f}"
    )

    print(
        f"MAE Test hg/ha : "
        f"{test_original['MAE']:.2f}"
    )

    print(
        f"Précision Test ±10 % : "
        f"{test_original['Precision_10_pct']:.2f}%"
    )


    # ========================================================
    # F. PARAMÈTRES MLFLOW
    # ========================================================

    mlflow.log_params({
        **params,

        "model": "XGBRegressor",

        "annee_limite":
            ANNEE_LIMITE,

        "train_annee_min":
            int(
                train_df["Year"].min()
            ),

        "train_annee_max":
            int(
                train_df["Year"].max()
            ),

        "test_annee_min":
            int(
                test_df["Year"].min()
            ),

        "test_annee_max":
            int(
                test_df["Year"].max()
            ),

        "dataset_size":
            len(df_model),

        "train_size":
            len(X_train),

        "test_size":
            len(X_test),

        "target_transform":
            "log1p",

        "categorical_encoding":
            "OneHotEncoder",

        "cv_strategy":
            "Chronological year-based folds",

        "cv_n_splits_effectifs":
            len(splits_temporels)
    })


    # ========================================================
    # G. MÉTRIQUES MLFLOW
    # ========================================================

    mlflow.log_metric(
        "global_r2_cv_log",
        score_global_cv
    )

    # Métriques log1p
    for prefix, metrics in [
        ("cv", cv_log),
        ("train", train_log),
        ("test", test_log)
    ]:

        for nom, valeur in metrics.items():

            if np.isfinite(valeur):

                mlflow.log_metric(
                    f"{prefix}_{nom.lower()}_log",
                    valeur
                )

    # Métriques originales
    for prefix, metrics in [
        ("cv", cv_original),
        ("train", train_original),
        ("test", test_original)
    ]:

        for nom, valeur in metrics.items():

            if np.isfinite(valeur):

                mlflow.log_metric(
                    f"{prefix}_{nom.lower()}_original",
                    valeur
                )


    # ========================================================
    # H. SIGNATURE MLFLOW
    # ========================================================

    signature = infer_signature(
        X_train,
        xgb_pipeline.predict(
            X_train
        )
    )


    # ========================================================
    # I. ENREGISTREMENT MLFLOW
    # ========================================================

    mlflow.sklearn.log_model(
        sk_model=xgb_pipeline,
        artifact_path=(
            "XGBoost_TemporalCV_model"
        ),
        signature=signature
    )


    # ========================================================
    # J. SAUVEGARDE DU MODÈLE POUR LE DÉPLOIEMENT
    # ========================================================

    joblib.dump(
        xgb_pipeline,
        MODEL_PATH
    )

    print("\n" + "=" * 80)
    print("MODÈLE SAUVEGARDÉ")
    print("=" * 80)

    print(
        f"Fichier : {MODEL_PATH}"
    )


# ============================================================
# 13. FIN
# ============================================================

print("\n" + "=" * 80)
print("PIPELINE MLOPS TERMINÉ AVEC SUCCÈS")
print("=" * 80)

print(
    f"MLflow experiment : "
    f"{MLFLOW_EXPERIMENT_NAME}"
)

print(
    f"Modèle : XGBoost"
)

print(
    f"Train : "
    f"{train_df['Year'].min()}-"
    f"{train_df['Year'].max()}"
)

print(
    f"Test : "
    f"{test_df['Year'].min()}-"
    f"{test_df['Year'].max()}"
)

print(
    f"R² CV log1p : "
    f"{score_global_cv:.4f}"
)

print(
    f"R² Test log1p : "
    f"{test_log['R2']:.4f}"
)

print(
    f"RMSE Test : "
    f"{test_original['RMSE']:.2f}"
)

print(
    f"MAE Test : "
    f"{test_original['MAE']:.2f}"
)

print(
    f"Précision ±10 % : "
    f"{test_original['Precision_10_pct']:.2f}%"
)

print(
    f"\nModèle prêt pour le déploiement : "
    f"{MODEL_PATH}"
)