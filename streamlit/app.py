import os

import pandas as pd
import requests
import streamlit as st

# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Crop Yield Prediction",
    page_icon="🌱",
    layout="wide"
)

# Adresse de l'API FastAPI
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

PREDICT_URL = f"{API_URL}/predict"
RECOMMEND_URL = f"{API_URL}/recommend"


# ============================================================
# STYLE ET TITRE
# ============================================================

st.title("🌱 Crop Yield Prediction")
st.markdown(
    """
    ### Système de prédiction et de recommandation des cultures

    Ajustez les paramètres climatiques et agricoles pour obtenir
    une estimation du rendement ou comparer les cultures possibles.
    """
)

st.divider()


# ============================================================
# FONCTIONS D'APPEL À L'API
# ============================================================

def call_api(url, payload):
    """Envoie les données à FastAPI et retourne la réponse JSON."""

    try:
        response = requests.post(
            url,
            json=payload,
            timeout=120
        )

        if not response.ok:
            try:
                detail = response.json().get(
                    "detail",
                    response.text
                )
            except ValueError:
                detail = response.text

            st.error(
                f"Erreur API ({response.status_code}) : {detail}"
            )
            return None

        return response.json()

    except requests.exceptions.ConnectionError:
        st.error(
            "Impossible de contacter FastAPI. "
            "Vérifiez que l'API fonctionne sur "
            f"{API_URL}."
        )
        return None

    except requests.exceptions.Timeout:
        st.error(
            "La requête a pris trop de temps. "
            "Réessayez dans quelques instants."
        )
        return None

    except requests.exceptions.RequestException as error:
        st.error(f"Erreur de communication : {error}")
        return None

    except ValueError:
        st.error("L'API a retourné une réponse JSON invalide.")
        return None


# ============================================================
# CONTEXTE AGRICOLE COMMUN
# ============================================================

st.subheader("📍 Contexte de production")

col1, col2 = st.columns(2)

with col1:
    area = st.text_input(
        "Pays ou zone géographique",
        value="France",
        help="Utilisez le nom d'une zone connue du modèle."
    )

    year = st.slider(
        "Année",
        min_value=1990,
        max_value=2030,
        value=2013,
        step=1
    )

    rainfall = st.slider(
        "Précipitations moyennes annuelles (mm)",
        min_value=0,
        max_value=5000,
        value=1200,
        step=50
    )

with col2:
    temperature = st.slider(
        "Température moyenne (°C)",
        min_value=-10.0,
        max_value=45.0,
        value=20.0,
        step=0.5
    )

    pesticides = st.number_input(
        "Utilisation totale de pesticides (tonnes)",
        min_value=0.0,
        max_value=10000000.0,
        value=2500.0,
        step=100.0
    )


# ============================================================
# CHOIX DU MODE
# ============================================================

st.divider()

mode = st.radio(
    "Que souhaitez-vous faire ?",
    ["🔎 Prédiction", "🌾 Recommandation"],
    horizontal=True
)


# ============================================================
# MODE PRÉDICTION
# ============================================================

if mode == "🔎 Prédiction":

    st.subheader("🔎 Prédire le rendement d'une culture")

    crop = st.text_input(
        "Nom de la culture",
        value="Maize",
        help="Exemple : Maize. Respectez le nom utilisé dans le dataset."
    )

    st.caption(
        "Le nom du pays et celui de la culture doivent correspondre "
        "aux catégories connues du modèle."
    )

    if st.button(
        "Calculer le rendement",
        type="primary",
        use_container_width=True
    ):

        if not area.strip() or not crop.strip():
            st.warning(
                "Veuillez renseigner le pays et la culture."
            )

        else:
            payload = {
                "Year": year,
                "Area": area.strip(),
                "Item": crop.strip(),
                "average_rain_fall_mm_per_year": rainfall,
                "avg_temp": temperature,
                "Pesticide_totale_use_tonnes": pesticides
            }

            with st.spinner("Prédiction en cours..."):
                result = call_api(PREDICT_URL, payload)

            if result is not None:

                st.success("Prédiction terminée ! 🎉")

                value_hg = result.get(
                    "predicted_yield_hg_per_ha"
                )
                value_kg = result.get(
                    "predicted_yield_kg_per_ha"
                )

                if value_hg is not None:
                    st.metric(
                        label=f"Rendement prédit — {crop}",
                        value=f"{value_hg:,.2f} hg/ha"
                    )

                if value_kg is not None:
                    st.metric(
                        label="Équivalent en kg/ha",
                        value=f"{value_kg:,.2f} kg/ha"
                    )

                st.info(
                    "Il s'agit d'une estimation produite par le modèle, "
                    "et non d'une garantie du rendement réel."
                )

                with st.expander("Voir les données envoyées et la réponse"):
                    st.json({
                        "request": payload,
                        "response": result
                    })


# ============================================================
# MODE RECOMMANDATION
# ============================================================

else:

    st.subheader("🌾 Recommander les cultures")

    st.write(
        "L'API va prédire le rendement de toutes les cultures "
        "disponibles et les classer du rendement le plus élevé "
        "au plus faible."
    )

    if st.button(
        "Recommander les cultures",
        type="primary",
        use_container_width=True
    ):

        if not area.strip():
            st.warning("Veuillez renseigner le pays ou la zone.")

        else:
            payload = {
                "Year": year,
                "Area": area.strip(),
                "average_rain_fall_mm_per_year": rainfall,
                "avg_temp": temperature,
                "Pesticide_totale_use_tonnes": pesticides
            }

            with st.spinner(
                "Analyse des cultures en cours..."
            ):
                result = call_api(RECOMMEND_URL, payload)

            if result is not None:

                recommendations = result.get(
                    "recommendations",
                    []
                )

                if not recommendations:
                    st.warning(
                        "L'API n'a retourné aucune recommandation."
                    )

                else:
                    df_results = pd.DataFrame(recommendations)

                    required_columns = {
                        "culture",
                        "predicted_yield_kg_per_ha"
                    }

                    if not required_columns.issubset(
                        df_results.columns
                    ):
                        st.error(
                            "La réponse de l'API ne contient pas "
                            "les champs attendus."
                        )

                    else:
                        # Trier par rendement décroissant
                        df_results = df_results.sort_values(
                            "predicted_yield_kg_per_ha",
                            ascending=False
                        ).reset_index(drop=True)

                        st.success(
                            f"{len(df_results)} cultures évaluées ! 🌱"
                        )

                        # Meilleure culture selon le rendement prédit
                        best = df_results.iloc[0]

                        st.metric(
                            label="Rendement prédit le plus élevé",
                            value=(
                                f"{best['predicted_yield_kg_per_ha']:,.2f} "
                                "kg/ha"
                            ),
                            delta=f"Culture : {best['culture']}"
                        )

                        st.divider()

                        # Graphique à barres
                        st.subheader(
                            "📊 Comparaison des rendements prédits"
                        )

                        chart_data = df_results.set_index(
                            "culture"
                        )[["predicted_yield_kg_per_ha"]]

                        st.bar_chart(
                            chart_data,
                            horizontal=True
                        )

                        # Tableau de classement
                        st.subheader(
                            "🏆 Classement complet des cultures"
                        )

                        display_columns = [
                            col for col in [
                                "rank",
                                "culture",
                                "predicted_yield_hg_per_ha",
                                "predicted_yield_kg_per_ha"
                            ]
                            if col in df_results.columns
                        ]

                        st.dataframe(
                            df_results[display_columns],
                            use_container_width=True,
                            hide_index=True
                        )

                        # Export CSV
                        csv_data = df_results.to_csv(
                            index=False
                        ).encode("utf-8-sig")

                        st.download_button(
                            label="📥 Télécharger les recommandations (CSV)",
                            data=csv_data,
                            file_name="crop_recommendations.csv",
                            mime="text/csv",
                            use_container_width=True
                        )

                        st.caption(
                            "Le classement reflète uniquement les "
                            "rendements prédits par le modèle. Il ne "
                            "tient pas compte des coûts, des prix de "
                            "vente, de la disponibilité en eau ni "
                            "de la rentabilité économique."
                        )


# ============================================================
# PIED DE PAGE
# ============================================================

st.divider()

st.caption(
    "Crop Yield Prediction | Interface Streamlit connectée à FastAPI "
    "| Modèle XGBoost"
)
