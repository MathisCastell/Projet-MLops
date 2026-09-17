import os

import requests
import streamlit as st

from src.config import Config

cfg = Config.from_yaml("configs/config.yaml")
default_api_url = os.environ.get("API_URL", cfg.app.api_url)

CATEGORY_OPTIONS = {
    "gender": ["Female", "Male"],
    "SeniorCitizen": [0, 1],
    "Partner": ["Yes", "No"],
    "Dependents": ["Yes", "No"],
    "PhoneService": ["Yes", "No"],
    "MultipleLines": ["No", "Yes", "No phone service"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "OnlineSecurity": ["No", "Yes", "No internet service"],
    "OnlineBackup": ["No", "Yes", "No internet service"],
    "DeviceProtection": ["No", "Yes", "No internet service"],
    "TechSupport": ["No", "Yes", "No internet service"],
    "StreamingTV": ["No", "Yes", "No internet service"],
    "StreamingMovies": ["No", "Yes", "No internet service"],
    "Contract": ["Month-to-month", "One year", "Two year"],
    "PaperlessBilling": ["Yes", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
}

st.set_page_config(page_title="Telco Churn Predictor", page_icon="📉")
st.title("📉 Prédiction de churn client")
st.caption("Interface Streamlit qui interroge l'API FastAPI (modèle chargé depuis le MLflow Model Registry)")

api_url = st.sidebar.text_input("URL de l'API", value=default_api_url)

payload = {}
col1, col2 = st.columns(2)
with col1:
    payload["tenure"] = st.number_input("Ancienneté (mois)", min_value=0, max_value=100, value=12)
    payload["MonthlyCharges"] = st.number_input("Charges mensuelles", min_value=0.0, value=70.0)
    payload["TotalCharges"] = st.number_input("Charges totales", min_value=0.0, value=840.0)

with col2:
    for feature in list(CATEGORY_OPTIONS)[:3]:
        payload[feature] = st.selectbox(feature, CATEGORY_OPTIONS[feature])

with st.expander("Autres caractéristiques du contrat"):
    for feature in list(CATEGORY_OPTIONS)[3:]:
        payload[feature] = st.selectbox(feature, CATEGORY_OPTIONS[feature])

if st.button("Prédire", type="primary"):
    with st.spinner("Prédiction en cours..."):
        try:
            response = requests.post(f"{api_url}/predict", json=payload, timeout=10)
            response.raise_for_status()
            result = response.json()
        except requests.RequestException as exc:
            st.error(f"Impossible de contacter l'API ({api_url}) : {exc}")
        else:
            proba = result["churn_probability"]
            if result["churn"]:
                st.error(f"Risque de churn élevé ({proba:.1%})")
            else:
                st.success(f"Client fidèle ({1 - proba:.1%} de rétention estimée)")
            st.progress(proba)
