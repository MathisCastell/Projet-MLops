import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI

from src.config import Config

cfg = Config.from_yaml("configs/config.yaml")
mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)

model_uri = f"models:/{cfg.model.registry_name}@{cfg.model.alias}"
model = mlflow.sklearn.load_model(model_uri)

app = FastAPI(title="Telco Churn Prediction API")


@app.get("/health")
def health():
    return {"status": "ok", "model": model_uri}


@app.post("/predict")
def predict(payload: dict):
    df = pd.DataFrame([payload])
    prediction = int(model.predict(df)[0])
    probability = float(model.predict_proba(df)[0, 1])
    return {"churn": bool(prediction), "churn_probability": probability}
