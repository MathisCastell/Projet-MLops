# Telco Customer Churn — Pipeline MLOps

Pipeline MLOps de bout en bout pour prédire le churn client sur le dataset Telco Customer Churn.
Le suivi des expériences et la gouvernance des modèles sont gérés avec **MLflow** (tracking, tuning
d'hyperparamètres, model registry), le service du modèle avec **FastAPI**, et une interface
**Streamlit** au-dessus de l'API. Des tests automatisés et une CI GitHub Actions sont inclus.

## Architecture

```
Projet/
├── configs/
│   └── config.yaml         # dataset, features, modèle, hyperparamètres, MLflow, API
├── src/
│   ├── config.py           # classe Config (chargement du YAML)
│   ├── get_data.py         # télécharge le CSV source -> data/raw.csv
│   ├── preprocess.py       # nettoyage, encodage de la cible, split train/test
│   ├── pipeline.py         # ColumnTransformer + modèle (sklearn Pipeline)
│   ├── train.py            # GridSearchCV + MLflow autolog + model registry
│   ├── evaluate.py         # évaluation sur le test set, ROC/PR/matrice de confusion loggés dans MLflow
│   ├── predict.py          # inférence batch depuis le modèle enregistré
│   ├── api.py              # API FastAPI qui sert le modèle depuis le registry MLflow
│   └── utils.py            # seed, helpers de tracé
├── app/
│   └── streamlit_app.py    # interface Streamlit qui appelle l'API
├── tests/
│   └── test_pipeline.py    # tests pytest (config, pipeline, préprocessing)
├── .github/workflows/ci.yml # lint (ruff) + tests à chaque push/PR
├── data/                   # généré (gitignoré)
├── reports/                # généré (gitignoré) : plots + métriques d'évaluation
├── mlruns/, mlflow.db      # généré (gitignoré) : tracking store MLflow local
├── Makefile
├── Dockerfile
├── requirements.txt
├── .env.example
└── README.md
```

Étapes du pipeline : `get-data → preprocess → train (tracking + tuning + registry MLflow) → evaluate → run_api → run_app`

## Tracking et registry MLflow

- **Tracking store** : `sqlite:///mlflow.db` (base SQLite locale, aucun serveur externe requis).
- **Artifact store** : `mlruns/` (modèles sérialisés, plots, résultats de cross-validation).
- **Autologging** (`mlflow.sklearn.autolog()`) : hyperparamètres testés par `GridSearchCV`,
  meilleur score de CV, le modèle entraîné, un exemple d'entrée et sa signature.
- **Model registry** : chaque entraînement enregistre une nouvelle version de
  `TelcoChurnClassifier` et lui assigne l'alias `production` (API actuelle du registry MLflow —
  l'ancienne API par "stages" est dépréciée). L'API et `predict.py` chargent toujours
  `models:/TelcoChurnClassifier@production`, jamais un fichier local codé en dur.
- **Run d'évaluation** : les métriques du test set (`accuracy`, `precision`, `recall`, `f1`,
  `roc_auc`) ainsi que les courbes ROC/PR et la matrice de confusion sont loggées comme un run
  MLflow distinct.

Pour visualiser tout ça :

```bash
make mlflow-ui
```

puis ouvrir http://localhost:5000.

## Installation

```bash
python -m pip install -U pip -r requirements.txt
```

## Exécution du pipeline

```bash
python -m src.get_data --config configs/config.yaml
python -m src.preprocess --config configs/config.yaml
python -m src.train --config configs/config.yaml
python -m src.evaluate --config configs/config.yaml
```

ou :

```bash
make all
```

Pour changer de modèle ou de grille d'hyperparamètres, modifier `configs/config.yaml`
(`model.type: logreg` ou `random_forest`, `training.param_grid`).

## Tests et lint

```bash
python -m pytest -q
ruff check src tests app
```

ou `make test` / `make lint`.

## API

```bash
python -m uvicorn src.api:app --host 0.0.0.0 --port 8000
```

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"tenure": 12, "MonthlyCharges": 70, "TotalCharges": 840, "gender": "Female", "SeniorCitizen": 0, "Partner": "Yes", "Dependents": "No", "PhoneService": "Yes", "MultipleLines": "No", "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No", "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes", "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes", "PaymentMethod": "Electronic check"}'
```

## Interface Streamlit

```bash
python -m streamlit run app/streamlit_app.py --server.port 8501
```

L'API doit être lancée en parallèle : Streamlit se contente d'appeler `/predict`, il ne charge pas
le modèle lui-même.

## Docker

```bash
make build_docker
make run_docker
```

L'image embarque les `mlruns/` et `mlflow.db` générés localement : `make train` doit donc avoir été
exécuté au moins une fois avant `docker build` pour que le modèle enregistré existe.

## Dépannage

- Sous Windows, si `uvicorn`/`streamlit`/`ruff`/`mlflow` ne sont pas reconnus en commande directe,
  utiliser `python -m uvicorn ...`, `python -m streamlit ...`, etc.
- La sauvegarde du modèle peut occasionnellement échouer sous Windows avec une erreur "Accès
  refusé" (généralement l'antivirus qui scanne le fichier juste après son écriture) : relancer la
  commande résout le problème.
