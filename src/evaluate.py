import argparse
import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.config import Config
from src.utils import save_confusion_matrix, save_pr_curve, save_roc_curve


def main(config_path: str) -> None:
    cfg = Config.from_yaml(config_path)

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    model_uri = f"models:/{cfg.model.registry_name}@{cfg.model.alias}"
    model = mlflow.sklearn.load_model(model_uri)

    test_df = pd.read_csv(cfg.data.test_path)
    X_test = test_df.drop(columns=[cfg.data.target])
    y_test = test_df[cfg.data.target]

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    print("Test set evaluation:")
    for key, value in metrics.items():
        print(f"  {key}: {value:.4f}")

    Path("reports").mkdir(exist_ok=True)
    save_confusion_matrix(y_test, y_pred, "reports/confusion_matrix.png")
    save_roc_curve(y_test, y_proba, "reports/roc_curve.png")
    save_pr_curve(y_test, y_proba, "reports/pr_curve.png")

    with mlflow.start_run(run_name="evaluation"):
        mlflow.log_metrics(metrics)
        mlflow.log_param("evaluated_model_uri", model_uri)
        mlflow.log_artifact("reports/confusion_matrix.png")
        mlflow.log_artifact("reports/roc_curve.png")
        mlflow.log_artifact("reports/pr_curve.png")

    Path("reports/test_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print("Metrics and plots logged to MLflow and saved under reports/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
