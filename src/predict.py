import argparse

import mlflow.sklearn
import pandas as pd

from src.config import Config


def main(config_path: str, input_csv: str, output_csv: str) -> None:
    cfg = Config.from_yaml(config_path)
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)

    model_uri = f"models:/{cfg.model.registry_name}@{cfg.model.alias}"
    model = mlflow.sklearn.load_model(model_uri)

    df = pd.read_csv(input_csv)
    features = df.drop(columns=[cfg.data.target], errors="ignore")

    df["prediction"] = model.predict(features)
    df["churn_probability"] = model.predict_proba(features)[:, 1]
    df.to_csv(output_csv, index=False)
    print(f"Wrote {len(df)} predictions to {output_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--input", required=True, help="CSV file with feature columns")
    parser.add_argument("--output", required=True, help="Where to write predictions")
    args = parser.parse_args()
    main(args.config, args.input, args.output)
