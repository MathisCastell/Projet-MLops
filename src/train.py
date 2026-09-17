import argparse

import mlflow
import mlflow.sklearn
import pandas as pd
from mlflow import MlflowClient
from sklearn.model_selection import GridSearchCV

from src.config import Config
from src.pipeline import build_pipeline
from src.utils import set_seed


def main(config_path: str) -> None:
    cfg = Config.from_yaml(config_path)
    set_seed(cfg.data.seed)

    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)
    mlflow.sklearn.autolog(log_input_examples=True, log_model_signatures=True)

    train_df = pd.read_csv(cfg.data.train_path)
    X = train_df.drop(columns=[cfg.data.target])
    y = train_df[cfg.data.target]

    model_type = cfg.model.type
    pipeline = build_pipeline(cfg.features.numeric, cfg.features.categorical, model_type)
    param_grid = getattr(cfg.training.param_grid, model_type)

    search = GridSearchCV(
        pipeline,
        param_grid=vars(param_grid),
        cv=cfg.training.cv_folds,
        scoring=cfg.training.scoring,
        n_jobs=-1,
    )

    with mlflow.start_run(run_name=f"{model_type}-gridsearch"):
        search.fit(X, y)
        mlflow.log_param("model_type", model_type)
        mlflow.log_metric("best_cv_score", search.best_score_)
        print(f"Best params: {search.best_params_}")
        print(f"Best CV {cfg.training.scoring}: {search.best_score_:.4f}")

        # RandomForest's tree storage isn't skops-trusted by default; we trained this
        # model ourselves in this run, so it's safe to explicitly trust it.
        model_info = mlflow.sklearn.log_model(
            search.best_estimator_,
            name="best_estimator",
            skops_trusted_types=["sklearn.tree._tree.Tree", "numpy.dtype"],
        )
        registered = mlflow.register_model(model_info.model_uri, cfg.model.registry_name)

    client = MlflowClient()
    client.set_registered_model_alias(
        name=cfg.model.registry_name,
        alias=cfg.model.alias,
        version=registered.version,
    )
    print(f"Registered {cfg.model.registry_name} v{registered.version} -> alias @{cfg.model.alias}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
