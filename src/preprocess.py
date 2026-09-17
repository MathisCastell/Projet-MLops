import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import Config


def clean(df: pd.DataFrame, target: str) -> pd.DataFrame:
    df = df.drop(columns=["customerID"], errors="ignore")
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df[target] = df[target].map({"Yes": 1, "No": 0})
    return df


def main(config_path: str) -> None:
    cfg = Config.from_yaml(config_path)

    raw = pd.read_csv(cfg.data.raw_path)
    raw = clean(raw, cfg.data.target)

    train_df, test_df = train_test_split(
        raw,
        test_size=cfg.data.test_size,
        random_state=cfg.data.seed,
        stratify=raw[cfg.data.target],
    )

    Path(cfg.data.train_path).parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(cfg.data.train_path, index=False)
    test_df.to_csv(cfg.data.test_path, index=False)

    print(f"Train: {len(train_df)} rows -> {cfg.data.train_path}")
    print(f"Test:  {len(test_df)} rows -> {cfg.data.test_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
