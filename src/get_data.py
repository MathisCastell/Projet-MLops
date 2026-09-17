import argparse
from pathlib import Path

import requests

from src.config import Config


def main(config_path: str) -> None:
    cfg = Config.from_yaml(config_path)

    raw_path = Path(cfg.data.raw_path)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Downloading dataset from {cfg.data.source_url}")
    response = requests.get(cfg.data.source_url, timeout=30)
    response.raise_for_status()

    raw_path.write_bytes(response.content)
    print(f"Saved raw data to {raw_path} ({len(response.content)} bytes)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    main(args.config)
