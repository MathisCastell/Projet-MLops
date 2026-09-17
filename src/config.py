from pathlib import Path

import yaml


class Config:
    """Wraps a nested dict so it can be accessed as attributes (cfg.model.type)."""

    def __init__(self, data: dict):
        for key, value in data.items():
            setattr(self, key, Config(value) if isinstance(value, dict) else value)

    def __repr__(self) -> str:
        return f"Config({self.__dict__!r})"

    def get(self, key, default=None):
        return getattr(self, key, default)

    @classmethod
    def from_yaml(cls, path: str = "configs/config.yaml") -> "Config":
        with open(Path(path), "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        return cls(raw)
