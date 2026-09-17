import pandas as pd

from src.config import Config
from src.pipeline import build_pipeline
from src.preprocess import clean


def test_config_loads_expected_sections():
    cfg = Config.from_yaml("configs/config.yaml")
    assert cfg.data.target == "Churn"
    assert cfg.model.type in ("logreg", "random_forest")


def test_build_pipeline_has_expected_steps():
    pipe = build_pipeline(["a"], ["b"], "logreg")
    assert list(pipe.named_steps) == ["pre", "model"]


def test_build_pipeline_random_forest():
    pipe = build_pipeline(["a"], ["b"], "random_forest")
    assert "model" in dict(pipe.named_steps)


def test_build_pipeline_rejects_unknown_model_type():
    try:
        build_pipeline(["a"], ["b"], "unknown")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_clean_encodes_target_and_parses_total_charges():
    df = pd.DataFrame(
        {
            "customerID": ["1", "2"],
            "TotalCharges": ["29.85", " "],
            "Churn": ["Yes", "No"],
        }
    )
    cleaned = clean(df, target="Churn")
    assert "customerID" not in cleaned.columns
    assert cleaned["Churn"].tolist() == [1, 0]
    assert cleaned["TotalCharges"].iloc[0] == 29.85
    assert pd.isna(cleaned["TotalCharges"].iloc[1])
