import json
from dataclasses import dataclass
from typing import Any, Dict, List

import joblib

from app.core.config import MODEL_PATH, METRICS_PATH


@dataclass
class ModelArtifacts:
    pipeline: Any
    threshold: float
    metrics: Dict[str, Any]


def load_artifacts() -> ModelArtifacts:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}. "
            "Run training first: python -m turnover_ml.train"
        )

    pipeline = joblib.load(MODEL_PATH)

    threshold = 0.5
    metrics: Dict[str, Any] = {}

    if METRICS_PATH.exists():
        metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        threshold = float(metrics.get("best_threshold_max_f1", 0.5))

    return ModelArtifacts(pipeline=pipeline, threshold=threshold, metrics=metrics)


def get_required_features(artifacts: ModelArtifacts) -> List[str]:
    """
    Retourne la liste des features attendues par le pipeline (en entrée).
    """
    pipeline = artifacts.pipeline

    if hasattr(pipeline, "feature_names_in_"):
        return list(pipeline.feature_names_in_)

    raise ValueError(
        "Pipeline does not expose feature_names_in_. "
        "Ensure it was trained with a pandas DataFrame."
    )