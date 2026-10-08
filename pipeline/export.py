"""Écrit le modèle et ses métadonnées là où l'API les lit (`model/`)."""

import json
import pickle
from pathlib import Path

from sklearn.pipeline import Pipeline

from pipeline import config
from pipeline.train import TrainedModel


def save_for_api(
    model: TrainedModel,
    version: int,
    metrics: dict[str, float],
    model_path: Path = config.MODEL_PATH,
    meta_path: Path = config.MODEL_META_PATH,
) -> None:
    """Le seuil voyage avec le modèle : un modèle réentraîné a son propre seuil
    optimal, et l'API doit appliquer celui-là, pas un 0,47 écrit en dur."""
    with open(model_path, "wb") as f:
        pickle.dump(model.pipeline, f)
    meta = {
        "model_version": version,
        "threshold": model.threshold,
        "training_rows": model.training_rows,
        "metrics": {name: round(value, 4) for name, value in metrics.items()},
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n")


def load_served(
    model_path: Path = config.MODEL_PATH, meta_path: Path = config.MODEL_META_PATH
) -> tuple[Pipeline, dict]:
    """Le modèle actuellement servi par l'API et ses métadonnées."""
    with open(model_path, "rb") as f:
        pipeline = pickle.load(f)
    return pipeline, json.loads(meta_path.read_text())
