"""Connexion au MLflow hébergé par DagsHub : suivi des entraînements et Model Registry.

Le modèle en production porte l'alias `champion`, le candidat l'alias `challenger`.
Identifiants lus dans l'environnement (secrets GitHub en CI, fichier .env en local).
"""

import json
import os

import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
from mlflow import MlflowClient

from pipeline import config
from pipeline.train import TrainedModel

DAGSHUB_REPO = "credit-scoring-mlops"
EXPERIMENT = "credit-scoring-v2"
REGISTERED_MODEL = "credit-scoring-lgbm"


def connect() -> MlflowClient:
    """Configure MLflow vers DagsHub. Échoue clairement si un identifiant manque."""
    load_dotenv(config.ROOT / ".env")
    missing = [n for n in ("DAGSHUB_USER", "DAGSHUB_TOKEN") if not os.environ.get(n)]
    if missing:
        raise RuntimeError(
            f"Identifiants DagsHub manquants : {', '.join(missing)} "
            "(fichier .env en local, secrets GitHub en CI)."
        )
    user = os.environ["DAGSHUB_USER"]
    os.environ["MLFLOW_TRACKING_USERNAME"] = user
    os.environ["MLFLOW_TRACKING_PASSWORD"] = os.environ["DAGSHUB_TOKEN"]
    mlflow.set_tracking_uri(f"https://dagshub.com/{user}/{DAGSHUB_REPO}.mlflow")
    mlflow.set_experiment(EXPERIMENT)
    return MlflowClient()


def register(
    model: TrainedModel, metrics: dict[str, float], tags: dict[str, str], alias: str
) -> int:
    """Journalise l'entraînement, enregistre le modèle et lui pose `alias`.
    Renvoie le numéro de version dans le Registry."""
    client = connect()
    with mlflow.start_run(run_name=f"{alias}-{tags.get('trained_on', '')}"):
        mlflow.log_params(config.CHAMPION_PARAMS)
        mlflow.log_metrics(
            {"threshold": model.threshold, "oof_auc": model.oof_auc, **metrics}
        )
        mlflow.set_tags({**tags, "training_rows": str(model.training_rows)})
        info = mlflow.sklearn.log_model(
            model.pipeline,
            artifact_path="model",
            registered_model_name=REGISTERED_MODEL,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
        )
    version = int(info.registered_model_version)
    client.set_model_version_tag(
        REGISTERED_MODEL, str(version), "threshold", str(model.threshold)
    )
    client.set_registered_model_alias(REGISTERED_MODEL, alias, str(version))
    return version


def promote_served() -> int:
    """Donne l'alias `champion` à la version que l'API sert (lue dans
    model_meta.json). Appelé par le CI/CD après chaque déploiement."""
    client = connect()
    version = json.loads(config.MODEL_META_PATH.read_text())["model_version"]
    client.set_registered_model_alias(REGISTERED_MODEL, "champion", str(version))
    return version


if __name__ == "__main__":
    print(f"Alias champion → version {promote_served()}")
