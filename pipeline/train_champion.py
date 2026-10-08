"""Entraîne le champion v1 sur la seule référence, l'enregistre et le met en service.

Le modèle de la V1 avait vu toutes les données étiquetées, lots compris : il ne
pouvait pas servir de point de départ honnête à la simulation.

Usage : python -m pipeline.train_champion
"""

import pandas as pd

from pipeline import config, registry
from pipeline.export import save_for_api
from pipeline.train import evaluate, train_model


def main() -> None:
    reference = pd.read_parquet(config.REFERENCE_PATH)
    frozen_test = pd.read_parquet(config.FROZEN_TEST_PATH)

    model = train_model(reference)
    test_metrics = evaluate(model, frozen_test)
    metrics = {
        "frozen_test_auc": test_metrics["auc"],
        "frozen_test_cost_per_client": test_metrics["cost_per_client"],
    }
    version = registry.register(
        model, metrics, tags={"trained_on": "reference"}, alias="champion"
    )
    save_for_api(model, version, metrics)

    print(f"Champion v{version} : seuil {model.threshold}, AUC OOF {model.oof_auc:.4f}")
    print(f"Test figé : {metrics}")


if __name__ == "__main__":
    main()
