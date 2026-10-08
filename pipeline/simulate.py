"""Fabrique le lot reçu un mois donné, avec la dérive simulée prévue pour ce mois.

Les mois 1 et 2 sont reçus tels quels. Le mois 3 porte une dérive des données, le
mois 4 une dérive de concept (paramètres dans pipeline/config.py).

Usage : python -m pipeline.simulate --month 3
"""

import argparse

import numpy as np
import pandas as pd

from pipeline import config


def inject_data_drift(lot: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Une campagne attire une clientèle plus jeune, pour des montants plus élevés.
    On garde tous les moins de 40 ans et seulement une partie des autres : chaque
    demande reste une vraie demande, seule la population change."""
    age = -lot["DAYS_BIRTH"] / 365.25
    keep_proba = np.where(
        age < config.DRIFT_YOUNG_AGE, 1.0, config.DRIFT_OLDER_KEEP_SHARE
    )
    drifted = lot[rng.random(len(lot)) < keep_proba].copy()
    for col in config.DRIFT_AMOUNT_COLUMNS:
        drifted[col] = drifted[col] * config.DRIFT_AMOUNT_FACTOR
    return drifted


def inject_concept_drift(lot: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Un choc économique fait défaillir une partie des salariés aux revenus modestes.
    Les variables ne bougent pas : seule la relation entre elles et le défaut change,
    ce qu'aucun test de dérive des données ne peut voir."""
    segment = (lot["NAME_INCOME_TYPE"] == "Working") & (
        lot["AMT_INCOME_TOTAL"] < lot["AMT_INCOME_TOTAL"].median()
    )
    newly_defaulting = (
        segment
        & (lot["TARGET"] == 0)
        & (rng.random(len(lot)) < config.CONCEPT_DRIFT_DEFAULT_RATE)
    )
    drifted = lot.copy()
    drifted.loc[newly_defaulting, "TARGET"] = 1
    return drifted


def receive_month(month: int) -> pd.DataFrame:
    lot = pd.read_parquet(config.batch_path(month))
    rng = np.random.default_rng(config.SEED + month)
    if month == config.DATA_DRIFT_MONTH:
        return inject_data_drift(lot, rng)
    if month == config.CONCEPT_DRIFT_MONTH:
        return inject_concept_drift(lot, rng)
    return lot


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--month", type=int, required=True, choices=range(1, 5))
    month = parser.parse_args().month

    received = receive_month(month)
    config.BATCHES_DIR.mkdir(parents=True, exist_ok=True)
    received.to_parquet(config.month_path(month), index=False)
    print(
        f"Mois {month} reçu : {len(received)} demandes, "
        f"{received['TARGET'].mean():.2%} de défauts → {config.month_path(month)}"
    )


if __name__ == "__main__":
    main()
