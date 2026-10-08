"""Découpe les données étiquetées en référence, lots mensuels et test figé.

Le jeu Home Credit n'a pas de dates : les « mois » sont simulés en tirant des lots
au hasard, de façon stratifiée pour que chacun garde le taux de défaut d'origine.

Usage : python -m pipeline.split
"""

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

from pipeline import config


def split_labeled(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, list[pd.DataFrame], pd.DataFrame]:
    """Renvoie (référence, lots mensuels, test figé). Même résultat à chaque appel."""
    rest, frozen_test = train_test_split(
        df,
        test_size=config.FROZEN_TEST_SHARE,
        stratify=df["TARGET"],
        random_state=config.SEED,
    )
    # Part des lots rapportée à ce qui reste après le test figé : 0,20 / 0,80
    batches_share_of_rest = config.BATCHES_SHARE / (1 - config.FROZEN_TEST_SHARE)
    reference, batch_pool = train_test_split(
        rest,
        test_size=batches_share_of_rest,
        stratify=rest["TARGET"],
        random_state=config.SEED,
    )
    folds = StratifiedKFold(
        n_splits=config.N_BATCHES, shuffle=True, random_state=config.SEED
    )
    batches = [
        batch_pool.iloc[batch_index]
        for _, batch_index in folds.split(batch_pool, batch_pool["TARGET"])
    ]
    return reference, batches, frozen_test


def describe(name: str, part: pd.DataFrame) -> None:
    print(f"{name:<10}: {len(part)} demandes, {part['TARGET'].mean():.2%} de défauts")


def main() -> None:
    df = pd.read_parquet(config.LABELED_DATA)
    reference, batches, frozen_test = split_labeled(df)

    config.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    reference.to_parquet(config.REFERENCE_PATH, index=False)
    frozen_test.to_parquet(config.FROZEN_TEST_PATH, index=False)
    for month, batch in enumerate(batches, start=1):
        batch.to_parquet(config.batch_path(month), index=False)

    describe("Référence", reference)
    for month, batch in enumerate(batches, start=1):
        describe(f"Lot {month}", batch)
    describe("Test figé", frozen_test)


if __name__ == "__main__":
    main()
