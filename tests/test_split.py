import numpy as np
import pandas as pd
import pytest

from pipeline.split import split_labeled


@pytest.fixture
def labeled() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 2000
    return pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(100000, 100000 + n),
            "AMT_CREDIT": rng.normal(500_000, 100_000, n),
            "TARGET": (rng.random(n) < 0.08).astype(int),
        }
    )


def ids(df: pd.DataFrame) -> set[int]:
    return set(df["SK_ID_CURR"])


def test_aucune_demande_n_est_dans_deux_parts(labeled):
    reference, batches, frozen_test = split_labeled(labeled)
    parts = [reference, frozen_test, *batches]
    assert sum(len(p) for p in parts) == len(labeled)
    assert set().union(*(ids(p) for p in parts)) == ids(labeled)


def test_les_parts_respectent_60_20_20_et_quatre_lots(labeled):
    reference, batches, frozen_test = split_labeled(labeled)
    assert len(reference) == pytest.approx(0.60 * len(labeled), abs=2)
    assert len(frozen_test) == pytest.approx(0.20 * len(labeled), abs=2)
    assert len(batches) == 4
    assert sum(len(b) for b in batches) == pytest.approx(0.20 * len(labeled), abs=2)


def test_chaque_part_garde_le_taux_de_defaut(labeled):
    overall = labeled["TARGET"].mean()
    reference, batches, frozen_test = split_labeled(labeled)
    for part in [reference, frozen_test, *batches]:
        assert part["TARGET"].mean() == pytest.approx(overall, abs=0.02)


def test_le_decoupage_est_identique_d_une_execution_a_l_autre(labeled):
    first = split_labeled(labeled)
    second = split_labeled(labeled)
    assert ids(first[0]) == ids(second[0])
    assert ids(first[2]) == ids(second[2])
    for batch_a, batch_b in zip(first[1], second[1]):
        assert ids(batch_a) == ids(batch_b)
