import numpy as np
import pandas as pd
import pytest

from pipeline.retrain import assemble_training_data, decide_promotion


@pytest.mark.parametrize(
    ("challenger_holdout", "challenger_test", "expected"),
    [
        (0.500, 0.400, False),  # égalité : réentraîner n'a rien apporté
        (0.495, 0.400, True),  # exactement -1 % sur les données récentes
        (0.496, 0.400, False),  # -0,8 % : gain dans le bruit
        (0.450, 0.408, True),  # garde-fou : exactement +2 % sur le test figé
        (0.450, 0.409, False),  # +2,25 % : le passé est trop dégradé
    ],
)
def test_la_regle_de_promotion(challenger_holdout, challenger_test, expected):
    promoted = decide_promotion(
        champion_holdout=0.500,
        challenger_holdout=challenger_holdout,
        champion_test=0.400,
        challenger_test=challenger_test,
    )
    assert promoted is expected


def lot(start: int, n: int, default_rate: float = 0.1) -> pd.DataFrame:
    rng = np.random.default_rng(start)
    return pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(start, start + n),
            "TARGET": (rng.random(n) < default_rate).astype(int),
        }
    )


def test_le_challenger_ne_voit_jamais_la_part_gardee_pour_l_evaluer():
    reference, month_1, month_2 = lot(0, 1000), lot(10_000, 300), lot(20_000, 300)
    train, holdout = assemble_training_data(reference, [month_1], month_2)

    train_ids, holdout_ids = set(train["SK_ID_CURR"]), set(holdout["SK_ID_CURR"])
    assert not train_ids & holdout_ids
    assert holdout_ids <= set(month_2["SK_ID_CURR"])
    assert len(holdout) == pytest.approx(0.30 * len(month_2), abs=1)
    # Tout le reste est appris : référence, mois précédents, 70 % du mois courant.
    expected = len(reference) + len(month_1) + len(month_2) - len(holdout)
    assert len(train) == expected
    assert set(reference["SK_ID_CURR"]) | set(month_1["SK_ID_CURR"]) <= train_ids


def test_la_part_gardee_respecte_le_taux_de_defaut_du_mois():
    month = lot(0, 2000, default_rate=0.2)
    _, holdout = assemble_training_data(lot(50_000, 100), [], month)
    assert holdout["TARGET"].mean() == pytest.approx(0.2, abs=0.02)
