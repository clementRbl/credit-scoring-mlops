import numpy as np
import pandas as pd
import pytest

from pipeline import config
from pipeline.simulate import inject_concept_drift, inject_data_drift


@pytest.fixture
def lot() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 4000
    return pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(n),
            "DAYS_BIRTH": -rng.uniform(20, 70, n) * 365.25,
            "AMT_CREDIT": rng.normal(500_000, 100_000, n),
            "AMT_GOODS_PRICE": rng.normal(450_000, 90_000, n),
            "AMT_ANNUITY": rng.normal(25_000, 5_000, n),
            "AMT_INCOME_TOTAL": rng.normal(150_000, 50_000, n),
            "NAME_INCOME_TYPE": rng.choice(["Working", "Pensioner"], n),
            "TARGET": (rng.random(n) < 0.08).astype(int),
        }
    )


def age(df: pd.DataFrame) -> pd.Series:
    return -df["DAYS_BIRTH"] / 365.25


def test_la_derive_des_donnees_rajeunit_le_lot_sans_inventer_de_demande(lot):
    drifted = inject_data_drift(lot, np.random.default_rng(1))
    assert set(drifted["SK_ID_CURR"]) < set(lot["SK_ID_CURR"])
    assert age(drifted).mean() < age(lot).mean() - 3
    young = lot[age(lot) < config.DRIFT_YOUNG_AGE]
    assert set(young["SK_ID_CURR"]) <= set(drifted["SK_ID_CURR"])


def test_la_derive_des_donnees_gonfle_les_montants(lot):
    drifted = inject_data_drift(lot, np.random.default_rng(1))
    original = lot.set_index("SK_ID_CURR").loc[drifted["SK_ID_CURR"]]
    for col in config.DRIFT_AMOUNT_COLUMNS:
        ratio = drifted[col].to_numpy() / original[col].to_numpy()
        assert ratio == pytest.approx(config.DRIFT_AMOUNT_FACTOR)


def test_la_derive_de_concept_ne_touche_que_les_etiquettes_du_segment(lot):
    drifted = inject_concept_drift(lot, np.random.default_rng(1))
    features = lot.columns.drop("TARGET")
    pd.testing.assert_frame_equal(drifted[features], lot[features])

    changed = drifted["TARGET"] != lot["TARGET"]
    segment = (lot["NAME_INCOME_TYPE"] == "Working") & (
        lot["AMT_INCOME_TOTAL"] < lot["AMT_INCOME_TOTAL"].median()
    )
    assert changed.any()
    assert (segment[changed]).all()
    assert (lot.loc[changed, "TARGET"] == 0).all()
    assert (drifted.loc[changed, "TARGET"] == 1).all()
