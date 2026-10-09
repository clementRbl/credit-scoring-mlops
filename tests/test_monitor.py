import numpy as np
import pandas as pd
import pytest

from pipeline.monitor import assess, measure_data_drift, reference_cost, top_features
from pipeline.train import train_model


def test_aucune_derive_entre_deux_tirages_de_la_meme_population():
    rng = np.random.default_rng(0)
    make = lambda: pd.DataFrame(  # noqa: E731
        {"A": rng.normal(0, 1, 3000), "B": rng.normal(5, 2, 3000)}
    )
    share, drifted = measure_data_drift(make(), make(), ["A", "B"])
    assert share == 0
    assert drifted == []


def test_la_derive_est_mesuree_variable_par_variable():
    rng = np.random.default_rng(0)
    reference = pd.DataFrame(
        {col: rng.normal(0, 1, 3000) for col in ["A", "B", "C", "D"]}
    )
    current = reference.copy()
    current["A"] = current["A"] + 2
    current["C"] = current["C"] * 3
    share, drifted = measure_data_drift(reference, current, ["A", "B", "C", "D"])
    assert share == pytest.approx(0.5)
    assert sorted(drifted) == ["A", "C"]


@pytest.mark.parametrize(
    ("drift_share", "cost", "expected_drift", "expected_alert"),
    [
        (0.29, 0.50, False, False),
        (0.30, 0.50, True, False),  # 30 % de variables qui dérivent : alerte
        (0.00, 0.55, False, False),  # exactement +10 % : pas encore d'alerte
        (0.00, 0.551, False, True),
    ],
)
def test_les_seuils_d_alerte(drift_share, cost, expected_drift, expected_alert):
    data_drift, performance_alert = assess(drift_share, cost, reference_cost=0.50)
    assert data_drift is expected_drift
    assert performance_alert is expected_alert


def test_les_variables_surveillees_sont_les_colonnes_d_origine():
    rng = np.random.default_rng(0)
    n = 500
    df = pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(n),
            "EFFORT": rng.random(n),
            "NOISE": rng.random(n),
            "CODE_GENDER": rng.choice(["F", "M"], n),
        }
    )
    df["TARGET"] = ((df["EFFORT"] > 0.7) | (df["CODE_GENDER"] == "F")).astype(int)
    model = train_model(df)
    features = top_features(model.pipeline, n=3)
    # « cat__CODE_GENDER_F » redevient « CODE_GENDER », une seule fois
    assert set(features) == {"EFFORT", "NOISE", "CODE_GENDER"}
    assert features.index("EFFORT") < features.index("NOISE")


def test_la_reference_d_un_modele_promu_est_son_cout_a_la_promotion():
    """Après un changement de régime, même un modèle adapté coûte plus qu'avant :
    le comparer à l'ancien régime déclencherait une alerte chaque semaine."""
    promoted = {
        "metrics": {
            "frozen_test_cost_per_client": 0.49,
            "holdout_cost_per_client": 0.68,
        }
    }
    assert reference_cost(promoted) == 0.68


def test_la_reference_du_champion_initial_est_le_test_fige():
    initial = {"metrics": {"frozen_test_cost_per_client": 0.49}}
    assert reference_cost(initial) == 0.49
