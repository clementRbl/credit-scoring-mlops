import numpy as np
import pandas as pd
import pytest

from pipeline.train import evaluate, train_model


@pytest.fixture
def labeled() -> pd.DataFrame:
    """Petit jeu où le risque monte avec le taux d'effort : appris en quelques ms."""
    rng = np.random.default_rng(0)
    n = 600
    effort = rng.random(n)
    return pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(n),
            "EFFORT": effort,
            "AMT_CREDIT": rng.normal(500_000, 100_000, n),
            "NAME_INCOME_TYPE": rng.choice(["Working", "Pensioner"], n),
            "TARGET": (rng.random(n) < effort * 0.4).astype(int),
        }
    )


def test_l_identifiant_n_est_jamais_une_variable_du_modele(labeled):
    model = train_model(labeled)
    column_transformer = model.pipeline.named_steps["preprocessor"].named_steps[
        "column_transformer"
    ]
    used = [col for _, _, cols in column_transformer.transformers_ for col in cols]
    assert "SK_ID_CURR" not in used
    assert "TARGET" not in used


def test_le_modele_apprend_le_signal_et_porte_son_seuil(labeled):
    model = train_model(labeled)
    assert model.oof_auc > 0.6
    assert 0.0 < model.threshold < 1.0
    assert model.training_rows == len(labeled)


def test_l_evaluation_utilise_le_seuil_du_modele(labeled):
    model = train_model(labeled)
    metrics = evaluate(model, labeled)
    assert set(metrics) == {"auc", "cost_per_client"}
    # Seuil à 1 : personne n'est refusé, le coût vaut 10 × le taux de défaut.
    model.threshold = 1.0
    expected = 10 * labeled["TARGET"].mean()
    assert evaluate(model, labeled)["cost_per_client"] == pytest.approx(expected)
