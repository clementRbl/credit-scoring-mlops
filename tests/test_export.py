import json
import pickle

import numpy as np
import pandas as pd

from pipeline.export import save_for_api
from pipeline.train import features, train_model


def test_l_api_retrouve_le_modele_et_son_seuil(tmp_path):
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            "SK_ID_CURR": np.arange(400),
            "EFFORT": rng.random(400),
            "NAME_INCOME_TYPE": rng.choice(["Working", "Pensioner"], 400),
        }
    )
    df["TARGET"] = (rng.random(400) < df["EFFORT"] * 0.4).astype(int)
    model = train_model(df)

    save_for_api(
        model,
        version=3,
        metrics={"frozen_test_auc": 0.71234},
        model_path=tmp_path / "model.pkl",
        meta_path=tmp_path / "model_meta.json",
    )

    with open(tmp_path / "model.pkl", "rb") as f:
        reloaded = pickle.load(f)
    meta = json.loads((tmp_path / "model_meta.json").read_text())
    assert np.allclose(
        reloaded.predict_proba(features(df))[:, 1],
        model.pipeline.predict_proba(features(df))[:, 1],
    )
    assert meta["threshold"] == model.threshold
    assert meta["model_version"] == 3
    assert meta["metrics"] == {"frozen_test_auc": 0.7123}
