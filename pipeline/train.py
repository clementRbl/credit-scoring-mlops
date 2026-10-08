"""Entraînement d'un modèle de scoring et choix de son seuil de décision."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

from pipeline import config
from src.business_metric import cost_per_client, find_best_threshold
from src.preprocessing import EXCLUDE_COLS, build_preprocessor, get_feature_columns


@dataclass
class TrainedModel:
    pipeline: Pipeline
    threshold: float  # seuil propre au modèle : refus dès que proba >= seuil
    oof_auc: float
    training_rows: int


def features(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(columns=[c for c in EXCLUDE_COLS if c in df.columns])


def build_model(df: pd.DataFrame) -> Pipeline:
    numeric_cols, categorical_cols = get_feature_columns(df)
    return Pipeline(
        [
            ("preprocessor", build_preprocessor(numeric_cols, categorical_cols)),
            ("classifier", LGBMClassifier(**config.CHAMPION_PARAMS)),
        ]
    )


def train_model(train_df: pd.DataFrame) -> TrainedModel:
    """Entraîne sur tout `train_df`, après avoir choisi le seuil sur des prédictions
    out-of-fold : chaque demande y est scorée par un modèle qui ne l'a pas vue."""
    X = features(train_df)
    y = train_df["TARGET"].to_numpy()
    folds = StratifiedKFold(
        n_splits=config.THRESHOLD_CV_FOLDS, shuffle=True, random_state=config.SEED
    )
    oof_proba = cross_val_predict(
        build_model(train_df), X, y, cv=folds, method="predict_proba"
    )[:, 1]
    pipeline = build_model(train_df).fit(X, y)
    return TrainedModel(
        pipeline=pipeline,
        threshold=find_best_threshold(y, oof_proba),
        oof_auc=float(roc_auc_score(y, oof_proba)),
        training_rows=len(train_df),
    )


def predict_proba(model: TrainedModel, df: pd.DataFrame) -> np.ndarray:
    return model.pipeline.predict_proba(features(df))[:, 1]


def evaluate(model: TrainedModel, df: pd.DataFrame) -> dict[str, float]:
    """AUC et coût métier par client, au seuil du modèle."""
    y = df["TARGET"].to_numpy()
    proba = predict_proba(model, df)
    return {
        "auc": float(roc_auc_score(y, proba)),
        "cost_per_client": cost_per_client(y, proba, model.threshold),
    }
