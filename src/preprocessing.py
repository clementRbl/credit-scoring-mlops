"""Prétraitement du modèle de scoring, repris du projet de modélisation.

`ColumnNameSanitizer` doit rester dans ce module : les modèles picklés y font
référence par son chemin `src.preprocessing.ColumnNameSanitizer`.
"""

import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Colonnes qui ne sont pas des variables explicatives
EXCLUDE_COLS = ["SK_ID_CURR", "TARGET"]


def sanitize_name(name: str) -> str:
    """Remplace tout caractère hors [A-Za-z0-9_] par « _ » : « Higher education »
    devient « Higher_education »."""
    return re.sub(r"[^\w]", "_", name).strip("_")


class ColumnNameSanitizer(BaseEstimator, TransformerMixin):
    """Nettoie les noms de colonnes pour LightGBM."""

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.copy()
            X.columns = [sanitize_name(col) for col in X.columns]
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.array([sanitize_name(f) for f in input_features])
        return input_features


def get_feature_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Sépare les variables explicatives en numériques et catégorielles."""
    features = df.drop(columns=[c for c in EXCLUDE_COLS if c in df.columns])
    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = features.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()
    return numeric_cols, categorical_cols


def build_preprocessor(
    numeric_cols: list[str], categorical_cols: list[str]
) -> Pipeline:
    """Imputation (médiane / « missing »), encodage one-hot, puis noms de colonnes
    nettoyés : LightGBM refuse les caractères spéciaux dans les noms."""
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    column_transformer = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )
    column_transformer.set_output(transform="pandas")
    preprocessor = Pipeline(
        [
            ("column_transformer", column_transformer),
            ("sanitize_columns", ColumnNameSanitizer()),
        ]
    )
    preprocessor.set_output(transform="pandas")
    return preprocessor
