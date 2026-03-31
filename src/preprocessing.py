import re

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class ColumnNameSanitizer(BaseEstimator, TransformerMixin):
    """Nettoie les noms de colonnes pour LightGBM."""

    def fit(self, X, y=None):
        self.fitted_ = True
        return self

    def transform(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.copy()
            X.columns = [re.sub(r"[^\w]", "_", col).strip("_") for col in X.columns]
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is not None:
            return np.array(
                [re.sub(r"[^\w]", "_", f).strip("_") for f in input_features]
            )
        return input_features
