"""Coût métier du scoring : un défaut non détecté coûte 10 fois un refus injustifié."""

import numpy as np
from sklearn.metrics import confusion_matrix

COST_FN = 10  # mauvais client prédit bon : crédit accordé, capital perdu
COST_FP = 1  # bon client prédit mauvais : crédit refusé, marge perdue


def business_cost(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> float:
    """Coût total `10 × FN + 1 × FP` quand on refuse dès que proba >= seuil."""
    y_pred = (y_proba >= threshold).astype(int)
    _tn, fp, fn, _tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return float(COST_FN * fn + COST_FP * fp)


def cost_per_client(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> float:
    """Coût moyen par demande : comparable entre jeux de tailles différentes."""
    return business_cost(y_true, y_proba, threshold) / len(y_true)


def find_best_threshold(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """Seuil qui minimise le coût métier, cherché de 0,01 à 0,99 par pas de 0,01."""
    thresholds = np.round(np.linspace(0.01, 0.99, 99), 2)
    costs = [business_cost(y_true, y_proba, t) for t in thresholds]
    return float(thresholds[int(np.argmin(costs))])
