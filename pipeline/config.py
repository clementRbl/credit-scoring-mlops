"""Paramètres du pipeline : chemins, découpage des données, réglages du modèle."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Données étiquetées (sortie du feature engineering du projet de modélisation)
LABELED_DATA = ROOT / "data" / "processed" / "train_merged.parquet"

# Découpage : référence (entraîne le champion v1) / lots mensuels / test figé
SPLITS_DIR = ROOT / "data" / "splits"
REFERENCE_PATH = SPLITS_DIR / "reference.parquet"
FROZEN_TEST_PATH = SPLITS_DIR / "frozen_test.parquet"
REFERENCE_SHARE = 0.60
BATCHES_SHARE = 0.20
FROZEN_TEST_SHARE = 0.20
N_BATCHES = 4

SEED = 42

# Modèle servi par l'API et ses métadonnées (seuil, version)
MODEL_PATH = ROOT / "model" / "model.pkl"
MODEL_META_PATH = ROOT / "model" / "model_meta.json"

# Hyperparamètres du modèle en production depuis la V1 : tout réentraînement les
# reprend, pour que seule la donnée change entre champion et challenger.
CHAMPION_PARAMS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "class_weight": "balanced",
    "random_state": SEED,
    "verbosity": -1,
}

# Seuil de décision cherché sur des prédictions out-of-fold
THRESHOLD_CV_FOLDS = 5


def batch_path(month: int) -> Path:
    """Lot propre du mois `month` (1 à N_BATCHES), avant toute dérive simulée."""
    return SPLITS_DIR / f"lot_{month}.parquet"


# --- Lots mensuels reçus (après dérives simulées) ---
BATCHES_DIR = ROOT / "data" / "batches"


def month_path(month: int) -> Path:
    """Lot reçu le mois `month`, tel que le pipeline de surveillance le lit."""
    return BATCHES_DIR / f"month_{month}.parquet"


# --- Dérives simulées (voir pipeline/simulate.py) ---
# Mois 3, dérive des données : une campagne attire une clientèle plus jeune,
# pour des montants plus élevés.
DATA_DRIFT_MONTH = 3
DRIFT_YOUNG_AGE = 40  # en années
DRIFT_OLDER_KEEP_SHARE = 0.25  # part des 40 ans et plus conservée dans le lot
DRIFT_AMOUNT_COLUMNS = ("AMT_CREDIT", "AMT_GOODS_PRICE", "AMT_ANNUITY")
DRIFT_AMOUNT_FACTOR = 1.3
# Mois 4, dérive de concept : un choc économique fait défaillir une partie des
# salariés aux revenus modestes. Les données ne bougent pas, seule la relation
# entre les variables et le défaut change.
CONCEPT_DRIFT_MONTH = 4
CONCEPT_DRIFT_DEFAULT_RATE = 0.15  # part des bons payeurs du segment qui font défaut

# --- Surveillance (voir pipeline/monitor.py) ---
MONITORED_FEATURES = 20  # les variables les plus importantes du champion
DRIFT_REFERENCE_SAMPLE = 20_000  # échantillon de la référence comparé à chaque lot
DRIFT_SHARE_ALERT = 0.30  # alerte si au moins 30 % de ces variables dérivent
COST_DEGRADATION_ALERT = 0.10  # alerte si le coût métier dépasse la référence de 10 %
REPORTS_DIR = ROOT / "reports"

# --- Réentraînement et promotion (voir pipeline/retrain.py) ---
HOLDOUT_SHARE = 0.30  # part du mois courant gardée pour comparer les deux modèles
PROMOTION_MIN_GAIN = 0.01  # le challenger doit coûter au moins 1 % de moins...
PROMOTION_MAX_LOSS = 0.02  # ... sans coûter plus de 2 % de plus sur le test figé
