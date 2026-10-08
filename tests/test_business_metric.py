import numpy as np
import pytest

from src.business_metric import business_cost, cost_per_client, find_best_threshold


def test_un_defaut_rate_coute_dix_fois_un_refus_injustifie():
    y_true = np.array([1, 0, 0, 0])
    y_proba = np.array([0.2, 0.9, 0.1, 0.1])  # 1 FN (premier), 1 FP (deuxième)
    assert business_cost(y_true, y_proba, threshold=0.5) == 11


def test_une_proba_egale_au_seuil_est_refusee():
    # L'API refuse dès que proba >= seuil : le coût doit suivre la même règle.
    y_true = np.array([0])
    assert business_cost(y_true, np.array([0.5]), threshold=0.5) == 1


def test_le_cout_par_client_ne_depend_pas_de_la_taille_du_jeu():
    y_true = np.array([1, 0])
    y_proba = np.array([0.1, 0.1])  # 1 FN
    double_y = np.concatenate([y_true, y_true])
    double_p = np.concatenate([y_proba, y_proba])
    assert cost_per_client(y_true, y_proba, 0.5) == pytest.approx(5.0)
    assert cost_per_client(double_y, double_p, 0.5) == pytest.approx(5.0)


def test_le_seuil_optimal_annule_le_cout_quand_les_classes_sont_separables():
    y_true = np.array([0, 0, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.7, 0.8])
    threshold = find_best_threshold(y_true, y_proba)
    assert business_cost(y_true, y_proba, threshold) == 0


def test_le_seuil_optimal_prefere_refuser_quand_le_doute_coute_cher():
    # Un défaut à 0,3 : l'attraper coûte 2 refus injustifiés (2), le rater coûte 10.
    y_true = np.array([0, 0, 1])
    y_proba = np.array([0.3, 0.3, 0.3])
    assert find_best_threshold(y_true, y_proba) <= 0.3
