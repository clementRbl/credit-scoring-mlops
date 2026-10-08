"""Réentraîne un challenger après une alerte de surveillance et décide de sa promotion.

Le challenger apprend sur la référence, les mois déjà reçus et 70 % du mois
courant. Les 30 % restants servent à le comparer au champion, aucun des deux ne
les ayant vus. Le test figé sert de garde-fou : on juge sur le présent sans
casser le passé.

Si le challenger est promu, le modèle et ses métadonnées sont écrits dans
`model/` : le workflow en fait une Pull Request, qu'un humain valide.

Usage : python -m pipeline.retrain --month 4
"""

import argparse
import json

import pandas as pd
from sklearn.model_selection import train_test_split

from pipeline import config, registry
from pipeline.export import load_served, save_for_api
from pipeline.train import evaluate, train_model
from src.business_metric import cost_per_client
from src.preprocessing import EXCLUDE_COLS


def assemble_training_data(
    reference: pd.DataFrame, previous_months: list[pd.DataFrame], month: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(données d'entraînement du challenger, part du mois gardée pour l'évaluer)."""
    month_train, holdout = train_test_split(
        month,
        test_size=config.HOLDOUT_SHARE,
        stratify=month["TARGET"],
        random_state=config.SEED,
    )
    train = pd.concat([reference, *previous_months, month_train], ignore_index=True)
    return train, holdout


def decide_promotion(
    champion_holdout: float,
    challenger_holdout: float,
    champion_test: float,
    challenger_test: float,
) -> bool:
    """Coûts métier par client. Le challenger doit faire mieux d'au moins 1 % sur
    les données récentes, sans dégrader de plus de 2 % le test figé."""
    better_now = challenger_holdout <= champion_holdout * (
        1 - config.PROMOTION_MIN_GAIN
    )
    past_preserved = challenger_test <= champion_test * (1 + config.PROMOTION_MAX_LOSS)
    return better_now and past_preserved


def served_cost(df: pd.DataFrame) -> float:
    """Coût par client du champion actuellement servi, à son propre seuil."""
    pipeline, meta = load_served()
    proba = pipeline.predict_proba(df.drop(columns=EXCLUDE_COLS))[:, 1]
    return cost_per_client(df["TARGET"].to_numpy(), proba, meta["threshold"])


def retrain(month: int) -> dict:
    reference = pd.read_parquet(config.REFERENCE_PATH)
    previous = [pd.read_parquet(config.month_path(m)) for m in range(1, month)]
    current = pd.read_parquet(config.month_path(month))
    frozen_test = pd.read_parquet(config.FROZEN_TEST_PATH)
    train, holdout = assemble_training_data(reference, previous, current)

    challenger = train_model(train)
    challenger_holdout = evaluate(challenger, holdout)["cost_per_client"]
    challenger_test = evaluate(challenger, frozen_test)
    champion_holdout = served_cost(holdout)
    champion_test = served_cost(frozen_test)
    _, champion_meta = load_served()

    promoted = decide_promotion(
        champion_holdout,
        challenger_holdout,
        champion_test,
        challenger_test["cost_per_client"],
    )
    metrics = {
        "frozen_test_auc": challenger_test["auc"],
        "frozen_test_cost_per_client": challenger_test["cost_per_client"],
        "holdout_cost_per_client": challenger_holdout,
    }
    version = registry.register(
        challenger,
        metrics,
        tags={"trained_on": f"reference+months_1-{month}"},
        alias="challenger",
    )
    if promoted:
        save_for_api(challenger, version, metrics)

    result = {
        "month": month,
        "promoted": promoted,
        "champion_version": champion_meta["model_version"],
        "challenger_version": version,
        "training_rows": challenger.training_rows,
        "holdout_rows": len(holdout),
        "champion_holdout_cost": round(champion_holdout, 4),
        "challenger_holdout_cost": round(challenger_holdout, 4),
        "champion_test_cost": round(champion_test, 4),
        "challenger_test_cost": round(challenger_test["cost_per_client"], 4),
        "champion_threshold": champion_meta["threshold"],
        "challenger_threshold": challenger.threshold,
    }
    out_dir = config.REPORTS_DIR / f"month_{month}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "promotion.json").write_text(json.dumps(result, indent=2) + "\n")
    (out_dir / "promotion.md").write_text(to_markdown(result))
    return result


def to_markdown(result: dict) -> str:
    """Corps de la Pull Request de promotion : ce qu'un relecteur doit trancher."""
    holdout_change = (
        result["challenger_holdout_cost"] / result["champion_holdout_cost"] - 1
    )
    test_change = result["challenger_test_cost"] / result["champion_test_cost"] - 1
    verdict = (
        "Le challenger remplit la règle de promotion."
        if result["promoted"]
        else "Le challenger ne remplit pas la règle : le champion reste en service."
    )
    lines = [
        f"## Mois {result['month']} : champion v{result['champion_version']} "
        f"contre challenger v{result['challenger_version']}",
        "",
        verdict,
        "",
        "| Coût métier par demande | Champion | Challenger | Écart | Règle |",
        "|---|---|---|---|---|",
        f"| Données récentes ({result['holdout_rows']} demandes jamais vues) "
        f"| {result['champion_holdout_cost']} | {result['challenger_holdout_cost']} "
        f"| {holdout_change:+.1%} | ≤ -{config.PROMOTION_MIN_GAIN:.0%} |",
        f"| Test figé (garde-fou) | {result['champion_test_cost']} "
        f"| {result['challenger_test_cost']} | {test_change:+.1%} "
        f"| ≤ +{config.PROMOTION_MAX_LOSS:.0%} |",
        "",
        f"Seuil de décision : {result['champion_threshold']} → "
        f"{result['challenger_threshold']}. Le challenger a appris sur "
        f"{result['training_rows']} demandes ; il est enregistré dans le Model "
        f"Registry sous l'alias `challenger`.",
        "",
        "Fusionner cette PR déploie le challenger et lui donne l'alias `champion`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--month", type=int, required=True)
    result = retrain(parser.parse_args().month)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
