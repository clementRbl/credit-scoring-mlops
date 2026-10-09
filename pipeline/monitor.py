"""Surveille un lot mensuel : dérive des données et performance du champion.

Deux déclencheurs, parce qu'ils ne voient pas la même chose :
- la dérive des données (Evidently) voit une population qui change ;
- le coût métier voit un modèle qui se trompe davantage, même quand les données
  n'ont pas bougé (dérive de concept). Il suppose que les défauts du lot sont
  connus, ce qui arrive en pratique quelques mois après l'octroi.

Écrit dans reports/month_<N>/ : summary.json, summary.md (corps de l'issue
d'alerte) et drift_report.html (rapport Evidently).

Usage : python -m pipeline.monitor --month 3
"""

import argparse
import json

import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report
from sklearn.pipeline import Pipeline

from pipeline import config
from pipeline.export import load_served
from src.business_metric import cost_per_client
from src.preprocessing import EXCLUDE_COLS, sanitize_name


def original_column(name: str, categorical_cols: list[str]) -> str:
    """Nom de la variable d'origine d'une colonne vue par LightGBM :
    « num__AMT_CREDIT » → « AMT_CREDIT », « cat__CODE_GENDER_F » → « CODE_GENDER »."""
    if name.startswith("num__"):
        return name.removeprefix("num__")
    encoded = name.removeprefix("cat__")
    candidates = [
        c for c in categorical_cols if encoded.startswith(sanitize_name(c) + "_")
    ]
    if not candidates:
        raise ValueError(f"Colonne sans variable d'origine connue : {name}")
    # Le plus long gagne : « NAME_TYPE_SUITE_x » appartient à NAME_TYPE_SUITE,
    # pas à une éventuelle variable NAME_TYPE.
    return max(candidates, key=len)


def top_features(pipeline: Pipeline, n: int = config.MONITORED_FEATURES) -> list[str]:
    """Les `n` variables d'origine qui pèsent le plus dans les décisions (gain)."""
    classifier = pipeline.named_steps["classifier"]
    column_transformer = pipeline.named_steps["preprocessor"].named_steps[
        "column_transformer"
    ]
    numeric_cols = list(column_transformer.transformers_[0][2])
    categorical_cols = list(column_transformer.transformers_[1][2])
    raw_by_sanitized = {f"num__{sanitize_name(c)}": c for c in numeric_cols}
    gains = pd.Series(
        classifier.booster_.feature_importance("gain"), index=classifier.feature_name_
    ).sort_values(ascending=False)

    features: list[str] = []
    for name in gains.index:
        raw = raw_by_sanitized.get(name) or original_column(name, categorical_cols)
        if raw not in features:
            features.append(raw)
        if len(features) == n:
            break
    return features


def drift_report(
    reference: pd.DataFrame, current: pd.DataFrame, columns: list[str]
) -> Report:
    report = Report(metrics=[DataDriftPreset(columns=columns)])
    report.run(reference_data=reference[columns], current_data=current[columns])
    return report


def drift_summary(report: Report) -> tuple[float, list[str]]:
    """(part des variables qui dérivent, liste de ces variables)."""
    metrics = report.as_dict()["metrics"]
    share = metrics[0]["result"]["share_of_drifted_columns"]
    by_column = metrics[1]["result"]["drift_by_columns"]
    drifted = [col for col, result in by_column.items() if result["drift_detected"]]
    return float(share), drifted


def measure_data_drift(
    reference: pd.DataFrame, current: pd.DataFrame, columns: list[str]
) -> tuple[float, list[str]]:
    return drift_summary(drift_report(reference, current, columns))


def assess(drift_share: float, cost: float, reference_cost: float) -> tuple[bool, bool]:
    """(dérive des données, alerte de performance) selon les seuils de config."""
    data_drift = drift_share >= config.DRIFT_SHARE_ALERT
    performance_alert = cost > reference_cost * (1 + config.COST_DEGRADATION_ALERT)
    return data_drift, performance_alert


def reference_cost(meta: dict) -> float:
    """Coût par client auquel le modèle servi a été accepté.

    Un modèle promu a été jugé sur les données récentes de sa promotion : c'est son
    niveau de référence. Le comparer au test figé, d'un régime antérieur, ferait
    sonner l'alerte chaque semaine après un choc qui a durablement relevé les
    défauts. Le champion initial, entraîné avant tout lot, n'a que le test figé."""
    metrics = meta["metrics"]
    return metrics.get(
        "holdout_cost_per_client", metrics["frozen_test_cost_per_client"]
    )


def latest_month() -> int:
    months = [
        int(p.stem.removeprefix("month_")) for p in config.BATCHES_DIR.glob("*.parquet")
    ]
    if not months:
        raise FileNotFoundError(f"Aucun lot reçu dans {config.BATCHES_DIR}")
    return max(months)


def monitor(month: int) -> dict:
    pipeline, meta = load_served()
    lot = pd.read_parquet(config.month_path(month))
    reference = pd.read_parquet(config.REFERENCE_PATH).sample(
        config.DRIFT_REFERENCE_SAMPLE, random_state=config.SEED
    )
    columns = top_features(pipeline)
    report = drift_report(reference, lot, columns)
    drift_share, drifted = drift_summary(report)

    features = lot.drop(columns=EXCLUDE_COLS)
    proba = pipeline.predict_proba(features)[:, 1]
    cost = cost_per_client(lot["TARGET"].to_numpy(), proba, meta["threshold"])
    reference = reference_cost(meta)
    data_drift, performance_alert = assess(drift_share, cost, reference)

    summary = {
        "month": month,
        "model_version": meta["model_version"],
        "rows": len(lot),
        "drift_share": round(drift_share, 3),
        "drifted_columns": drifted,
        "cost_per_client": round(cost, 4),
        "reference_cost_per_client": reference,
        "cost_change": round(cost / reference - 1, 3),
        "data_drift": data_drift,
        "performance_alert": performance_alert,
        "retrain_needed": data_drift or performance_alert,
    }
    out_dir = config.REPORTS_DIR / f"month_{month}"
    out_dir.mkdir(parents=True, exist_ok=True)
    report.save_html(str(out_dir / "drift_report.html"))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out_dir / "summary.md").write_text(to_markdown(summary))
    return summary


def to_markdown(summary: dict) -> str:
    """Corps de l'issue d'alerte, lisible par l'équipe risque."""
    causes = []
    if summary["data_drift"]:
        causes.append(
            f"**Dérive des données** : {summary['drift_share']:.0%} des variables "
            f"surveillées ont changé ({', '.join(summary['drifted_columns'])})."
        )
    if summary["performance_alert"]:
        causes.append(
            f"**Performance dégradée** : coût métier de {summary['cost_per_client']} "
            f"par demande, soit {summary['cost_change']:+.0%} par rapport à la "
            f"référence ({summary['reference_cost_per_client']})."
        )
    title = f"Surveillance du mois {summary['month']}"
    lines = [
        f"## {title} (modèle v{summary['model_version']})",
        "",
        *(f"- {cause}" for cause in causes or ["Aucune alerte."]),
        "",
        "| Indicateur | Valeur | Seuil d'alerte |",
        "|---|---|---|",
        f"| Variables qui dérivent | {summary['drift_share']:.0%} "
        f"| {config.DRIFT_SHARE_ALERT:.0%} |",
        f"| Variation du coût métier | {summary['cost_change']:+.0%} "
        f"| +{config.COST_DEGRADATION_ALERT:.0%} |",
        f"| Demandes dans le lot | {summary['rows']} | |",
        "",
        "Le rapport Evidently complet est joint à l'exécution du workflow.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--month", type=int, help="par défaut : le dernier lot reçu")
    month = parser.parse_args().month or latest_month()
    summary = monitor(month)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
