# Boucle dérive → réentraînement → promotion (V2)

- **Date** : 2026-10-08
- **Type** : feature
- **Statut** : vérifié

## Besoin
- **Problème** : le modèle de scoring est déployé et surveillé, mais la surveillance s'arrête à un notebook. Personne n'est alerté quand les données dérivent, rien ne réentraîne le modèle, et ni les données ni les modèles ne sont versionnés. Le modèle se dégrade donc en silence.
- **Pour qui** : l'équipe risque de Prêt à Dépenser, qui doit pouvoir se fier au score dans la durée. C'est aussi le projet personnel technique du portfolio (thèmes de la consigne : cycle de vie MLOps, réentraînement automatique, suivi de la dérive, versionnage DVC / MLflow Model Registry).
- **Pourquoi maintenant** : ce sont les quatre limites listées dans le rapport V1. Soutenance le 20 octobre 2026.
- **Exemple concret** : un lot mensuel arrive avec des montants de crédit 30 % plus élevés → la surveillance hebdomadaire détecte la dérive → le workflow ouvre une issue d'alerte → il réentraîne un challenger, le compare au champion et le propose dans une Pull Request avec le tableau des scores → après fusion, le CI/CD déploie et l'API annonce la nouvelle version du modèle.

## Simulation des données (le jeu Home Credit n'a pas de dates)
Un découpage unique, stratifié et à graine fixe, répartit les 307 511 demandes étiquetées (8,07 % de défauts) :

| Part | Rôle |
|---|---|
| 60 % | Référence : entraîne le champion v1 |
| 20 % | Quatre lots mensuels : les « nouvelles données » |
| 20 % | Test figé : jamais utilisé pour l'entraînement |

Les lots simulent quatre mois :
1. Mois 1 : stable (lot témoin).
2. Mois 2 : stable.
3. Mois 3 : dérive des données injectée. Une campagne attire une clientèle plus jeune : tous les moins de 40 ans sont gardés, mais seulement 25 % des autres. Les montants (crédit, prix du bien, annuité) sont multipliés par 1,3.
4. Mois 4 : dérive de concept injectée. Un choc économique fait défaillir 15 % des bons payeurs parmi les salariés aux revenus modestes ; les variables ne bougent pas.

`pipeline/config.py` fixe les paramètres des dérives, et le README les décrit.

Le modèle actuellement en production a vu 100 % des données étiquetées. Le champion v1 repart donc de la seule référence, avec les mêmes hyperparamètres (LightGBM, 200 arbres, profondeur 6, taux 0,1, `class_weight="balanced"`). Sinon, il aurait déjà appris les lots censés être nouveaux.

## Critères d'acceptation
- [ ] `dvc pull` récupère les données (découpages et lots) depuis DagsHub, puis `dvc status` ne signale aucun écart. Le modèle servi (732 Ko) reste dans git ; ses versions sont tenues par le MLflow Model Registry.
- [ ] Le découpage est déterministe : deux exécutions donnent les mêmes fichiers (même empreinte). Aucun `SK_ID_CURR` n'est commun à la référence, aux lots et au test figé (test).
- [ ] Le champion v1 est enregistré dans le MLflow Model Registry de DagsHub avec l'alias `champion`. AUC, coût métier et seuil y sont journalisés.
- [ ] L'API lit le seuil et la version dans les métadonnées du modèle (plus de 0,47 en dur). `GET /health` renvoie `model_version` (tests).
- [ ] Surveillance du mois 1 : ni dérive ni alerte de performance. Mois 3 : dérive détectée (part de variables dérivantes ≥ 30 % sur les 20 plus importantes). Mois 4 : alerte de performance (coût métier par client > référence + 10 %). Les fonctions de décision sont testées sur données synthétiques.
- [ ] En cas de dérive ou d'alerte de performance, le workflow ouvre une issue GitHub avec le résumé et joint le rapport Evidently à l'exécution (preuve : lien de l'exécution).
- [ ] Le challenger est entraîné sur la référence et les lots reçus, sans les 30 % du dernier lot gardés pour l'évaluation. Il est enregistré avec l'alias `challenger`.
- [ ] Règle de promotion : coût du challenger ≤ 0,99 × celui du champion sur les 30 % gardés, ET coût sur le test figé ≤ 1,02 × celui du champion. Testée aux bords : égalité, exactement −1 %, garde-fou exactement +2 %.
- [ ] Si la règle est remplie, le workflow ouvre une Pull Request : `model/model.pkl` et `model/model_meta.json` mis à jour, tableau champion / challenger dans la description. Après fusion, le CI/CD déploie, passe l'alias `champion` au nouveau modèle, et `/health` du Space affiche la nouvelle version.
- [ ] Le workflow de surveillance se lance chaque semaine et à la main (`workflow_dispatch` avec le numéro de lot).
- [ ] Les tests existants de l'API passent toujours. Le taux de couverture est mesuré et le même chiffre est reporté partout.

## Hors périmètre
- Journal de production persistant, validation des entrées, test de charge (listés en limites dans le rapport).
- Déploiement sans validation humaine.
- Recherche d'hyperparamètres à chaque réentraînement.
- Réentraînement à partir des requêtes réelles de l'API (le jeu servi n'est pas étiqueté).
- Passage d'Evidently 0.4.33 à 0.7.

## Métrique et évaluation
- **Métrique principale** : coût métier `10 × FN + 1 × FP` par client, au seuil optimal propre à chaque modèle. C'est ce que la banque perd réellement.
- **Jeux d'évaluation** : 30 % du dernier lot (jamais vus à l'entraînement) et le test figé (≈ 61 500 demandes).
- **Seuil bloquant** : la règle de promotion ci-dessus. Baseline = le champion v1, mesuré au premier entraînement.
- **Métriques secondaires** : AUC, part de variables dérivantes, durée d'entraînement, latence de l'API.

## Décisions techniques
| Décision | Choix | Pourquoi ici |
|---|---|---|
| Exécution | GitHub Actions (chaque semaine + à la main) | Le réentraînement est réellement automatique et visible par le jury |
| Registry + stockage DVC | DagsHub (gratuit) | MLflow hébergé, consultable par un lien, et stockage DVC compatible S3 au même endroit |
| Entraînement du challenger | Hyperparamètres du champion | Quelques minutes en CI ; seules les données changent, donc la comparaison est juste |
| Promotion | Pull Request validée par un humain | Décision de crédit, secteur régulé |
| Détection de dérive | Evidently 0.4.33 (déjà dans le projet) | Pas de changement d'interface à absorber |
| Seuil de décision | Stocké avec le modèle | Un modèle réentraîné a son propre seuil optimal |
| Dépendances | `requirements-pipeline.txt` séparé, `uv` | L'image de l'API reste légère |
| Versionnage | DVC pour les données, Registry pour les modèles | Le modèle pèse 732 Ko : en git, la PR montre le changement et la CI n'a pas besoin d'identifiants |
| Pilotage | Kanban GitHub Projects | Preuves prévu / réel pour la partie 5 du rapport |

## Plan
1. Mise en place : DagsHub, DVC, tableau Kanban, découpage des données, champion v1 entraîné et enregistré.
2. Simulation des lots et surveillance (dérive des données + performance), avec tests.
3. Réentraînement, règle de promotion, enregistrement du challenger, avec tests.
4. Workflows : surveillance hebdomadaire, issue d'alerte, Pull Request de promotion, déploiement inchangé (le modèle est dans git). L'API lit seuil et version.
5. Démonstration de bout en bout sur les mois 1 à 4, preuves et README.

## Vérification
| Critère | Preuve (commande → extrait de sortie) | OK / KO |
|---|---|---|
| `dvc pull` depuis DagsHub | CI rejouée depuis un clone vierge → `10 files fetched and 10 files added` ; étape « Récupérer les données » verte ([run](https://github.com/clementRbl/credit-scoring-mlops/actions/runs/37904674657)) | OK |
| Découpage déterministe, sans recouvrement | `python -m pipeline.split` lancé deux fois → `md5sum` identiques ; `tests/test_split.py` (4 tests) | OK |
| Champion v1 enregistré, alias `champion` | `get_model_version_by_alias("champion")` → `version 1`, étiquette `threshold: 0.5` (08/10) | OK |
| L'API lit seuil et version | API en ligne : `/health` → `{"status":"ok","model_version":4}`, `/predict` → `"threshold":0.48` | OK |
| Mois 1 sans alerte, mois 3 dérive, mois 4 performance | Runs GitHub : [mois 1](https://github.com/clementRbl/credit-scoring-mlops/actions/runs/37817647750) dérive 0 %, coût +1,9 % ; [mois 3](https://github.com/clementRbl/credit-scoring-mlops/actions/runs/37818016873) dérive 35 % ; [mois 4](https://github.com/clementRbl/credit-scoring-mlops/actions/runs/37904674657) dérive 0 %, coût +45,9 % | OK |
| Issue d'alerte | [#15](https://github.com/clementRbl/credit-scoring-mlops/issues/15) (mois 3), [#16](https://github.com/clementRbl/credit-scoring-mlops/issues/16) (mois 4) ; pas de doublon à la relance : « Issue déjà ouverte pour le mois 4 » | OK |
| Challenger entraîné sans la part gardée, alias `challenger` | `tests/test_retrain.py` (aucun identifiant commun) ; Registry : v2 (mois 3), v4 (mois 4) | OK |
| Règle de promotion testée aux bords | `tests/test_retrain.py` : égalité, -1 %, -0,8 %, +2 %, +2,25 % | OK |
| PR, fusion, déploiement, alias, `/health` | [PR #18](https://github.com/clementRbl/credit-scoring-mlops/pull/18) ouverte par le workflow (45 tests passés avant) ; après fusion : [CI/CD](https://github.com/clementRbl/credit-scoring-mlops/actions/runs/37905513756) vert, `Alias champion → version 4`, `/health` → `model_version: 4` | OK |
| Chaque semaine et à la main | `workflow_dispatch` prouvé (5 lancements) ; le déclenchement du lundi n'a pas encore été observé (premier : 12/10) | Partiel |
| Tests existants verts, couverture identique partout | 47 tests verts ; API couverte à 96 % (CI). README corrigé (98 % → 96 %) ; portfolio et rapport encore à aligner | Partiel |

- **La sortie est-elle bonne ?** Oui. Les chiffres obtenus sur GitHub Actions sont identiques au dix-millième près à ceux calculés en local (graine fixe) : mois 3, challenger 0,5616 contre 0,557 (non promu) ; mois 4, 0,6816 contre 0,7418 (promu).
- **Est-ce que ça fait sens ? Pourquoi ?** Le mois 3 rend la clientèle plus risquée sans rendre le modèle faux : réentraîner n'apporte rien et la règle refuse, à juste titre. Le mois 4 change la relation entre variables et défaut : seul le coût métier le voit (0 % de dérive des données), et un modèle qui apprend le nouveau régime fait nettement mieux (-8,1 % ; intervalle de confiance à 90 % mesuré en exploration : [-11,6 % ; -4,2 %]). La v4 gagne aussi sur le passé (AUC 0,782 contre 0,778 sur le test figé) : elle a appris sur 27 % de données en plus.
- **Points douteux / limites** :
  - Paramètres de simulation ajustés après essai : dérive du mois 3 renforcée (30 % de variables, pile au seuil, puis 35 %) ; choc du mois 4 porté de 8 % à 15 %, car à 8 % le gain du challenger restait dans le bruit. Le rapport de conduite de projet explique aussi ces deux ajustements.
  - Marge de promotion de 1 % petite face au bruit d'un mois (quelques milliers de demandes) : elle ne protège pas d'un gain dû au hasard.
  - Les défauts d'un lot sont supposés connus au moment de la surveillance ; dans une banque, ils arrivent des mois après l'octroi.
  - La dérive est mesurée par rapport à la population d'origine, même après une promotion.
  - Incidents : aiohttp 2.x retenu en CI (#14), sortie MLflow mêlée au JSON qui faisait sauter la PR (#17), et fausse alerte hebdomadaire anticipée après la promotion (référence de coût restée sur l'ancien régime, corrigée). La v3 du Registry, promue mais jamais livrée, est annotée.
- **Validé par l'utilisateur le** : 2026-10-09
