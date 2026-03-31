---
title: Credit Scoring API
emoji: 🏦
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
---

# Credit Scoring MLOps

API de scoring crédit pour l'entreprise "Prêt à Dépenser". Ce projet déploie un modèle LightGBM qui prédit la probabilité de défaut de paiement d'un client.

## Lancer l'API

```bash
# Avec Docker
docker build -t credit-scoring-api .
docker run -p 8000:8000 credit-scoring-api

# Sans Docker
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000
```

## Endpoints

- `GET /health` — vérification de l'état de l'API
- `POST /predict?SK_ID_CURR=100001` — prédiction pour un client
- `GET /docs` — documentation Swagger auto-générée

## Lancer les tests

```bash
pytest tests/ -v --cov=app
```

## Structure du projet

```
├── app.py                # API FastAPI
├── src/preprocessing.py  # Preprocessing du modèle
├── tests/                # Tests unitaires
├── model/                # Modèle sérialisé (.pkl)
├── data/processed/       # Données clients
├── logs/                 # Logs de prédiction (JSON)
├── Dockerfile
├── requirements.txt
└── .github/workflows/    # Pipeline CI/CD
```
