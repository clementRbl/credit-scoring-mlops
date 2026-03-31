# Credit Scoring MLOps

API de scoring crédit pour l'entreprise "Prêt à Dépenser". Ce projet déploie un modèle LightGBM qui prédit la probabilité de défaut de paiement d'un client.

## Lancer l'API

```bash
# Avec Docker
docker build -t credit-scoring-api .
docker run -p 8000:8000 credit-scoring-api

# Sans Docker
pip install -r requirements.txt
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

## Lancer les tests

```bash
pytest tests/ -v
```

## Monitoring

Le dashboard Streamlit permet de visualiser :
- La distribution des scores prédits
- Le temps de réponse de l'API
- L'analyse de data drift (Evidently)

```bash
streamlit run dashboard/app.py
```

## Structure du projet

```
├── api/              # Code de l'API FastAPI
├── tests/            # Tests unitaires
├── model/            # Modèle sérialisé (.pkl)
├── dashboard/        # Dashboard Streamlit de monitoring
├── notebooks/        # Analyse de data drift
├── Dockerfile
├── requirements.txt
└── .github/workflows/  # Pipeline CI/CD
```
