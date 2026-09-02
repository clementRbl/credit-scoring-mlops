import json
import logging
import pickle
import time
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

# --- Logging structuré JSON ---
logger = logging.getLogger("api")
logger.setLevel(logging.INFO)

log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
file_handler = logging.FileHandler(log_dir / "predictions.jsonl")
file_handler.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(file_handler)

# --- Chargement modèle et données au démarrage ---
MODEL_PATH = Path("model/model.pkl")
DATA_PATH = Path("data/processed/test_merged.parquet")
THRESHOLD = 0.47

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

# Forcer le modèle en CPU
classifier = model.named_steps["classifier"]
if hasattr(classifier, "set_params"):
    classifier.set_params(device="cpu")

# Récupérer les features
preprocessor = model.named_steps["preprocessor"]
ct = preprocessor.named_steps["column_transformer"]
FEATURES = list(ct.transformers_[0][2]) + list(ct.transformers_[1][2])

# Charger les données clients
clients_df = pd.read_parquet(DATA_PATH)
clients_df = clients_df.set_index("SK_ID_CURR")

DESCRIPTION = """
Scoring de risque de crédit, servi en production.

Le modèle est un **LightGBM** entraîné sur le jeu *Home Credit Default Risk*. Il estime
la
probabilité qu'une demande de crédit se solde par un défaut de paiement ; un seuil de
décision
transforme ensuite cette probabilité en avis d'octroi.

### Comment l'essayer en trente secondes

1. `GET /clients` renvoie des identifiants de demandes valides.
2. Copiez-en un dans `POST /predict`, ou gardez celui qui est déjà proposé en exemple.

L'API ne prend **qu'un identifiant de demande**, pas un dossier complet : elle retrouve
la
demande dans le jeu qu'elle sert, en extrait les variables attendues par le modèle, et
renvoie
la probabilité accompagnée de la décision.

### Pourquoi le seuil est à 0,47 et non à 0,50

Parce que les deux erreurs ne coûtent pas la même chose. Accorder un prêt à quelqu'un
qui ne
remboursera pas fait perdre le capital ; refuser un bon client ne fait perdre qu'une
marge. Le
projet chiffre ce rapport à **dix contre un**, et 0,47 est le seuil qui minimise le coût
total
sur le jeu de validation. Le seuil n'appartient donc pas au modèle : c'est une décision
de
gestion posée sur sa sortie, et l'API la rend explicite en la renvoyant dans chaque
réponse.

### Limites connues

- Les demandes proviennent du jeu de **test** Kaggle, qui n'est pas étiqueté : il est
  impossible de vérifier ici si une prédiction était juste.
- Aucune donnée personnelle. Les identifiants et les variables sont anonymisés à la
  source, et
  les montants sont exprimés dans l'unité du jeu de données, sans devise précisée.
- **20 requêtes par minute et par adresse IP** sur `/predict`. Au-delà, la réponse
  est un code 429.
- Le service tourne sur un espace gratuit : le premier appel après une période
  d'inactivité
  peut demander quelques secondes de réveil.
"""

TAGS = [
    {
        "name": "Scoring",
        "description": "Obtenir une probabilité de défaut et la décision associée.",
    },
    {"name": "Données", "description": "Trouver des identifiants de demandes valides."},
    {"name": "Santé", "description": "Vérifier que le service répond."},
]

# --- API ---
app = FastAPI(
    title="Credit Scoring API",
    description=DESCRIPTION,
    version="1.1.0",
    openapi_tags=TAGS,
    contact={"name": "Clément Reboul", "url": "https://clement-reboul.fr/portfolio/"},
    license_info={
        "name": "Code source",
        "url": "https://github.com/clementRbl/credit-scoring-mlops",
    },
)

# --- Rate limiting (protège le compute HF d'une boucle abusive) ---
limiter = Limiter(key_func=get_remote_address, default_limits=["60/minute"])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# --- CORS restreint (au lieu de "*") : n'autoriser que les origines connues ---
ALLOWED_ORIGINS = [
    "https://clement-reboul.fr",
    "https://clementrbl.github.io",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# --- En-têtes de sécurité + masquage de la bannière serveur ---
@app.middleware("http")
async def security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    # Swagger UI (/docs) et ReDoc (/redoc) chargent leurs assets depuis un CDN :
    # une CSP "default-src 'none'" les casse (page blanche). On l'allège pour ces pages,
    # et on garde la CSP verrouillée pour les endpoints API/données.
    path = request.url.path
    if path.startswith("/docs") or path.startswith("/redoc"):
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "style-src 'self' 'unsafe-inline' https://cdn.jsdelivr.net; "
            "img-src 'self' data: https://fastapi.tiangolo.com; "
            "worker-src 'self' blob:; "
            "frame-ancestors 'none'"
        )
    else:
        response.headers["Content-Security-Policy"] = (
            "default-src 'none'; frame-ancestors 'none'"
        )
    response.headers["Strict-Transport-Security"] = (
        "max-age=31536000; includeSubDomains"
    )
    if "server" in response.headers:
        del response.headers["server"]  # ne pas exposer "uvicorn"
    return response


# --- Handler d'erreur générique : aucune stacktrace ne fuit en prod ---
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error(
        json.dumps({"event": "unhandled_error", "path": str(request.url.path)})
    )
    return JSONResponse(status_code=500, content={"detail": "Erreur interne"})


class PredictionResponse(BaseModel):
    """Réponse du scoring. `threshold` est renvoyé pour que la décision soit
    rejouable : avec la probabilité et le seuil, on refait le calcul soi-même."""

    SK_ID_CURR: int
    probability: float
    decision: str
    threshold: float

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "SK_ID_CURR": 100001,
                    "probability": 0.3718,
                    "decision": "ACCORDE",
                    "threshold": 0.47,
                },
                {
                    "SK_ID_CURR": 100005,
                    "probability": 0.666,
                    "decision": "REFUSE",
                    "threshold": 0.47,
                },
            ]
        }
    }


class ClientSummary(BaseModel):
    """Caractéristiques lisibles d'une demande, pour situer un dossier.
    Le modèle en lit bien davantage ; celles-ci ne servent qu'à comprendre."""

    SK_ID_CURR: int
    age_years: int
    employment_years: float | None
    income: float
    credit: float
    annuity: float | None
    children: int
    income_type: str


class ClientsResponse(BaseModel):
    count: int
    note: str
    clients: list[ClientSummary]


# Échantillon fixe : le même à chaque redémarrage, pour que la documentation et
# les exemples restent stables d'une visite à l'autre.
SAMPLE_IDS = sorted(int(i) for i in clients_df.sample(n=20, random_state=0).index)

# 365243 est la valeur sentinelle de DAYS_EMPLOYED dans Home Credit : elle marque
# les personnes sans emploi en cours, essentiellement des retraités. La rendre
# telle quelle donnerait « 1000 ans d'ancienneté ».
EMPLOYED_SENTINEL = 365243


@app.get(
    "/health",
    tags=["Santé"],
    summary="Le service répond-il ?",
    description="Sonde de disponibilité. Ne charge rien et ne consomme aucun quota.",
    responses={200: {"content": {"application/json": {"example": {"status": "ok"}}}}},
)
def health():
    return {"status": "ok"}


@app.get(
    "/clients",
    response_model=ClientsResponse,
    tags=["Données"],
    summary="Des identifiants de demandes valides",
    description=(
        "Renvoie un échantillon d'identifiants réellement présents dans le jeu "
        "servi, avec quelques caractéristiques lisibles. Sert à essayer `/predict` "
        "sans avoir à deviner un identifiant : tout entier pris au hasard "
        "renvoie un 404."
    ),
)
@limiter.limit("30/minute")
def clients(
    request: Request,
    limit: int = Query(10, ge=1, le=20, description="Nombre de demandes à renvoyer."),
):
    rows = []
    for sk_id in SAMPLE_IDS[:limit]:
        row = clients_df.loc[sk_id]
        days_employed = int(row["DAYS_EMPLOYED"])
        annuity = row["AMT_ANNUITY"]
        rows.append(
            ClientSummary(
                SK_ID_CURR=int(sk_id),
                age_years=int(-int(row["DAYS_BIRTH"]) // 365.25),
                employment_years=(
                    None
                    if days_employed == EMPLOYED_SENTINEL
                    else round(-days_employed / 365.25, 1)
                ),
                income=float(row["AMT_INCOME_TOTAL"]),
                credit=float(row["AMT_CREDIT"]),
                annuity=None if pd.isna(annuity) else float(annuity),
                children=int(row["CNT_CHILDREN"]),
                income_type=str(row["NAME_INCOME_TYPE"]),
            )
        )
    return ClientsResponse(
        count=len(rows),
        note=(
            "Montants dans l'unité du jeu de données, sans devise précisée. "
            "employment_years vaut null pour les personnes sans emploi en cours."
        ),
        clients=rows,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    tags=["Scoring"],
    summary="Scorer une demande de crédit",
    description=(
        "Renvoie la probabilité de défaut de la demande et la décision qui en découle. "
        "L'identifiant doit exister dans le jeu servi : voyez `GET /clients` "
        "pour en obtenir."
    ),
    responses={
        404: {
            "description": "Aucune demande ne porte cet identifiant.",
            "content": {
                "application/json": {
                    "example": {"detail": "Client 999999999 non trouvé"}
                }
            },
        },
        429: {
            "description": "Plus de 20 requêtes en une minute depuis cette adresse IP.",
            "content": {
                "application/json": {"example": {"error": "Rate limit exceeded"}}
            },
        },
    },
)
@limiter.limit("20/minute")
def predict(
    request: Request,
    SK_ID_CURR: int = Query(
        ...,
        description=(
            "Identifiant anonymisé d'une demande de crédit dans le jeu Home Credit. "
            "C'est la seule entrée de l'API."
        ),
        examples=[100001],
        openapi_examples={
            "accorde": {
                "summary": "Une demande acceptée",
                "description": "Probabilité de défaut sous le seuil.",
                "value": 100001,
            },
            "refuse": {
                "summary": "Une demande refusée",
                "description": "Probabilité de défaut au-dessus du seuil.",
                "value": 100005,
            },
            "inconnu": {
                "summary": "Un identifiant qui n'existe pas",
                "description": "Pour voir la réponse 404.",
                "value": 999999999,
            },
        },
    ),
):
    if SK_ID_CURR not in clients_df.index:
        raise HTTPException(status_code=404, detail=f"Client {SK_ID_CURR} non trouvé")

    client_data = clients_df.loc[[SK_ID_CURR], FEATURES]

    start = time.time()
    # Inférence directe via le pipeline sklearn
    proba = float(model.predict_proba(client_data)[:, 1][0])
    inference_time_ms = (time.time() - start) * 1000

    decision = "REFUSE" if proba >= THRESHOLD else "ACCORDE"

    # Log structuré
    log_entry = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "SK_ID_CURR": SK_ID_CURR,
        "probability": round(float(proba), 4),
        "decision": decision,
        "inference_time_ms": round(inference_time_ms, 2),
    }
    logger.info(json.dumps(log_entry))

    return PredictionResponse(
        SK_ID_CURR=SK_ID_CURR,
        probability=round(float(proba), 4),
        decision=decision,
        threshold=THRESHOLD,
    )
