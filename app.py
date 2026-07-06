import json
import logging
import pickle
import time
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Request
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

# --- API ---
app = FastAPI(title="Credit Scoring API", version="1.0.0")

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
    response.headers["Content-Security-Policy"] = "default-src 'none'; frame-ancestors 'none'"
    response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
    if "server" in response.headers:
        del response.headers["server"]  # ne pas exposer "uvicorn"
    return response


# --- Handler d'erreur générique : aucune stacktrace ne fuit en prod ---
@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    logger.error(json.dumps({"event": "unhandled_error", "path": str(request.url.path)}))
    return JSONResponse(status_code=500, content={"detail": "Erreur interne"})


class PredictionResponse(BaseModel):
    SK_ID_CURR: int
    probability: float
    decision: str
    threshold: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
@limiter.limit("20/minute")
def predict(request: Request, SK_ID_CURR: int):
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
