import json
import logging
import pickle
import time
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

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

# Récupérer les features attendues
ct = model.named_steps["preprocessor"].named_steps["column_transformer"]
FEATURES = list(ct.transformers_[0][2]) + list(ct.transformers_[1][2])

# Charger les données clients
clients_df = pd.read_parquet(DATA_PATH)
clients_df = clients_df.set_index("SK_ID_CURR")

# --- API ---
app = FastAPI(title="Credit Scoring API", version="1.0.0")


class PredictionResponse(BaseModel):
    SK_ID_CURR: int
    probability: float
    decision: str
    threshold: float


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(SK_ID_CURR: int):
    if SK_ID_CURR not in clients_df.index:
        raise HTTPException(status_code=404, detail=f"Client {SK_ID_CURR} non trouvé")

    client_data = clients_df.loc[[SK_ID_CURR], FEATURES]

    start = time.time()
    proba = model.predict_proba(client_data)[:, 1][0]
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
