from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


# --- Health ---
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# --- Prédiction valide ---
def test_predict_valid_client():
    response = client.post("/predict?SK_ID_CURR=100001")
    assert response.status_code == 200
    data = response.json()
    assert data["SK_ID_CURR"] == 100001
    assert 0.0 <= data["probability"] <= 1.0
    assert data["decision"] in ("ACCORDE", "REFUSE")
    assert data["threshold"] == 0.47


# --- Format de réponse ---
def test_predict_response_fields():
    response = client.post("/predict?SK_ID_CURR=100001")
    data = response.json()
    expected_keys = {"SK_ID_CURR", "probability", "decision", "threshold"}
    assert set(data.keys()) == expected_keys


# --- Client inexistant (404) ---
def test_predict_unknown_client():
    response = client.post("/predict?SK_ID_CURR=999999999")
    assert response.status_code == 404
    assert "non trouvé" in response.json()["detail"]


# --- Type invalide (422) ---
def test_predict_invalid_type():
    response = client.post("/predict?SK_ID_CURR=abc")
    assert response.status_code == 422


# --- Paramètre manquant (422) ---
def test_predict_missing_param():
    response = client.post("/predict")
    assert response.status_code == 422


# --- Valeur négative (doit quand même retourner 404 car pas dans les données) ---
def test_predict_negative_id():
    response = client.post("/predict?SK_ID_CURR=-5")
    assert response.status_code == 404
