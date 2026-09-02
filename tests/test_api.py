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


# --- Identifiants de démonstration ---
def test_clients_shape():
    response = client.get("/clients")
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 10
    assert len(data["clients"]) == 10
    first = data["clients"][0]
    expected = {
        "SK_ID_CURR",
        "age_years",
        "employment_years",
        "income",
        "credit",
        "annuity",
        "children",
        "income_type",
    }
    assert set(first.keys()) == expected


def test_clients_limit():
    assert client.get("/clients?limit=3").json()["count"] == 3
    assert client.get("/clients?limit=0").status_code == 422
    assert client.get("/clients?limit=21").status_code == 422


def test_clients_ids_are_scorable():
    """La promesse de l'endpoint : ces identifiants marchent vraiment.
    Sans ce test, /clients pourrait renvoyer des identifiants que /predict
    refuse, et l'ajout serait pire qu'inutile."""
    for entry in client.get("/clients?limit=5").json()["clients"]:
        response = client.post(f"/predict?SK_ID_CURR={entry['SK_ID_CURR']}")
        assert response.status_code == 200, entry["SK_ID_CURR"]


def test_clients_age_is_plausible():
    for entry in client.get("/clients?limit=20").json()["clients"]:
        assert 18 <= entry["age_years"] <= 100
        # La sentinelle 365243 de DAYS_EMPLOYED ne doit jamais ressortir en années.
        assert entry["employment_years"] is None or 0 <= entry["employment_years"] <= 60


# --- La documentation elle-même ---
def test_openapi_documente_le_404():
    """Le code lève un 404 depuis toujours ; le schéma ne le disait pas."""
    schema = client.get("/openapi.json").json()
    assert "404" in schema["paths"]["/predict"]["post"]["responses"]


def test_openapi_propose_un_exemple():
    """Sans exemple, « Try it out » part sur un champ vide et tombe en 404."""
    schema = client.get("/openapi.json").json()
    param = schema["paths"]["/predict"]["post"]["parameters"][0]
    assert param["description"]
    assert param.get("examples") or param["schema"].get("examples")


def test_openapi_a_une_description():
    schema = client.get("/openapi.json").json()
    assert "0,47" in schema["info"]["description"]
    assert {t["name"] for t in schema["tags"]} == {"Scoring", "Données", "Santé"}
