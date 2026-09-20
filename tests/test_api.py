"""Jev's evaluate call served unchanged (ADR-0003): shapes, extension field, auth, errors."""

from __future__ import annotations

import math

from starlette.testclient import TestClient

from jevify.api.app import create_app
from jevify.domain.engine import Engine
from jevify.recipes import load_recipe
from tests.fakes.fake_backend import FakeBackend

LN = math.log
REQUEST = {
    "state": {"subject": "Duplicate charge", "body": "I was billed twice. Refund me today."},
    "model": "jev-latest",
    "questions": {
        "dept": {
            "type": "choice",
            "instructions": "Which team handles this?",
            "criteria": {"billing": "invoices and refunds", "support": None},
        },
        "urgency": {
            "type": "score",
            "instructions": "How urgent?",
            "criteria": ["not urgent", "soon", "critical"],
        },
        "churn": {"type": "noul", "instructions": "The customer threatens to leave."},
    },
}
SCRIPTED = {
    "dept": {"billing": LN(0.84), "support": LN(0.16)},
    "urgency": {"0": LN(0.05), "1": LN(0.86), "2": LN(0.09)},
    "churn": {"false": LN(0.108), "true": LN(0.892)},
}


def client_for(fixtures, api_key=None):
    recipe = load_recipe(fixtures / "recipes" / "fake-vllm.yaml")
    app = create_app(Engine(FakeBackend(SCRIPTED), recipe), recipe, api_key=api_key)
    return TestClient(app), recipe


def test_systemone_answers_match_the_sdk_shapes(fixtures):
    client, _ = client_for(fixtures)
    response = client.post("/v1/systemone", json=REQUEST)
    assert response.status_code == 200
    body = response.json()
    assert body["model"] == "fake-decoder"
    dept, urgency, churn = (
        body["answers"]["dept"],
        body["answers"]["urgency"],
        body["answers"]["churn"],
    )
    assert dept["type"] == "choice" and dept["choice"] == "billing"
    assert round(dept["probabilities"]["billing"], 2) == 0.84
    assert isinstance(dept["confidence"], float)
    assert urgency["type"] == "score" and round(urgency["score"], 2) == 1.04
    assert urgency["legend"] == {"0": "not urgent", "1": "soon", "2": "critical"}
    assert set(urgency["probabilities"]) == {"0", "1", "2"}
    assert churn["type"] == "noul" and round(churn["noul"], 3) == 0.892
    assert "confidence" not in churn
    assert body["usage"] == {"input_tokens": 300, "output_tokens": 3}


def test_x_jevify_carries_semantics_rung_and_request_context(fixtures):
    client, recipe = client_for(fixtures)
    body = client.post("/v1/systemone", json=REQUEST).json()
    ext = body["answers"]["dept"]["x_jevify"]
    assert ext["semantics"] == "readout" and ext["rung"] == "top_k"
    assert body["x_jevify"]["requested_model"] == "jev-latest"
    assert body["x_jevify"]["state_id"] == "state-1"
    assert len(body["x_jevify"]["recipe_hash"]) == 64


def test_request_id_header_is_present(fixtures):
    client, _ = client_for(fixtures)
    response = client.post("/v1/systemone", json=REQUEST)
    assert response.headers["x-typesafe-request-id"].startswith("req_")


def test_models_lists_the_recipe(fixtures):
    client, _ = client_for(fixtures)
    body = client.get("/v1/models").json()
    assert body["models"][0]["name"] == "fake-decoder"
    assert body["models"][0]["release_date"] == "2026-09-19"


def test_bearer_key_checked_only_when_configured(fixtures):
    client, _ = client_for(fixtures, api_key="s3cret")
    assert client.post("/v1/systemone", json=REQUEST).status_code == 401
    ok = client.post("/v1/systemone", json=REQUEST, headers={"Authorization": "Bearer s3cret"})
    assert ok.status_code == 200
    open_client, _ = client_for(fixtures)
    assert open_client.post("/v1/systemone", json=REQUEST).status_code == 200


def test_validation_error_is_fastapi_shaped(fixtures):
    client, _ = client_for(fixtures)
    bad = dict(REQUEST, questions={"q": {"type": "choice", "instructions": "x"}})  # no criteria
    response = client.post("/v1/systemone", json=bad)
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert isinstance(detail, list) and detail
    assert "criteria" in [str(part) for part in detail[0]["loc"]]
    assert set(detail[0]) >= {"loc", "msg", "type"}


def test_health_names_model_and_recipe(fixtures):
    client, recipe = client_for(fixtures)
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["model"] == "fake-decoder"
