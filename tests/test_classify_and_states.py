"""ADR-0003 D4 and D6: the classify convenience route and explicit state handles."""

from __future__ import annotations

import math
import time

from starlette.testclient import TestClient

from jevify.api.app import create_app
from jevify.api.translate import classify_to_systemone
from jevify.domain.engine import Engine
from jevify.recipes import load_recipe
from tests.fakes.fake_backend import FakeBackend

LN = math.log
SCRIPTED = {
    "billing": {"false": LN(0.1), "true": LN(0.9)},
    "spam": {"false": LN(0.97), "true": LN(0.03)},
    "category": {"billing": LN(0.8), "spam": LN(0.2)},
    "urgency": {"0": LN(0.2), "1": LN(0.8)},
}


def client_for(fixtures, ttl_s: float = 60.0):
    recipe = load_recipe(fixtures / "recipes" / "fake-vllm.yaml")
    backend = FakeBackend(SCRIPTED)
    app = create_app(Engine(backend, recipe), recipe, state_ttl_s=ttl_s)
    return TestClient(app), backend


def test_classify_boolean_builds_one_noul_per_category():
    request = classify_to_systemone(
        {
            "context": "I was billed twice.",
            "categories": {"billing": "about invoices or charges", "spam": None},
            "mode": "boolean",
        }
    )
    assert request["state"] == "I was billed twice."
    assert set(request["questions"]) == {"billing", "spam"}
    assert request["questions"]["billing"]["type"] == "noul"
    assert "about invoices or charges" in request["questions"]["billing"]["instructions"]
    assert request["questions"]["spam"]["type"] == "noul"


def test_classify_choice_and_score_modes():
    choice = classify_to_systemone(
        {"context": "x", "categories": ["billing", "spam"], "mode": "choice"}
    )
    assert list(choice["questions"]) == ["category"]
    assert choice["questions"]["category"]["type"] == "choice"
    assert list(choice["questions"]["category"]["criteria"]) == ["billing", "spam"]
    score = classify_to_systemone(
        {
            "context": "x",
            "categories": ["urgency"],
            "mode": "score",
            "levels": ["low", "high"],
        }
    )
    assert score["questions"]["urgency"]["type"] == "score"
    assert score["questions"]["urgency"]["criteria"] == ["low", "high"]


def test_classify_route_returns_per_category_results(fixtures):
    client, _ = client_for(fixtures)
    response = client.post(
        "/v1/classify",
        json={
            "context": "I was billed twice.",
            "categories": ["billing", "spam"],
            "mode": "boolean",
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert round(body["results"]["billing"]["probability"], 2) == 0.9
    assert body["results"]["billing"]["decision"] is True
    assert body["results"]["spam"]["decision"] is False
    assert body["results"]["spam"]["x_jevify"]["semantics"] == "readout"
    assert body["answers"]["billing"]["type"] == "noul"  # the Jev-shaped answers ride along


def test_states_handle_roundtrip(fixtures):
    client, backend = client_for(fixtures)
    created = client.post("/v1/states", json={"state": "I was billed twice."})
    assert created.status_code == 201
    handle = created.json()["state_id"]
    assert handle and created.json()["expires_in_s"] > 0
    response = client.post(
        "/v1/systemone",
        json={
            "state": "",
            "model": "jev-latest",
            "questions": {"spam": {"type": "noul", "instructions": "It is spam."}},
            "x_jevify": {"state_ref": handle},
        },
    )
    assert response.status_code == 200
    assert response.json()["x_jevify"]["state_id"] == handle
    assert len(backend.warmed) == 1  # warmed once at ingest, reused by the question


def test_expired_or_unknown_handle_fails_loudly(fixtures):
    client, _ = client_for(fixtures, ttl_s=0.05)
    handle = client.post("/v1/states", json={"state": "x"}).json()["state_id"]
    time.sleep(0.1)
    body = {
        "state": "",
        "model": "jev-latest",
        "questions": {"q": {"type": "noul", "instructions": "x"}},
        "x_jevify": {"state_ref": handle},
    }
    expired = client.post("/v1/systemone", json=body)
    assert expired.status_code == 404
    assert "expired" in expired.json()["error"]["message"]
    body["x_jevify"]["state_ref"] = "no-such-handle"
    assert client.post("/v1/systemone", json=body).status_code == 404
