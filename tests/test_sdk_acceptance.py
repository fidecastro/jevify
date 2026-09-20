"""INV-8's acceptance test: the unmodified typesafe-sdk, pointed at jevify, round-trips."""

from __future__ import annotations

import math
import threading
import time

import pytest
import uvicorn

from jevify.api.app import create_app
from jevify.domain.engine import Engine
from jevify.recipes import load_recipe
from tests.fakes.fake_backend import FakeBackend

pytestmark = pytest.mark.sdk

LN = math.log
SCRIPTED = {
    "dept": {"billing": LN(0.84), "support": LN(0.16)},
    "urgency": {"0": LN(0.05), "1": LN(0.86), "2": LN(0.09)},
    "churn": {"false": LN(0.108), "true": LN(0.892)},
}


@pytest.fixture
def served_jevify(fixtures):
    recipe = load_recipe(fixtures / "recipes" / "fake-vllm.yaml")
    app = create_app(Engine(FakeBackend(SCRIPTED), recipe), recipe)
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=0, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started:
        if time.monotonic() > deadline:
            raise RuntimeError("jevify server did not start")
        time.sleep(0.01)
    port = server.servers[0].sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(timeout=5)


def test_sdk_round_trip_all_three_types(served_jevify):
    typesafe = pytest.importorskip("typesafe_sdk")
    client = typesafe.TypeSafeClient(api_key="not-checked", base_url=served_jevify)
    response = client.system_one(
        {"subject": "Duplicate charge", "body": "I was billed twice."},
        {
            "dept": typesafe.Choice(
                instructions="Which team handles this?",
                criteria={"billing": "invoices and refunds", "support": None},
            ),
            "urgency": typesafe.Score(
                instructions="How urgent?", criteria=["not urgent", "soon", "critical"]
            ),
            "churn": typesafe.Noul(instructions="The customer threatens to leave."),
        },
    )
    assert response.model == "fake-decoder"
    dept = response.answers["dept"]
    assert dept.choice == "billing"
    assert round(dept.probabilities["billing"], 2) == 0.84
    assert 0.0 <= dept.confidence <= 1.0
    urgency = response.answers["urgency"]
    assert round(urgency.score, 2) == 1.04
    assert urgency.legend[1] == "soon"
    churn = response.answers["churn"]
    assert round(churn.noul, 3) == 0.892
    assert response.usage.input_tokens == 300 and response.usage.output_tokens == 3
    assert response.request_id.startswith("req_")
    models = client.models.list()
    assert models.models[0].name == "fake-decoder"
