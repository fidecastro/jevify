"""Live proof of INV-8: the unmodified typesafe-sdk against a running `jevify serve`.

Set JEVIFY_LIVE_SERVER to the server's origin (for example http://127.0.0.1:8600).
"""

from __future__ import annotations

import os

import pytest

SERVER = os.environ.get("JEVIFY_LIVE_SERVER")

pytestmark = [pytest.mark.live, pytest.mark.sdk]


@pytest.mark.skipif(not SERVER, reason="JEVIFY_LIVE_SERVER not set")
def test_sdk_against_live_jevify_server() -> None:
    typesafe = pytest.importorskip("typesafe_sdk")
    client = typesafe.TypeSafeClient(api_key="local", base_url=SERVER)
    response = client.system_one(
        "The customer wrote: I was charged twice this month and I am furious. "
        "If this is not fixed today I will cancel.",
        {
            "dept": typesafe.Choice(
                instructions="Which team should handle this?",
                criteria={"billing": "invoices and refunds", "technical": None, "sales": None},
            ),
            "churn": typesafe.Noul(instructions="The customer threatens to cancel."),
            "pleased": typesafe.Noul(instructions="The customer is pleased."),
        },
    )
    assert response.answers["dept"].choice == "billing"
    assert response.answers["churn"].noul > 0.5
    assert response.answers["pleased"].noul < 0.5
    assert response.usage.output_tokens == 3
    assert response.request_id.startswith("req_")
    assert client.models.list().models[0].name == response.model
