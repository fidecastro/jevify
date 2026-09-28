"""Guard: `endpoint.extra_body` may never override a field jevify sets (ADR-0004 D2).

The client refuses a reserved key when the backend is built, from one registry,
`RESERVED_REQUEST_FIELDS`. A registry kept by hand drifts, so this guard drives
every adapter kind and dialect through the fakes the way an operator does
(probe, write the probe in, warm, evaluate), collects every top-level field that
reached a server, and fails naming any field the registry lacks.
"""

from __future__ import annotations

import asyncio

from jevify.adapters.endpoint.client import RESERVED_REQUEST_FIELDS
from jevify.compose import build_backend
from jevify.domain.questions import ChoiceQuestion, NoulQuestion, Option, ScoreQuestion, State
from jevify.recipes import apply_probe, load_recipe
from tests.fakes.fake_openai_server import Behaviour

STATE = State.from_jev("The customer wrote: I was charged twice and I am furious.")
CHOICE = ChoiceQuestion(
    id="dept",
    instructions="Which team handles this?",
    options=(Option("billing", "invoices and refunds"), Option("support", None)),
)
SCORE = ScoreQuestion(id="anger", instructions="How angry?", levels=("calm", "annoyed", "furious"))
NOUL = NoulQuestion(id="cancel", instructions="The customer threatens to cancel.")
ALL = [CHOICE, SCORE, NOUL]

# Five answer tokens, all inside the fake's top-5 cap, so every readout succeeds.
SCORES = {" yes": 0.0, " no": -1.0, " A": -0.5, " B": -1.2, " C": -1.5}
RAW_SCORES = {"yes": 0.0, "no": -1.0, "A": -0.5, "B": -1.2, "C": -1.5}

# Every adapter kind and every dialect, with the question types each kind answers. The
# probe also issues the request of every rung it measures.
RUNS = [
    (Behaviour(scores=SCORES), "fake-vllm.yaml", "auto", ALL),
    (Behaviour(dialect="llamacpp", grammar=True, scores=RAW_SCORES), "fake-raw.yaml", "auto", ALL),
    (
        Behaviour(dialect="generic", thinking_kwarg=None, thinking_default_on=False, scores=SCORES),
        "fake-vllm.yaml",
        "generic",
        ALL,
    ),
    (
        Behaviour(dialect="llamacpp", rerank_scores={"x": 0.5}),
        "fake-rerank.yaml",
        "auto",
        [CHOICE, NOUL],
    ),
    (Behaviour(embeddings={"x": [1.0, 0.0]}), "fake-embedding.yaml", "auto", [CHOICE]),
]


def _sent(server) -> set[str]:
    requests = server.requests + server.template_requests
    return {key for req in requests for key in req if key != "path"}


def test_every_field_the_adapters_send_is_in_the_reserved_registry(fixtures, fake_server_factory):
    unregistered: dict[str, set[str]] = {}
    for behaviour, recipe_name, dialect, questions in RUNS:
        server, http = fake_server_factory(behaviour)
        recipe = load_recipe(fixtures / "recipes" / recipe_name)
        recipe = recipe.model_copy(
            update={"endpoint": recipe.endpoint.model_copy(update={"dialect": dialect})}
        )
        capabilities = asyncio.run(build_backend(recipe, http=http).probe())
        backend = build_backend(apply_probe(recipe, capabilities), http=http)
        handle = asyncio.run(backend.warm(STATE))
        answers = asyncio.run(backend.evaluate(handle, questions))
        assert len(answers) == len(questions)
        missing = _sent(server) - RESERVED_REQUEST_FIELDS
        if missing:
            unregistered[f"{recipe_name} ({dialect})"] = missing
    assert not unregistered, (
        f"request fields not in RESERVED_REQUEST_FIELDS (jevify/adapters/endpoint/client.py): "
        f"{unregistered}. Add them there so endpoint.extra_body cannot override them."
    )
