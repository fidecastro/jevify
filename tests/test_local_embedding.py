"""The in-process embedder (marker `encoder`): needs torch, transformers and a download."""

from __future__ import annotations

import asyncio

import pytest

from jevify.compose import build_backend
from jevify.domain.questions import ChoiceQuestion, Option, State
from jevify.recipes import load_recipe

pytestmark = pytest.mark.encoder


def test_local_embedder_ranks_the_obvious_option(fixtures):
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    backend = build_backend(load_recipe(fixtures / "recipes" / "local-embedding.yaml"))
    caps = asyncio.run(backend.probe())
    assert caps.kind == "embedding" and any("dimension 384" in n for n in caps.notes)
    question = ChoiceQuestion(
        id="q",
        instructions="Which topic?",
        options=(
            Option("cooking", "recipes and kitchens"),
            Option("astronomy", "stars and planets"),
        ),
    )
    handle = asyncio.run(backend.warm(State.from_jev("How long should I boil an egg?")))
    [answer] = asyncio.run(backend.evaluate(handle, [question]))
    assert answer.semantics == "similarity"
    assert max(answer.logprobs, key=answer.logprobs.get) == "cooking"
    assert -1.0 <= answer.raw["cosine"]["cooking"] <= 1.0
