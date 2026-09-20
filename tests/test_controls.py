"""Permutation policy (ADR-0002 D5) and the scorecard controls (ADR-0005 D3)."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import yaml

from jevify.compose import build_backend
from jevify.domain.distribution import Distribution
from jevify.domain.questions import ChoiceQuestion, Option, State
from jevify.recipes import load_recipe
from tests.fakes.fake_openai_server import Behaviour

SUITES = Path(__file__).resolve().parents[1] / "suites"
STATE = State.from_jev("The customer wrote: keep my subscription, stop the newsletters.")


def run(coro):
    return asyncio.run(coro)


def test_cyclic_permutation_averages_out_position_bias(fixtures, fake_server_factory):
    # The fake scores the identifier token, not the option: pure position bias.
    server, http = fake_server_factory(Behaviour(scores={" A": 0.0, " B": -1.0}))
    recipe = load_recipe(fixtures / "recipes" / "fake-vllm.yaml")
    recipe = recipe.model_copy(
        update={
            "readout": recipe.readout.model_copy(
                update={"permutation": "cyclic", "permutation_calls": 2}
            )
        }
    )
    backend = build_backend(recipe, http=http)
    question = ChoiceQuestion(
        id="dept", instructions="Which team?", options=(Option("billing"), Option("support"))
    )
    [answer] = run(backend.evaluate(run(backend.warm(STATE)), [question]))
    assert answer.calls == 2
    assert answer.composition == "permuted"
    d = Distribution.from_logprobs(answer.logprobs)
    # Order (billing, support) gives billing 0.7311; order (support, billing) gives billing
    # 0.2689; the average is exactly one half.
    assert d.as_mapping()["billing"] == pytest.approx(0.5)
    orders = [req["messages"][-1]["content"].split("Options:\n")[1] for req in server.requests[1:]]
    assert orders[0].startswith("A. billing") and orders[1].startswith("A. support")


def test_permutation_none_is_a_single_call(fixtures, fake_server_factory):
    server, http = fake_server_factory(Behaviour(scores={" A": 0.0, " B": -1.0}))
    backend = build_backend(load_recipe(fixtures / "recipes" / "fake-vllm.yaml"), http=http)
    question = ChoiceQuestion(id="q", instructions="Pick.", options=(Option("x"), Option("y")))
    [answer] = run(backend.evaluate(run(backend.warm(STATE)), [question]))
    assert answer.calls == 1 and answer.composition == "single"


def test_eval_controls_shuffled_context_and_option_permutation(
    fixtures, fake_http_server_factory, tmp_path, capsys
):
    from jevify.cli import main

    live = fake_http_server_factory(
        Behaviour(scores={" A": 0.0, " B": -1.0, " C": -2.0}, max_context=100_000)
    )
    doc = yaml.safe_load((fixtures / "recipes" / "fake-vllm.yaml").read_text())
    doc["endpoint"]["base_url"] = live.base_url
    (tmp_path / "r.yaml").write_text(yaml.safe_dump(doc))
    code = main(
        [
            "eval",
            str(tmp_path / "r.yaml"),
            str(SUITES / "policy-29.jsonl"),
            "--runs-dir",
            str(tmp_path / "runs"),
            "--evidence-dir",
            str(tmp_path / "evidence"),
            "--controls",
        ]
    )
    assert code == 0
    report = json.loads(capsys.readouterr().out)
    summary = json.loads(Path(report["summary_json"]).read_text())
    controls = summary["controls"]
    # Shuffled context: each case answered against the next case's state; the fake ignores the
    # state, so agreement with the original label equals the main accuracy.
    assert controls["shuffled_context"]["cases"] == 29
    assert controls["shuffled_context"]["agreement_with_label"] == pytest.approx(
        summary["metrics"]["accuracy"]
    )
    # Option permutation: the fake always picks position A, so reversing the order flips
    # every selection: agreement is zero.
    assert controls["option_permutation"]["cases"] == 29
    assert controls["option_permutation"]["same_choice"] == pytest.approx(0.0)
    assert "shuffled_context" in Path(report["summary_md"]).read_text()
