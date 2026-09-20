"""The probe (ADR-0002 D3): measure what a backend can do, and write it into the recipe."""

from __future__ import annotations

import asyncio

import pytest
import yaml

from jevify.cli import main
from jevify.compose import build_backend
from jevify.recipes import load_recipe, recipe_hash
from tests.fakes.fake_openai_server import Behaviour

SCORES = {" yes": 0.0, " no": -1.5, "yes": -3.5, "no": -3.8, " A": -1.0, " B": -1.2}
RAW_SCORES = {"yes": 0.0, "no": -1.5, "Yes": -3.5, "No": -3.8, "A": -1.0, "B": -1.2}


def run(coro):
    return asyncio.run(coro)


def probe_with(fixtures, factory, behaviour, recipe_name="fake-vllm.yaml"):
    server, http = factory(behaviour)
    recipe = load_recipe(fixtures / "recipes" / recipe_name)
    caps = run(build_backend(recipe, http=http).probe())
    return server, caps


def test_probe_detects_dialect_and_version(fixtures, fake_server_factory):
    _, caps = probe_with(fixtures, fake_server_factory, Behaviour(scores=SCORES))
    assert caps.dialect == "vllm"
    assert caps.backend_version == "0.99.0-fake"
    _, caps = probe_with(
        fixtures,
        fake_server_factory,
        Behaviour(dialect="llamacpp", scores=RAW_SCORES),
        recipe_name="fake-raw.yaml",
    )
    assert caps.dialect == "llamacpp"
    assert caps.backend_version.startswith("b0.99")


def test_probe_verifies_single_token_answers(fixtures, fake_server_factory):
    _, caps = probe_with(fixtures, fake_server_factory, Behaviour(scores=SCORES))
    assert caps.token_ids_verified is True
    assert caps.answer_tokens[" yes"] == 14452
    assert caps.answer_tokens[" B"] == 406
    assert caps.multi_token_answers == ()

    _, caps = probe_with(
        fixtures, fake_server_factory, Behaviour(scores=SCORES, multi_token_answers={" B"})
    )
    assert caps.multi_token_answers == (" B",)


def test_probe_detects_thinking_not_disabled(fixtures, fake_server_factory):
    _, caps = probe_with(fixtures, fake_server_factory, Behaviour(scores=SCORES))
    assert caps.template_kwargs_honored is True
    # A template whose switch is not the recipe's kwarg leaves thinking on.
    _, caps = probe_with(
        fixtures, fake_server_factory, Behaviour(scores=SCORES, thinking_kwarg="enable_thinking")
    )
    assert caps.template_kwargs_honored is False


def test_probe_detects_pre_bias_logprobs(fixtures, fake_server_factory):
    _, caps = probe_with(fixtures, fake_server_factory, Behaviour(scores=SCORES))
    assert caps.logprobs_post_bias is False
    _, caps = probe_with(
        fixtures, fake_server_factory, Behaviour(scores=SCORES, logprobs_mode="processed")
    )
    assert caps.logprobs_post_bias is True


def test_probe_detects_logprob_token_ids_ignored(fixtures, fake_server_factory):
    _, caps = probe_with(fixtures, fake_server_factory, Behaviour(scores=SCORES))
    assert "named_token_logprobs" in caps.rungs
    _, caps = probe_with(
        fixtures, fake_server_factory, Behaviour(scores=SCORES, supports_logprob_token_ids=False)
    )
    assert "named_token_logprobs" not in caps.rungs
    assert "top_k" in caps.rungs


def test_probe_records_top_k_cap_and_max_context(fixtures, fake_server_factory):
    _, caps = probe_with(fixtures, fake_server_factory, Behaviour(scores=SCORES, top_k_cap=7))
    assert caps.top_k_cap == 7
    assert caps.max_context == 512
    _, caps = probe_with(
        fixtures,
        fake_server_factory,
        Behaviour(dialect="llamacpp", scores=RAW_SCORES, max_context=4096),
        recipe_name="fake-raw.yaml",
    )
    assert caps.max_context == 4096


def test_probe_measures_cache_block_granularity(fixtures, fake_server_factory):
    _, caps = probe_with(
        fixtures,
        fake_server_factory,
        Behaviour(scores=SCORES, max_context=100_000, block_tokens=64),
    )
    assert caps.cache is not None
    assert caps.cache.block_tokens == 64
    assert caps.cache.cached_tokens is not None and caps.cache.cached_tokens > 0


def test_probe_reports_no_cache_evidence_when_server_reports_none(fixtures, fake_server_factory):
    _, caps = probe_with(
        fixtures,
        fake_server_factory,
        Behaviour(scores=SCORES, max_context=100_000, report_cached_tokens=False),
    )
    assert caps.cache is not None
    assert caps.cache.cached_tokens is None
    assert caps.cache.block_tokens is None


def test_probe_writes_into_recipe_and_rehashes(
    fixtures, fake_http_server_factory, tmp_path, capsys
):
    live = fake_http_server_factory(Behaviour(scores=SCORES, max_context=100_000))
    doc = yaml.safe_load((fixtures / "recipes" / "fake-vllm.yaml").read_text())
    doc["endpoint"]["base_url"] = live.base_url
    doc["endpoint"]["dialect"] = "auto"
    doc["answers"]["noul"]["true"][0].pop("id")  # the probe fills verified ids
    path = tmp_path / "probe-me.yaml"
    path.write_text(yaml.safe_dump(doc))
    before = recipe_hash(load_recipe(path))

    assert main(["probe", str(path)]) == 0
    out = capsys.readouterr().out
    after = load_recipe(path)
    assert after.probe is not None
    assert after.probe["dialect"] == "vllm"
    assert after.probe["probed_at"]
    assert after.endpoint.dialect == "vllm"
    assert after.provenance.status == "probed"
    assert after.answers.noul["true"][0].id == 14452
    assert recipe_hash(after) != before
    assert before[:12] in out and recipe_hash(after)[:12] in out


def test_probe_refuses_when_no_readout_is_possible(fixtures, fake_server_factory):
    server, http = fake_server_factory(Behaviour(scores=SCORES, top_k_cap=0))
    recipe = load_recipe(fixtures / "recipes" / "fake-vllm.yaml")
    with pytest.raises(Exception, match="no readout"):
        run(build_backend(recipe, http=http).probe())


def test_probe_accepts_a_closed_empty_think_block_as_thinking_off(fixtures, fake_server_factory):
    """Qwen3.5-style templates spell thinking off as `<think>\\n\\n</think>\\n\\n`."""
    _, caps = probe_with(
        fixtures, fake_server_factory, Behaviour(scores=SCORES, thinking_off_style="empty_block")
    )
    assert caps.template_kwargs_honored is True


def test_probe_reads_a_token_granular_cache_with_a_reevaluated_tail(fixtures, fake_server_factory):
    """llama.cpp reports every prompt token cached except a few trailing ones it recomputes."""
    _, caps = probe_with(
        fixtures,
        fake_server_factory,
        Behaviour(
            dialect="llamacpp",
            scores=RAW_SCORES,
            max_context=100_000,
            block_tokens=1,
            cache_tail_tokens=4,
        ),
        recipe_name="fake-raw.yaml",
    )
    assert caps.cache is not None and caps.cache.block_tokens == 1
    assert any("last 4 prompt tokens" in note for note in caps.notes)


def test_probe_names_a_grammar_that_does_not_constrain_probabilities(fixtures, fake_server_factory):
    from jevify.ports.backend import Rung

    _, caps = probe_with(
        fixtures,
        fake_server_factory,
        Behaviour(dialect="llamacpp", scores=RAW_SCORES, grammar=False),
        recipe_name="fake-raw.yaml",
    )
    assert Rung.GRAMMAR not in caps.rungs
    assert any("grammar rung is not proven" in note for note in caps.notes)


def test_probe_measures_whether_a_warm_prefix_is_reused_by_a_question(
    fixtures, fake_server_factory
):
    """A hybrid model on llama.cpp resumes only from a checkpoint at the end of an earlier
    prompt. A messages-mode warm ends inside the template's assistant turn, so no question
    prompt extends it; a raw-mode warm ends at the state boundary and every question does."""
    _, caps = probe_with(
        fixtures,
        fake_server_factory,
        Behaviour(
            dialect="llamacpp",
            scores=RAW_SCORES,
            max_context=100_000,
            block_tokens=1,
            checkpoint_reuse_only=True,
        ),
        recipe_name="fake-raw.yaml",
    )
    assert caps.cache is not None
    assert caps.cache.warm_prefix_tokens and caps.cache.warm_reuse_tokens
    assert caps.cache.warm_reuse_tokens >= caps.cache.warm_prefix_tokens - 2
    _, caps = probe_with(
        fixtures,
        fake_server_factory,
        Behaviour(scores=SCORES, max_context=100_000, block_tokens=1, checkpoint_reuse_only=True),
        recipe_name="fake-vllm.yaml",
    )
    assert caps.cache is not None and caps.cache.warm_reuse_tokens == 0
    assert any("warm prefix is not reused" in note for note in caps.notes)
