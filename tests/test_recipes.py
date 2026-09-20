"""Recipes: the hashed file that is the only way a model is asked (ADR-0004, INV-6)."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from jevify.domain.questions import ChoiceQuestion, NoulQuestion, Option, ScoreQuestion, State
from jevify.recipes import (
    RecipeError,
    load_recipe,
    recipe_hash,
    render_prefix,
    render_question,
)

FIXTURES = Path(__file__).parent / "fixtures" / "recipes"


def test_load_valid_fixture() -> None:
    recipe = load_recipe(FIXTURES / "fake-vllm.yaml")
    assert recipe.model.name == "fake-decoder"
    assert recipe.endpoint.dialect == "vllm"
    assert recipe.template.mode == "messages"
    assert recipe.answers.noul["true"][0].id == 14452
    assert len(recipe.answers.identifiers) == 4


def test_invalid_recipe_names_field(tmp_path: Path) -> None:
    doc = yaml.safe_load((FIXTURES / "fake-vllm.yaml").read_text())
    doc["readout"]["rung"] = "telepathy"
    path = tmp_path / "bad.yaml"
    path.write_text(yaml.safe_dump(doc))
    with pytest.raises(RecipeError, match=r"readout\.rung"):
        load_recipe(path)


def test_schema_version_required(tmp_path: Path) -> None:
    doc = yaml.safe_load((FIXTURES / "fake-vllm.yaml").read_text())
    del doc["schema_version"]
    path = tmp_path / "noversion.yaml"
    path.write_text(yaml.safe_dump(doc))
    with pytest.raises(RecipeError, match="schema_version"):
        load_recipe(path)


def test_hash_ignores_yaml_formatting(tmp_path: Path) -> None:
    original = load_recipe(FIXTURES / "fake-vllm.yaml")
    doc = yaml.safe_load((FIXTURES / "fake-vllm.yaml").read_text())
    reordered = tmp_path / "reordered.yaml"
    # Different key order, different quoting, a comment: same recipe.
    reordered.write_text("# reordered\n" + yaml.safe_dump(doc, sort_keys=True, default_style='"'))
    assert recipe_hash(load_recipe(reordered)) == recipe_hash(original)
    doc["readout"]["top_k"] = 5
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(doc))
    assert recipe_hash(load_recipe(changed)) != recipe_hash(original)
    assert len(recipe_hash(original)) == 64


def test_render_messages_mode_puts_question_last_with_kwargs() -> None:
    recipe = load_recipe(FIXTURES / "fake-vllm.yaml")
    prefix = render_prefix(recipe, State.from_jev("The customer wants a refund."))
    question = render_question(
        recipe, NoulQuestion(id="refund", instructions="The customer asks for money back.")
    )
    assert prefix.mode == "messages"
    assert prefix.system == "You answer typed questions about the state with a single token."
    assert prefix.text == "State:\nThe customer wants a refund.\n\n"
    assert prefix.kwargs == {"thinking": False}
    assert question.text == "Question: The customer asks for money back.\nAnswer yes or no."
    assert question.labels == ("false", "true")
    assert [t.id for t in question.tokens["true"]] == [14452, 16520]
    assert [t.id for t in question.tokens["false"]] == [1119, 3567]


def test_render_raw_mode_ends_at_answer_slot() -> None:
    recipe = load_recipe(FIXTURES / "fake-raw.yaml")
    prefix = render_prefix(recipe, State.from_jev("a duplicate invoice"))
    question = render_question(recipe, NoulQuestion(id="dup", instructions="It is a duplicate."))
    assert prefix.mode == "raw"
    assert prefix.text.startswith("<|im_start|>system\n")
    assert prefix.text.endswith("<Query>: a duplicate invoice\n")
    # The suffix carries everything after the state, ending exactly at the answer slot.
    assert question.text.startswith("\n<Instruct>: It is a duplicate.")
    assert question.text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")


def test_render_choice_uses_single_token_identifiers() -> None:
    recipe = load_recipe(FIXTURES / "fake-vllm.yaml")
    q = ChoiceQuestion(
        id="dept",
        instructions="Which team handles this?",
        options=(Option("billing", "invoices and refunds"), Option("technical", None)),
    )
    rendered = render_question(recipe, q)
    assert rendered.text == (
        "Question: Which team handles this?\nOptions:\n"
        "A. billing: invoices and refunds\nB. technical\n"
        "Answer with the identifier of the best option."
    )
    assert rendered.labels == ("billing", "technical")
    assert [t.id for t in rendered.tokens["billing"]] == [334]
    assert [t.id for t in rendered.tokens["technical"]] == [406]


def test_render_choice_honours_option_order() -> None:
    recipe = load_recipe(FIXTURES / "fake-vllm.yaml")
    q = ChoiceQuestion(id="q", instructions="Pick.", options=(Option("x", None), Option("y", None)))
    rendered = render_question(recipe, q, order=(1, 0))
    assert rendered.text.splitlines()[2:4] == ["A. y", "B. x"]
    assert rendered.labels == ("x", "y")
    assert rendered.tokens["y"][0].id == 334
    assert rendered.tokens["x"][0].id == 406


def test_render_fails_loudly_beyond_identifier_alphabet() -> None:
    recipe = load_recipe(FIXTURES / "fake-vllm.yaml")
    q = ChoiceQuestion(
        id="many", instructions="Pick.", options=tuple(Option(f"o{i}", None) for i in range(5))
    )
    with pytest.raises(RecipeError, match="4 identifiers"):
        render_question(recipe, q)


def test_render_score_levels_and_noul_criteria() -> None:
    recipe = load_recipe(FIXTURES / "fake-vllm.yaml")
    score = render_question(
        recipe, ScoreQuestion(id="urg", instructions="How urgent?", levels=("low", "high"))
    )
    assert score.text == (
        "Question: How urgent?\nLevels:\nA. low\nB. high\n"
        "Answer with the identifier of the level that applies."
    )
    assert score.labels == ("0", "1")
    assert score.tokens["1"][0].id == 406
    noul = render_question(
        recipe,
        NoulQuestion(
            id="n", instructions="Spam?", true_criterion="unsolicited", false_criterion="legit"
        ),
    )
    assert "Spam?" in noul.text


def test_render_structured_state_deterministically() -> None:
    recipe = load_recipe(FIXTURES / "fake-vllm.yaml")
    state = State.from_jev({"subject": "Refund", "body": "Charged twice"})
    prefix = render_prefix(recipe, state)
    assert prefix.text == 'State:\n{\n  "subject": "Refund",\n  "body": "Charged twice"\n}\n\n'
    assert render_prefix(recipe, state).text == prefix.text


def test_render_choice_per_option_yields_one_yes_no_part_per_option() -> None:
    recipe = load_recipe(FIXTURES / "fake-raw.yaml")  # choice_strategy: per_option
    q = ChoiceQuestion(
        id="dept",
        instructions="Which team handles this?",
        options=(Option("billing", "invoices"), Option("support", None)),
    )
    rendered = render_question(recipe, q)
    assert rendered.composition == "per_option"
    assert rendered.labels == ("billing", "support")
    assert [part.label for part in rendered.parts] == ["billing", "support"]
    first = rendered.parts[0]
    assert first.text.startswith(
        "\n<Document>: Which team handles this? The answer is: billing - invoices"
    )
    assert first.text.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    assert first.labels == ("false", "true")
    assert [t.text for t in first.tokens["true"]] == ["yes"]


def test_render_identifier_choice_is_a_single_part() -> None:
    recipe = load_recipe(FIXTURES / "fake-vllm.yaml")
    q = ChoiceQuestion(id="q", instructions="Pick.", options=(Option("x"), Option("y")))
    rendered = render_question(recipe, q)
    assert rendered.composition == "single"
    assert len(rendered.parts) == 1
    assert rendered.parts[0].labels == ("x", "y")
