"""Load, dump and hash recipes. The hash is over canonical JSON of the parsed document."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from jevify.recipes.schema import Recipe


class RecipeError(ValueError):
    """A recipe could not be loaded, validated or rendered. The message names the field."""


def _describe(error: ValidationError) -> str:
    lines = []
    for item in error.errors():
        location = ".".join(str(part) for part in item["loc"]) or "<root>"
        lines.append(f"{location}: {item['msg']}")
    return "; ".join(lines)


def load_recipe(path: str | Path) -> Recipe:
    source = Path(path)
    try:
        document = yaml.safe_load(source.read_text(encoding="utf-8"))
    except OSError as exc:
        raise RecipeError(f"cannot read recipe {source}: {exc}") from exc
    except yaml.YAMLError as exc:
        raise RecipeError(f"recipe {source} is not valid YAML: {exc}") from exc
    if not isinstance(document, dict):
        raise RecipeError(f"recipe {source} must be a mapping at the top level")
    try:
        return Recipe.model_validate(document)
    except ValidationError as exc:
        raise RecipeError(f"recipe {source} is invalid: {_describe(exc)}") from exc


def dump_recipe(recipe: Recipe, path: str | Path) -> None:
    document = recipe.model_dump(mode="json", exclude_none=True)
    Path(path).write_text(
        yaml.safe_dump(document, sort_keys=False, allow_unicode=True, width=100),
        encoding="utf-8",
    )


def recipe_hash(recipe: Recipe) -> str:
    """SHA-256 of the canonical JSON form: independent of YAML layout, quoting and comments."""
    canonical = json.dumps(
        recipe.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def apply_probe(recipe: Recipe, capabilities: Any) -> Recipe:
    """Return the recipe with the probe's findings written in: capabilities, verified ids,
    the detected dialect, and the provenance status."""
    document = recipe.model_dump(mode="json")
    document["probe"] = capabilities.to_dict()
    if capabilities.answer_tokens:
        for group in document["answers"]["noul"].values():
            for token in group:
                token["id"] = capabilities.answer_tokens.get(token["text"], token.get("id"))
        for group in document["answers"]["identifiers"]:
            for token in group:
                token["id"] = capabilities.answer_tokens.get(token["text"], token.get("id"))
    if document.get("endpoint") and capabilities.dialect:
        document["endpoint"]["dialect"] = str(capabilities.dialect)
    if document["provenance"]["status"] == "draft":
        document["provenance"]["status"] = "probed"
    return Recipe.model_validate(document)
