"""Guard (ADR-0003 D2): jevify's wire schema is a pinned mirror of the SDK's, field by field.

jevify may add optional `x_jevify` fields; it may never drop, rename or
change the required-ness of a field Jev's SDK defines.
"""

from __future__ import annotations

import pytest

from jevify.api import schema as ours

PAIRS = (
    "NoulCriteria",
    "NoulQuestion",
    "ChoiceQuestion",
    "ScoreQuestion",
    "SystemOneRequest",
    "NoulAnswer",
    "ChoiceAnswer",
    "ScoreAnswer",
    "Usage",
    "SystemOneResponse",
    "ModelMetadata",
    "ModelMetadataList",
    "ValidationError",
    "HTTPValidationError",
)
EXTENSION_FIELDS = {"x_jevify"}


def test_sdk_schema_mirror() -> None:
    theirs = pytest.importorskip("typesafe_sdk._schemas.models")
    drift: list[str] = []
    for name in PAIRS:
        sdk_model = getattr(theirs, name)
        our_model = getattr(ours, name)
        sdk_fields = {k: v.is_required() for k, v in sdk_model.model_fields.items()}
        our_fields = {k: v.is_required() for k, v in our_model.model_fields.items()}
        for field, required in sdk_fields.items():
            if field not in our_fields:
                drift.append(f"{name}.{field}: missing in jevify")
            elif our_fields[field] != required:
                drift.append(
                    f"{name}.{field}: required={required} in the SDK, {our_fields[field]} here"
                )
        for field in our_fields:
            if field not in sdk_fields and field not in EXTENSION_FIELDS:
                drift.append(f"{name}.{field}: not a Jev field and not under x_jevify")
    assert not drift, "jevify/api/schema.py drifted from typesafe-sdk:\n  " + "\n  ".join(drift)
