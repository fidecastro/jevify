# ADR-0004 — Recipes: the hashed file that is the only way a model is asked

- **Status:** accepted (2026-09-19).
- **Extends:** ADR-0002 (the recipe selects an adapter kind and records the
  probe). Implements INV-6; supports design criterion 3.

## Context

A model behaves differently under every template, answer-token choice,
identifier scheme and context budget. The precursors showed how much of an
outcome is the prompt: the same reranker with a policy moved into the query
changed its answers at every amount tested, and a chat template that opens a
thinking block or injects a date silently breaks the readout or the prefix
cache. If those choices live in code, two people running "the same model" get
different numbers and cannot say why. If they live in a hashed file, a
scorecard names the file and the numbers are attributable.

The founder's intent is that a tinkerer picks up a pretrained model and tries
it. The thing they write, share and compare is therefore not a model and not
code. It is the recipe.

## Decision

### D1 — A recipe is a file

One YAML file per model per adapter kind under `recipes/`, validated against a
schema in `jevify.recipes`, and hashed over its canonical serialization. The
hash is the recipe's identity in every answer and every scorecard.

### D2 — Fields

| Section | Holds |
|---|---|
| `model` | Identifier as the backend knows it, revision when known, the adapter kind, and for endpoints the dialect the probe detected. |
| `template` | For endpoints: how state, instructions and options are rendered into messages or a raw prompt; where the question goes; the answer-slot suffix; template arguments such as thinking off; whether assistant prefill is used. For encoders: the sequence layout and marker scheme. |
| `answers` | The answer tokens for `noul`, the identifier scheme for `choice`, the level tokens for `score`, each with the token ids the probe verified. |
| `readout` | The rung of the ladder to use or `auto`; the permutation policy and call count for `choice`. |
| `budgets` | Context limit, option count limit, and what the probe measured. |
| `calibration` | Absent, or a temperature table keyed by question type and option-count bucket with a pointer to its evidence (ADR-0005 D5). |
| `probe` | The capability list the probe returned, with its date and the backend version. |
| `provenance` | Who wrote it, when, and the scorecards it has been run under. |

### D3 — The recipe is the chokepoint for how a model is asked

No code path renders a prompt, chooses an answer token or sets a readout
method except by reading a recipe. A structure guard rejects string literals
that look like prompt fragments outside `jevify.recipes` and the fixtures
directory. The `ask`, `eval` and `serve` commands all take a recipe and
nothing else about the model.

### D4 — Recipes ship and are shared

The repository ships recipes for a small set of models on each adapter kind,
each with at least one committed scorecard. A contributed recipe is accepted
with its probe section filled and a scorecard under `docs/evidence/`. A
recipe without a scorecard is a draft and says so in `provenance`.

### D5 — The probe writes into the recipe

`jevify probe <recipe>` fills or refreshes the `probe` section and re-hashes.
A recipe whose probe is older than its backend's recorded version is flagged
stale by `eval` and `serve`, and they refuse to run it unless told to.

## Consequences

- A tinkerer's contribution is a file plus a scorecard, which is small,
  reviewable and comparable.
- Prompt engineering becomes versioned and attributable. Two recipes for one
  model can be scored against each other.
- The structure guard for prompt literals will have false positives; they are
  resolved by moving the string into a recipe or a fixture, not by widening
  the allowlist.
- The recipe hash changes whenever the probe refreshes, so scorecards must
  record the hash at run time, not by name.

## Rejected alternatives

- **Python objects as recipes.** Not hashable in a stable way, not shareable
  without code, and they invite ad-hoc prompt strings.
- **Per-model adapter classes.** A model would need code to be tried; that is
  the opposite of bring-your-own-model.
- **Storing calibration outside the recipe.** Would separate a model from the
  only thing that makes its numbers calibrated; the table travels with the
  recipe and its evidence is committed separately.
