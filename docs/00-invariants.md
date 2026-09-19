# 00 — Invariants

> **Audience:** everyone who changes this repository, human or agent.
> **Status:** normative. This document ranks above every ADR. An ADR that
> contradicts an invariant is void from the moment the contradiction is
> noticed. Changing an invariant is an amendment to this document, dated,
> with the ADR that motivated it (§7).
> **Adopted:** 2026-09-19.

## 1. Purpose

jevify makes a pretrained model that you already have answer typed questions
over a shared state, in one pass, in the shape that TypeSafe's Jev made
familiar, and it measures how well that model does it. It is a harness. It is
named after the verb: to jevify a model is to give it a recipe that turns it
into a typed decision engine without training it.

Two things motivate the project. The useful part of Jev is its interface and
its inference strategy, not a closed checkpoint: state in, typed questions
in, distributions out, no generation loop. That part is reproducible on
open-weight models. And the week Jev launched produced a dozen open
reimplementations whose numbers could not be reproduced from their
repositories. jevify is meant to be the one whose numbers can.

## 2. Vocabulary

These words have one meaning here. Documents, code and tests use them and no
synonyms.

| Term | Meaning |
|---|---|
| **State** | The shared input every question is asked about: text, images, video frames, or a structured document rendered to text. |
| **Question** | A typed query over the state. Types: **choice** (one of N options, distribution over N), **score** (a position on an ordered rubric, distribution over levels plus its expectation), **noul** (probability that a statement holds). |
| **Answer** | A question's result: the selected value, the full distribution, a confidence, and a semantics label. |
| **Semantics label** | What the numbers in an answer mean. One of `readout`, `relevance`, `similarity`, `calibrated`. Defined in §5. |
| **Readout** | Obtaining a distribution from a model without generating text: reading logits or log-probabilities at an answer position, or a classifier head's outputs. |
| **Adapter** | The implementation that talks to one kind of model. Four kinds: endpoint-served decoder, in-process encoder classifier, reranker, embedding. |
| **Port** | The one interface the core calls; every adapter implements it. |
| **Probe** | The connect-time check that measures what an adapter and its model can actually do, and records it. |
| **Recipe** | The file that says how one model is asked: adapter kind, template, answer tokens, readout method, budgets, calibration table, probe results. |
| **Warm** | The phase that encodes the state so that later questions reuse it. |
| **Evaluate** | The phase that answers questions against a warmed state. |
| **Suite** | A frozen, hashed set of cases with expected answers. |
| **Scorecard** | The reproducible record of one recipe run against one suite. |
| **Backend** | The server or library that runs the model: an OpenAI-compatible endpoint, or an in-process library. |

## 3. Invariants

**INV-1 — jevify is a harness, not a model.** The repository owns no weights
and trains nothing. Every model is brought by the user. A change that adds a
training loop, a checkpoint, or a dependency on a specific model's weights
violates this invariant.

**INV-2 — Bring your own model, of four kinds.** Endpoint-served decoders,
in-process encoder classifiers, rerankers and embedding models all answer the
same questions through the same port. The core never branches on adapter
kind; it branches on the capability list the probe returns.

**INV-3 — One pass per question.** No question ever costs more than one
sampled token, and most cost none. The harness never runs a generation loop,
never asks a model to write an answer, and never parses model text into an
answer. A readout that cannot be obtained without generating is not a readout
and is not shipped.

**INV-4 — Every answer carries a semantics label.** The harness never emits a
bare number. The label says what the number is (§5). Code that drops, defaults
or guesses the label is a defect.

**INV-5 — Questions are isolated.** No question sees another question's text
or answer. Each question is its own branch off the shared state. A recipe may
declare a joint mode explicitly, in which case the answers carry that fact.

**INV-6 — A model is asked only through a recipe.** There is no ad-hoc prompt
anywhere in a code path. The recipe is the chokepoint for how a model is
addressed, and it is a hashed file. Two runs with the same recipe hash, the
same suite hash and the same backend revision asked the model the same way.

**INV-7 — Every number is reproducible.** A number that appears in any
document carries the recipe hash, the suite hash, the backend and model
revisions, the software versions, and points to the per-decision raw output it
was computed from. Summaries are derived from raw outputs by committed code,
never typed in. A number without a run does not exist.

**INV-8 — Jev's interface is the primary interface.** The request and response
shapes of Jev's evaluate call are served unchanged, extensions live in a
namespaced field that Jev's SDK ignores, and the official SDK pointed at
jevify with only its base URL changed is the acceptance test.

**INV-9 — The harness imposes no context limit and never truncates.** Limits
come from the backend and the probe reports them. Input that exceeds a limit
fails loudly with the limit named. Silent truncation, of state or of options,
is a defect.

**INV-10 — Nothing large or secret enters the tree.** Weights, caches, raw run
outputs and credentials stay out of git; the ignore file is the chokepoint for
that rule. Recipes, suites and evidence summaries are committed.

## 4. Design criteria

The criteria a design must satisfy, adopted after the discussion recorded in
ADR-0001. They constrain ADRs; they are not themselves invariants and may be
revised by an ADR that says why.

1. **Harness, not model.** Restates INV-1.
2. **Four adapter kinds behind one port, with a capability probe.** Restates INV-2. Adapters are added, never privileged.
3. **Recipes are the unit of configuration and sharing.** A tinkerer tries a model by writing or copying a recipe, probing, evaluating and reading the scorecard. Recipes are what the community exchanges and compares.
4. **Jev-compatible API, convenience layer on top.** The context-plus-categories-to-booleans-or-scores form is a thin route that maps onto the Jev-shaped call, never a second implementation.
5. **Semantics on every answer.** Restates INV-4.
6. **Two-phase execution.** Warm the state, then evaluate. One request per question, concurrent where the backend shares the state's cache, sequential where it does not. The probe decides which.
7. **Multimodal wherever the model kind allows it.** Images and video frames are state parts. An adapter that cannot take them says so in its capability list rather than dropping them.
8. **Evaluation ships in version one.** Frozen suites, hashed protocols, scorecards, committed evidence. The scorecard is the product as much as the server is.
9. **Calibration is post-hoc, per recipe, off by default.** A temperature table fitted on the user's held-out labels, attached to the recipe, and reported with its own evidence. Until then the label is `readout`, not `calibrated`.
10. **Non-goals for version one:** training, custom inference engines, hosting models, any user interface beyond the CLI and the API.

## 5. Score semantics contract

This section is the claims contract. No document, README line, scorecard or
API response may claim more than it permits.

| Label | What the number is | Required to emit it |
|---|---|---|
| `readout` | A distribution taken from a model at an answer position or from a classifier head, normalized over the answer set. It is the model's belief under this recipe. It is **not** a probability that the answer is correct. | Any readout-capable adapter. |
| `relevance` | An independent per-option score from a reranker's relevance objective. Options do not compete; several can be high or all low. | The reranker adapter. |
| `similarity` | A cosine similarity between an instruction-conditioned state vector and an option vector. Pertinence, not decision. | The embedding adapter. |
| `calibrated` | A `readout` transformed by a temperature table fitted on held-out labeled decisions for this recipe, with the fit's evidence (log loss, Brier score, expected calibration error, reliability plot) committed under `docs/evidence/`. | A fitted table in the recipe plus its committed evidence. Never by default. |

Rules that follow:

- **Confidence** is a statistic of the returned distribution: one minus its normalized entropy for choice and score, distance from one half for noul. It is not an independent estimate that the decision is right, and copy may not describe it as one.
- **"Faster"** may be claimed only against a measured direct call on the same backend, same model, same state, with the measurement in a scorecard.
- **"Accuracy"** names its suite and the suite's hash. A suite whose expected answers came from a teacher model measures agreement with that teacher, and copy says so.
- **A fine-tuned result is never compared with a zero-shot result in one table** without the words fine-tuned and zero-shot in the row.
- **"Cannot hallucinate"** is a statement about type validity, which INV-3 gives by construction, and not about correctness. Copy does not use the phrase.
- **An untested path is named.** A scorecard, a README and a release note say which adapter kinds, backends and modalities were exercised and which were not.

## 6. Non-goals, stated so they can be cited

- Training a model, fine-tuning a model, or distilling one into another. A future project may do this on top of jevify's recipes and suites; it is not jevify.
- A custom inference engine, a patched vLLM, a diffusion answer-slot server. jevify consumes engines; it does not build them.
- Hosting or distributing weights.
- A graphical interface.
- Being an agent, a workflow engine or a router. jevify answers questions; deciding what to do with the answers belongs above it.

## 7. Amendments

An invariant changes only by an ADR whose title begins "Amend INV-n", accepted
in the ADR index, followed by an edit here that records the date, the ADR
number and the previous wording. The edit and the ADR land in the same commit.

| Date | Invariant | ADR | Change |
|---|---|---|---|
| 2026-09-19 | all | ADR-0001 | Adopted. |

## 8. Document map

| Document | Holds |
|---|---|
| `docs/00-invariants.md` | This file: purpose, vocabulary, invariants, design criteria, claims contract. |
| `docs/adr/README.md` | The ADR index. Its status column is the ledger. |
| `docs/adr/0001-…` onward | One decision each: context, decision, consequences, rejected alternatives. |
| `docs/evidence/` | Committed scorecard summaries and calibration evidence, each derived from a recorded run. |
| `AGENTS.md` | How work is done: the four engineering principles and house rules. |
| `recipes/` | Recipe files, one per model per adapter kind, hashed. |
| `suites/` | Frozen evaluation suites, hashed. |
