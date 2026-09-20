# ADR-0001 — Founding choices: name, scope, licence, toolchain, conventions

- **Status:** accepted (2026-09-19). Founder ruling after the design
  discussion of 2026-09-19; adopts `docs/00-invariants.md` in the same commit.
- **Extends:** nothing. This is the first decision.

## Context

TypeSafe launched Jev on 2026-09-15: a hosted, closed-weight model that takes
program state and typed questions and returns distributions in one
non-autoregressive pass. Within days a dozen open reimplementations appeared.
Reading them, and reading the two local precursors, produced three
observations that shape this project.

1. The useful part of Jev is its interface and its inference strategy. Both
   are reproducible on open-weight models through a logit readout at an answer
   position, which is what most of the clones do.
2. The local precursors split cleanly. `vinnylarouge/jevlike` (MIT) is a tiny
   from-scratch option-attention scorer whose head is superseded by encoder
   layouts such as Laya's, but whose data format and evaluation controls are
   worth keeping. `fidecastro/decision-poc` measured Qwen3 embeddings,
   rerankers and a yes/no readout under hashed protocols and found that
   4B-class models solve a small policy suite, that every model fails an
   inverted instruction, and that relevance-trained models are not decision
   models. Its discipline is the standard here; its framing drifted and is not
   carried.
3. Most public numbers from that week cannot be reproduced from their
   repositories. Laya's benchmark document cites result files that are absent
   from its checkout after a history rewrite. TypeSafe's own evaluations use
   two frontier models' average as ground truth.

The founder's intent, fixed on 2026-09-19: a bring-your-own-model harness for
inference and use, with training out of scope; the Jev interface; multimodal
where the model allows; no context limit imposed by the harness; and, above
all, numbers that can be checked.

## Decision

### D1 — Name

The project, the repository, the folder and the Python package are **jevify**,
after the verb: to jevify a model is to make it answer typed questions in one
pass. Other names considered are listed under rejected alternatives.

### D2 — Scope

jevify is a harness (INV-1) for four kinds of user-supplied model (INV-2). It
serves Jev's interface (INV-8) and measures what it serves (INV-7). Training,
custom inference engines, hosting weights and any user interface beyond a CLI
and an API are non-goals (`docs/00-invariants.md` §6).

### D3 — Licence

MIT, since 2026-09-20. The founding choice was GPL-3.0; the founder relicensed once the harness existed, so that it can be embedded as a
library, in a Jev SDK adapter or in a proprietary service, without a
process-boundary argument. Every line in the tree at the time of the change
was the author's own work (the only ported code came from the author's
decision-poc), so no third-party consent was needed. Borrowed code keeps its
own notice: any encoder layout taken from Laya (Apache-2.0) or code from
jevlike (MIT) is credited in `THIRD_PARTY_LICENSES.md` when it enters the
tree. Copyleft sources may not enter: MIT cannot carry them.

### D4 — Toolchain

Python **3.12** as the floor: current enough for modern typing and `asyncio`,
old enough to be on every host the author runs. Dependencies are declared in
`pyproject.toml`; the core depends on nothing heavier than an HTTP client. Model libraries are
optional extras per adapter kind, so a user who only talks to endpoints never
installs PyTorch.

### D5 — Engineering principles

The four principles in `AGENTS.md`: Always Works, TDD at public seams, SOLID
with ports and one composition root, and one authoritative path per singular
process. They bind every change.

### D6 — Repository layout

```
jevify/            the package: core, ports, adapters, api, cli
tests/             tests at public seams; structure guards live here too
recipes/           committed recipe files, one per model per adapter kind
suites/            committed frozen suites, hashed
docs/              numbered documents; 00 is the invariants
docs/adr/          decisions, indexed in docs/adr/README.md
docs/evidence/     committed scorecard summaries and calibration evidence
runs/              raw run outputs, git-ignored
```

### D7 — Documentation conventions

Numbered documents carry an audience line and a status line. The ADR index's
status column is the ledger. A change to a public seam updates its document in
the same commit. Documents are written for a reader who arrives through their
coding agent: the README tells the reader what to ask the agent, and the
documents the agent reads say what is decided without ceremony.

### D8 — Ignore rules

The ignore file is the chokepoint for INV-10. It excludes virtual
environments, Python caches, the model cache and every weight file extension,
raw run outputs, and environment or token files. Recipes, suites and evidence
summaries are committed.

## Consequences

- The name binds the project to a competitor's product name. Accepted: the
  repository is private, the verb is the concept, and a rename is cheap.
- MIT lets anyone embed jevify, including in closed products, without giving
  anything back. Accepted (2026-09-20); the earlier GPL-3.0 reasoning, that
  jevify would only ever be used as a service over HTTP, stopped holding once
  the adapter kinds became importable modules.
- Python 3.12 excludes some older hosts. Accepted; the floor is revisited
  when a user needs 3.11.
- The optional-extras rule means the encoder, reranker and embedding adapters
  cannot be exercised by the base install; CI installs the extras it tests.

## Rejected alternatives

- **JevCake as the name.** Considered first; the founder kept it for another
  project.
- **Staying on GPL-3.0.** It would have kept jevify out of any proprietary
  caller. Reversed on 2026-09-20 in favour of MIT;
  Apache-2.0 was not taken because the author wanted the shortest permissive
  text and no patent clause to explain.
- **Python 3.10 floor**, which the precursors used. Rejected for the reason
  in D4.
