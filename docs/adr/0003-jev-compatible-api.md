# ADR-0003 — Jev's evaluate call is the API; extensions are namespaced; the SDK is the acceptance test

- **Status:** accepted (2026-09-19); D8 added 2026-09-23: `confidence` is
  Jev's peak statistic, amending `docs/00-invariants.md` §5.
- **Extends:** ADR-0001 D2. Implements INV-8; constrained by INV-4, INV-9.

## Context

Jev exposes one call: a model name, a state, and a map of question ids to
typed questions (`choice` with criteria, `score` with rubric levels, `noul`
with a statement) come in; a map of answers, each with the selected value, a
probability distribution and a confidence, plus usage, goes out. Laya copied
that shape almost field for field, which shows how portable it is. TypeSafe
publishes MIT-licensed SDKs for Python and JavaScript
([typesafe-sdk-python](https://github.com/typesafe-ai/typesafe-sdk-python)),
whose request and response models are the most precise public statement of
the wire format.

Jev's state is text and its call is stateless: the server caches implicitly.
jevify needs two things Jev lacks, multimodal state parts and an explicit
ingest for the ingest-once, query-many pattern, and it must add them without
breaking a client that only knows Jev.

## Decision

### D1 — The primary route

`POST /v1/systemone` accepts and returns exactly the shapes the official SDK
sends and expects. Field names, nesting, types and the three question types
are Jev's. The acceptance test is the official Python SDK, unmodified, with
only its base URL pointed at jevify, exercising all three question types
against a fixture backend.

### D2 — The SDK models are a pinned mirror

Under the chokepoint principle, jevify does not hand-maintain a copy of Jev's
schema. A test loads the SDK's request and response models and compares them
field by field with jevify's, so that a schema change on either side turns a
test red. The SDK version is pinned in the test's requirements and bumped
deliberately.

### D3 — Extensions live under one namespaced field

Everything jevify adds to a response goes under `x_jevify`, which Jev's SDK
ignores: the semantics label (INV-4), the readout method and rung, the
permutation policy and call count, the recipe hash, the backend and model
revisions, the off-menu mass when visible, and optionally the raw
logprobs. Everything jevify accepts beyond Jev's request goes under the same
key on the request.

### D4 — Optional ingest

`POST /v1/states` accepts a state, warms it, and returns a handle. The
primary route accepts `x_jevify.state_ref` in place of `state`. Without it,
the primary route warms implicitly and stays stateless, exactly as Jev.
Handles expire on a schedule the recipe declares; an expired handle fails
loudly.

### D5 — Multimodal state

A state may be a string, a structured document rendered to text, or a list of
content parts in the shape the chat endpoints use: text, image, video frames.
An adapter whose capability list lacks a modality rejects the request with the
modality named (INV-9); it never drops the part.

### D6 — The convenience route

`POST /v1/classify` takes a context, a list of categories with optional
descriptions, and a mode: `boolean` (one `noul` per category), `score` (one
`score` per category with the caller's rubric), or `choice` (one `choice`
over all categories). It builds a Jev-shaped request and calls the same
function the primary route calls. It is a translation, never a second
implementation, and a structure guard forbids it from touching the port.

### D7 — The server is thin

`jevify serve <recipe>` runs the HTTP API in one process over the library.
The library is the product; the server is one caller of it, and a program
that imports jevify gets identical behaviour without HTTP.

### D8 — `confidence` is Jev's statistic, computed Jev's way

Jev's `confidence` on a `choice` or `score` answer is
`(n × peak − 1) / (n − 1)`, where `n` is the number of options or levels and
`peak` is the largest probability: 0 for a uniform distribution, 1 for a
certain one. This is TypeSafe's published definition
(<https://docs.typesafe.ai/confidence>, worked example
`[0.90, 0.06, 0.04] → 0.85`), and it is what the thresholds TypeSafe
recommends to clients (act above 0.9, review between 0.5 and 0.9, route to
a human below 0.5) are calibrated against. Jev's `noul` answers carry no
confidence.

jevify computes exactly this in the one statistics chokepoint
(`jevify/domain/distribution.py`) and nowhere else. Until 2026-09-23 it
served one minus the normalized entropy under Jev's field name, which is
systematically lower (`[0.90, 0.06, 0.04]` gave 0.64) and would have moved
a client's threshold decisions; a reader of the NVIDIA forum thread caught
it. For `noul`, jevify keeps its own statistic (distance of P(true) from one
half, scaled to [0, 1]) under `x_jevify.confidence`, never on Jev's field.

Scorecards under `docs/evidence/` whose `run_at` precedes 2026-09-23 record
the entropy statistic in their wrong-case column; the reliability table bins
on the top probability and is unaffected. They are not rewritten, because a
number in a document comes from a run.

## Consequences

- Compatibility is tested, not asserted. A client written for Jev runs
  against jevify.
- The `x_jevify` namespace grows without ever touching Jev's fields.
- The explicit ingest introduces server-side state and its expiry; the
  stateless route remains the default and the ingest is opt-in.
- Jev's `confidence` is served with Jev's meaning and Jev's formula (D8), a
  statistic of the distribution, and the claims contract forbids describing
  it otherwise.

## Rejected alternatives

- **jevify's own API with a Jev compatibility shim.** Two schemas to keep in
  sync; rejected under the chokepoint principle and INV-8.
- **Extending Jev's fields in place.** Would break the SDK's parsing on unknown
  fields in some versions; the namespace is safer.
- **A stateful-only API.** Would break Jev clients; ingest is optional.
