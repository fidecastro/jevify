# ADR-0005 — Suites, scorecards, committed evidence, post-hoc calibration

- **Status:** accepted (2026-09-19).
- **Extends:** ADR-0004 (a scorecard names a recipe hash). Implements INV-7
  and design criteria 8 and 9; enforces the claims contract in
  `docs/00-invariants.md` §5.

## Context

The week Jev launched, the public numbers were mostly unverifiable. Laya's
benchmark document cites result files absent from its repository. Jev's own
evaluations grade against the average of two frontier models. Independent
clones compare a fine-tuned model with a zero-shot one in the same row. The
local precursor `decision-poc` did this right: protocols hashed before the
run, seeds and revisions recorded, controls for shuffled context and option
permutation, and a semantics label on every score. It also got one thing
wrong: raw outputs were git-ignored, so every link from its reports to
evidence is broken on GitHub.

The precursor's findings also fix what the first suites must contain: the
inverted instruction that every model failed, negation combined with a
reversal, a numeric policy boundary, and an insufficient-information option.

## Decision

### D1 — Suites are frozen and hashed

A suite is a JSONL file under `suites/` in jevlike's one-row-per-case format
extended with an instruction and a question type, plus a manifest that
records its source, its construction, how expected answers were produced
(human, program, teacher model, teacher ensemble), and its SHA-256. A suite
never changes; a corrected suite is a new file with a new hash.

The initial suites:

- `policy-29`: the precursor's 29-case policy and negation suite, carried
  over verbatim with its hash.
- `flips`: to be authored; instruction inversions, negation plus reversal,
  missing correct option, insufficient evidence, evidence deep in a long
  state. Expected answers by program or human, never by teacher.
- `typed-decisions`: the public benchmark on Hugging Face, four workflows,
  1,200 training and 400 test cases, teacher-ensemble soft labels,
  Apache-2.0. Used zero-shot on its test split only. Its manifest says the
  expected answers are teacher agreement, not truth.
- `multimodal`: to be authored; image-dependent cases and cases where image
  and text conflict.

### D2 — A scorecard is the unit of evidence

`jevify eval <recipe> <suite>` writes one scorecard: the recipe hash, the
suite hash, the backend kind, dialect, version and flags, the model
identifier and revision, jevify's version and the versions of its
dependencies, the device, the date, every per-decision raw answer with its
distribution, semantics label, readout rung, permutation record and latency,
and the summary computed from those raw answers by committed code. Raw
answers go to `runs/`, git-ignored. The summary and a digest of the raw file
go to `docs/evidence/`, committed. A summary whose raw digest cannot be
verified is invalid.

### D3 — Every scorecard carries controls

- Shuffled-context control: each case answered against a different case's
  state, to show the answers depend on the state.
- Option-permutation control: agreement of the choice under a permuted
  option order.
- The direct call: the same backend and model asked to write its answers,
  with and without its reasoning mode, timed and scored on the same suite.
  This is the only basis on which "faster" may be claimed, and it shows what
  the readout gives up in accuracy.
- Where available, the embedding and reranker kinds on the same suite, as the
  relevance and similarity baselines.

### D4 — Metrics

Per question type: accuracy of the selected value; log loss and Brier score of
the distribution against the expected answer; expected calibration error and
a reliability table over ten bins; for `score`, mean absolute error of the
expectation and the ranked probability score. Latency: warm and evaluate
separately, median and 95th percentile, per question and per batch, with the
question count. Abstention: the fraction of decisions whose confidence falls
below a stated threshold, reported next to the error rate among the rest.

Calibration metrics on fewer than a few hundred independent decisions are
reported with the count in the same cell and no conclusion drawn from them.

### D5 — Calibration is post-hoc and per recipe

`jevify calibrate <recipe> <labels>` fits one temperature per question type
and option-count bucket, Laya's bucketing, on the user's labeled decisions,
using a held-out split the command creates and records. It writes the table
into the recipe's `calibration` section and the fit's evidence, log loss and
Brier before and after, ECE and the reliability table, to `docs/evidence/`.
Only then may an answer carry the `calibrated` label, and only for questions
in a bucket the table covers. A temperature never changes a ranking; the
command says so in its output.

### D6 — Reproduction is a command

Every evidence summary names the command line that regenerates it from the
recipe, the suite and a reachable backend. CI runs at least one such
regeneration against a fixture backend on every change to the evaluator.

## Consequences

- Publishing a number costs a run and a commit. That is the intent.
- The teacher-graded suite can show a recipe agreeing with a teacher more or
  less than Jev does; it cannot show either being right. Its rows say so.
- Direct-call controls are slow on large models. They are required for a
  scorecard that claims speed and optional otherwise, and the scorecard says
  which.
- `docs/evidence/` grows with every scorecard. Summaries are small; raw
  outputs never enter git.

## Rejected alternatives

- **Git-ignoring evidence and linking to it**, as the precursor did. Broken
  links are unreproducible numbers.
- **Teacher-graded suites as the primary measure**, as TypeSafe and the
  benchmark do. Kept as one suite, labeled as agreement.
- **Calibrating by default with a shipped table.** A table fitted on one
  domain is wrong on another; the English Laya checkpoint's 95% confidence at
  zero accuracy on Khmer is the cautionary case.
- **Conformal prediction sets as the abstention mechanism.** Deferred; the
  marginal coverage guarantee does not apply per decision, and a threshold on
  a measured error curve is simpler to explain.
