# jevify

Make a pretrained model you already have answer typed questions over a shared
state, in one pass, in the shape TypeSafe's Jev made familiar, and measure how
well it does. jevify is a harness: it owns no weights and trains nothing. You
bring the model; a recipe says how it is asked; a scorecard says how it did.

Status: design adopted, implementation not started.

Governing documents:

- [`docs/00-invariants.md`](docs/00-invariants.md) — purpose, vocabulary, invariants, design criteria and the score-semantics contract. Ranks above everything else.
- [`docs/adr/README.md`](docs/adr/README.md) — the decision ledger.
- [`AGENTS.md`](AGENTS.md) — engineering principles that bind every change.

Licensed under [GPL-3.0](LICENSE).
