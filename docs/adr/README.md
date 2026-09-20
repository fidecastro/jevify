# ADR Index

> The decision ledger. One row per ADR; the **Status** column is authoritative.
> A new ADR adds a row. Superseding or amending an ADR updates **both** its row
> here and the header of the ADR itself, in the same commit. An ADR never
> contradicts `docs/00-invariants.md`; one that would is written as an
> amendment to that document instead (its §7).

Format of every ADR: title, status line with date, context, decision,
consequences, rejected alternatives. Decisions are numbered inside the ADR so
later documents can cite "ADR-0002 D3".

| # | Title | Status |
|---|---|---|
| [0001](0001-founding-choices.md) | Founding choices: name, scope, licence, toolchain, conventions | Accepted (2026-09-19) |
| [0002](0002-backend-port-and-adapter-kinds.md) | One backend port, four adapter kinds, a capability probe, a readout ladder | Accepted (2026-09-19) |
| [0003](0003-jev-compatible-api.md) | Jev's evaluate call is the API; extensions are namespaced; the SDK is the acceptance test | Accepted (2026-09-19) |
| [0004](0004-recipes.md) | Recipes: the hashed file that is the only way a model is asked | Accepted (2026-09-19) |
| [0005](0005-evaluation-evidence-and-calibration.md) | Suites, scorecards, committed evidence, post-hoc calibration | Accepted (2026-09-19) |
| [0006](0006-toolchain-dependency-budget-and-portability.md) | Toolchain, dependency budget and portability: five light core packages, extras, argparse, uv/pipx, Python 3.12–3.14, no lock-in | Accepted (2026-09-19); makes ADR-0001 D4 precise |
