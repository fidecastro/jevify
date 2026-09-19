# jevify

Make a pretrained model you already have answer typed questions over a shared
state, in one pass, in the shape TypeSafe's Jev made familiar, and measure how
well it does. jevify is a harness: it owns no weights and trains nothing. You
bring the model, a recipe says how it is asked, and a scorecard says how it
did.

**Status:** the design is adopted and the implementation has not started.
What exists today is the governing documentation below.

## Start with your agent

jevify assumes you work through a coding agent (Claude Code, Codex, Grok
Build, or another). The documents are written to be read by it, and this
README tells you what to ask. Clone the repository, open it in your agent,
and start with:

> Read `docs/00-invariants.md` and `docs/adr/README.md`. Explain what jevify
> is, what it refuses to be, and what has been decided so far. Then read
> `AGENTS.md` and tell me which rules you will follow when you change
> anything here.

From there, the things you will want to ask:

- **To understand a decision:** "Explain ADR-0002 and how a new model kind
  would plug into the backend port."
- **To propose a change:** "Draft an ADR for X. Check it against every
  invariant first; if it needs one changed, write it as an amendment under
  `docs/00-invariants.md` §7 instead."
- **To review a contribution:** "Check this pull request against the claims
  contract in `docs/00-invariants.md` §5 and the four principles in
  `AGENTS.md`. Name anything unproven."
- **To find your way:** "Show me the document map at the end of
  `docs/00-invariants.md` and tell me where recipes, suites and evidence will
  live."

Once the code exists, the same pattern continues: you will ask your agent to
probe an endpoint, write a recipe for a model, run a suite and read the
scorecard, or start the server. A `SKILL.md` will make that operating
knowledge explicit for the agent. Until it exists, `AGENTS.md` and the
invariants are what the agent reads.

## Without an agent

Everything is plain Markdown and reads in order: `docs/00-invariants.md`,
then the ADR index at `docs/adr/README.md`, then `AGENTS.md`. Changes follow
the same rules either way: a decision is an ADR, an invariant changes only by
amendment, and no number appears without the run that produced it.

## Documents

| Document | What it holds |
|---|---|
| [`docs/00-invariants.md`](docs/00-invariants.md) | Purpose, vocabulary, the ten invariants, the design criteria and the score-semantics contract. Ranks above everything else. |
| [`docs/adr/README.md`](docs/adr/README.md) | The decision ledger, one row per ADR. |
| [`AGENTS.md`](AGENTS.md) | The four engineering principles and house rules that bind every change. |

## Licence

[GPL-3.0](LICENSE).
