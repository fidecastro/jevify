# jevify

Make a pretrained model you already have answer typed questions over a shared
state, in one pass, in the shape TypeSafe's Jev made familiar, and measure how
well it does. jevify is a harness: it owns no weights and trains nothing. You
bring the model, a recipe says how it is asked, and a scorecard says how it
did.

**Status:** the design is adopted and every planned slice has shipped. The
section "What works today" lists the commands, the model kinds and the
evidence behind each claim; "Not yet" lists what is unproven.

## Install

jevify is a Python package with a command-line tool. It needs Python 3.12 or
newer and five small dependencies (httpx, pydantic, PyYAML, starlette,
uvicorn); the model runs elsewhere, behind an OpenAI-compatible endpoint you
point it at. The optional `encoder` extra adds torch and transformers for the
two in-process kinds.

```sh
uv tool install jevify              # or: pipx install jevify
uv tool install 'jevify[encoder]'   # adds the in-process kinds
jevify --version
```

To work on jevify itself:

```sh
uv sync --group dev                 # add --extra encoder for the in-process kinds
uv run pytest                       # live and encoder tests skip unless you opt in
```

Ask your agent: "Install jevify as a tool, confirm `jevify --version`, and
tell me which commands exist in this checkout."

## What works today

Five commands, one recipe file per model deployment:

| Command | What it does |
|---|---|
| `jevify probe <recipe>` | Measures the backend (dialect, context, logprob rungs, token ids, cache granularity, fan-out, images) and writes the findings into the recipe. |
| `jevify ask <recipe> --state ... --choice/--score/--noul ...` | Warms one state and answers typed questions in Jev's response shape, with the readout details under `x_jevify`. |
| `jevify serve <recipe>` | Serves Jev's API for that recipe: `POST /v1/systemone`, `/v1/states`, `/v1/classify`, `GET /v1/models`, `/health`. The unmodified `typesafe-sdk` round trip is a CI job. |
| `jevify eval <recipe> <suite> [--controls]` | Runs a frozen, hashed suite and writes a scorecard: raw decisions to `runs/`, the summary to `docs/evidence/`. |
| `jevify calibrate <recipe> <raw>` | Fits one temperature per question type and option-count bucket on a raw run, reports held-out metrics, and refuses a table it cannot defend. |

Four model kinds plug into the same backend port (ADR-0002):

| Kind | Talks to | Proven live with |
|---|---|---|
| `endpoint` | Any OpenAI-compatible server; dialect extras for vLLM and llama.cpp | DeepSeek-V4-Flash on vLLM; Ternary Bonsai 2 27B and Qwen3-VL-Reranker-8B on llama.cpp; text and images on all three |
| `rerank` | A `/v1/rerank` route | Fakes only; the llama.cpp build at hand returned zero scores for the GGUF tried |
| `embedding` | A `/v1/embeddings` route, or a model in-process | Qwen3-Embedding-0.6B in-process; the route against fakes only |
| `encoder` | A sequence-classification NLI head in-process (`nli` layout) | facebook/bart-large-mnli in-process |

The same 29-case policy suite, one run each, five recipes across the four
kinds. Every row links
to the scorecard that records the command, the recipe hash, the suite hash
and the raw file's digest:

| Recipe | Kind | Accuracy | Median latency | Scorecard |
|---|---|---|---|---|
| `deepseek-v4-flash-vision-exp-keys.vllm` | endpoint | 28/29 | 428 ms | [summary](docs/evidence/deepseek-v4-flash-vision-exp-keys.vllm--policy-29.md) |
| `ternary-bonsai-2-27b.llamacpp` | endpoint | 29/29 | 99 ms | [summary](docs/evidence/ternary-bonsai-2-27b.llamacpp--policy-29.md) |
| `qwen3-vl-reranker-8b.llamacpp` | endpoint | 26/29 | 68 ms | [summary](docs/evidence/qwen3-vl-reranker-8b.llamacpp--policy-29.md) |
| `qwen3-embedding-0.6b.local` | embedding | 19/29 | 8 ms | [summary](docs/evidence/qwen3-embedding-0.6b.local--policy-29.md) |
| `bart-large-mnli.local` | encoder | 15/29 | 7 ms | [summary](docs/evidence/bart-large-mnli.local--policy-29.md) |

DeepSeek has scored 28/29 and 29/29 on separate runs; the case it misses
sits at the policy boundary and flips between runs. Twenty-nine cases rank
models; they do not measure calibration, which is why
`jevify calibrate` refused to write a table for any of them
([evidence](docs/evidence/deepseek-v4-flash-vision-exp-keys.vllm--calibration.md)).
Latencies are one machine's, one day's, and recorded as such.

## Not yet

Named here so that nobody has to discover it: the direct-call control (no
"faster than generation" claim until it exists); suites beyond `policy-29`
(instruction flips, multimodal); the rerank kind and the embedding route
against a live server; Laya's marker-slot layout for the encoder kind; the
grammar rung on the PrismML llama.cpp fork (accepted, but it does not
constrain the reported probabilities there); Windows. Choice menus on the endpoint kind are bounded by the
single-token identifier alphabet the probe verifies (26 on the DeepSeek
recipe); Jev's 255-option ceiling is not reachable there.

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

To operate jevify, point your agent at [`SKILL.md`](SKILL.md): it holds the
workflow (write a recipe, probe, ask, eval, serve), the anatomy of a recipe
per model kind, how to read `x_jevify`, and what each claim is allowed to
mean. Typical asks:

- "Write a recipe for the model behind `http://host:8000/v1`, probe it, and
  tell me which readout rung it landed on and why."
- "Run `policy-29` against this recipe with controls and read me the
  scorecard against the claims contract."
- "Serve this recipe and show me an unmodified typesafe-sdk call against it."

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
| [`SKILL.md`](SKILL.md) | How an agent operates jevify: recipes, probe, ask, eval, serve, and what the outputs mean. |
| [`docs/evidence/`](docs/evidence/) | Committed scorecard summaries, one per recipe and suite, each naming its run. |

## Licence

[MIT](LICENSE). Relicensed from GPL-3.0 on 2026-09-20 (ADR-0001 D3, amended); every line in the tree is the author's own or from a permissively licensed source credited in `THIRD_PARTY_LICENSES.md` when one enters.
