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

Three frozen suites, every recipe on each, one run each. Every row links to
the scorecard that records the command, the recipe hash, the suite hash and
the raw file's digest. Latencies are one machine's, one day's, per question.

**policy-29**, the diagnostic suite carried from decision-poc (choice only):

| Recipe | Kind | Accuracy | Median latency | Scorecard |
|---|---|---|---|---|
| `gemma-4-e4b-it.llamacpp` | endpoint | 29/29 | 28 ms | [summary](docs/evidence/gemma-4-e4b-it.llamacpp--policy-29.md) |
| `ternary-bonsai-2-27b.llamacpp` | endpoint | 29/29 | 99 ms | [summary](docs/evidence/ternary-bonsai-2-27b.llamacpp--policy-29.md) |
| `deepseek-v4-flash-vision-exp-keys.vllm` | endpoint | 28/29 | 428 ms | [summary](docs/evidence/deepseek-v4-flash-vision-exp-keys.vllm--policy-29.md) |
| `qwen3-vl-reranker-8b.llamacpp` | endpoint | 26/29 | 68 ms | [summary](docs/evidence/qwen3-vl-reranker-8b.llamacpp--policy-29.md) |
| `qwen3-vl-reranker-2b.llamacpp` | endpoint | 21/29 | 49 ms | [summary](docs/evidence/qwen3-vl-reranker-2b.llamacpp--policy-29.md) |
| `qwen3-embedding-0.6b.local` | embedding | 19/29 | 8 ms | [summary](docs/evidence/qwen3-embedding-0.6b.local--policy-29.md) |
| `bart-large-mnli.local` | encoder | 15/29 | 7 ms | [summary](docs/evidence/bart-large-mnli.local--policy-29.md) |

**policy-hard-52**, written for jevify before any model saw it: instruction
flips, negations, two-threshold policies, distractors, and long transcripts
with the decisive line buried (`tools/author_policy_hard.py`):

| Recipe | Accuracy | Flips | Thresholds | Long states | Scorecard |
|---|---|---|---|---|---|
| `deepseek-v4-flash-vision-exp-keys.vllm` | 47/52 | 6/9 | 16/18 | 12/12 | [summary](docs/evidence/deepseek-v4-flash-vision-exp-keys.vllm--policy-hard-52.md) |
| `ternary-bonsai-2-27b.llamacpp` | 46/52 | 6/9 | 15/18 | 12/12 | [summary](docs/evidence/ternary-bonsai-2-27b.llamacpp--policy-hard-52.md) |
| `gemma-4-e4b-it.llamacpp` | 45/52 | 5/9 | 16/18 | 12/12 | [summary](docs/evidence/gemma-4-e4b-it.llamacpp--policy-hard-52.md) |
| `qwen3-vl-reranker-2b.llamacpp` | 34/52 | 3/9 | 9/18 | 12/12 | [summary](docs/evidence/qwen3-vl-reranker-2b.llamacpp--policy-hard-52.md) |
| `bart-large-mnli.local` | 28/52 | 3/9 | 8/18 | 12/12 | [summary](docs/evidence/bart-large-mnli.local--policy-hard-52.md) |
| `qwen3-embedding-0.6b.local` | 24/52 | 3/9 | 8/18 | 6/12 | [summary](docs/evidence/qwen3-embedding-0.6b.local--policy-hard-52.md) |

Every generative model misses the same inverted instructions ("the action the
customer asked NOT to take"), the weakness decision-poc found a year of
models ago. Negations in the state and long transcripts are solved.

**doom-frames-41**, 164 questions over 41 ViZDoom screenshots at 640×480,
every label from the engine's labels buffer or a fixed expert rule
(`tools/record_doom_suite.py`; frames regenerate byte-identically and their
hashes are pinned in the manifest):

| Recipe | Enemy visible | Enemy count | Enemy side | Expert button | Median latency | Scorecard |
|---|---|---|---|---|---|---|
| `deepseek-v4-flash-vision-exp-keys.vllm` | 0.88 | 0.83 | 0.56 | 0.24 | 672 ms | [summary](docs/evidence/deepseek-v4-flash-vision-exp-keys.vllm--doom-frames-41.md) |
| `ternary-bonsai-2-27b.llamacpp` | 0.93 | 0.85 | 0.37 | 0.24 | 97 ms | [summary](docs/evidence/ternary-bonsai-2-27b.llamacpp--doom-frames-41.md) |
| `gemma-4-e4b-it.llamacpp` | 0.78 | 0.49 | 0.41 | 0.20 | 36 ms | [summary](docs/evidence/gemma-4-e4b-it.llamacpp--doom-frames-41.md) |
| `qwen3-vl-reranker-2b.llamacpp` | 0.10 | 0.56 | 0.41 | 0.10 | 64 ms | [summary](docs/evidence/qwen3-vl-reranker-2b.llamacpp--doom-frames-41.md) |

Presence and count are perception; the expert button asks the model to agree
with a rule it is not told, so that column is a floor for a scripted policy,
not a skill score. Enemy side is where the vision models diverge most.

**long-state-24**, one decisive customer line at the start, middle or end of
a filler transcript of about 2k, 8k, 16k or 32k tokens
(`tools/author_long_state.py`). This is the suite the warm step exists for:
the state is sent once, and each question then costs only its own tokens.

| Recipe | 2k | 8k | 16k | 32k | Warm at 32k | Per question after warm | Scorecard |
|---|---|---|---|---|---|---|---|
| `deepseek-v4-flash-vision-exp-keys.vllm` | 6/6 | 6/6 | 6/6 | 6/6 | 20 s | 359 ms, 31,232 tokens cached | [summary](docs/evidence/deepseek-v4-flash-vision-exp-keys.vllm--long-state-24.md) |
| `ternary-bonsai-2-27b.llamacpp` | 6/6 | 6/6 | 6/6 | 6/6 | 13 s | 136 ms, 33,946 tokens cached | [summary](docs/evidence/ternary-bonsai-2-27b.llamacpp--long-state-24.md) |
| `gemma-4-e4b-it.llamacpp` | 6/6 | context | context | context | | 37 ms at 2k | [summary](docs/evidence/gemma-4-e4b-it.llamacpp--long-state-24.md) |

"context" means the launch's slot context was shorter than the state and the
case recorded a context error, which is the finding for that launch (Gemma
was launched with two 8k slots; Bonsai first ran with two 16k slots and
failed 16k and 32k the same way, then with one 64k slot for the row above). No model that could read the state missed
the decisive line at any position.

## Not yet

Named here so that nobody has to discover it: the direct-call control (no
"faster than generation" claim until it exists); a closed-loop Doom player; the rerank kind and
the embedding route against a live server; Laya's marker-slot layout for the encoder kind; the
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
