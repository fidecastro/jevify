# jevify

**Make any model you already have answer typed questions in one pass, the way
Jev does, and measure how well it does.**

[![CI](https://github.com/fidecastro/jevify/actions/workflows/ci.yml/badge.svg)](https://github.com/fidecastro/jevify/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/jevify.svg)](https://pypi.org/project/jevify/)
[![Licence: MIT](https://img.shields.io/badge/licence-MIT-blue.svg)](https://github.com/fidecastro/jevify/blob/main/LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)](https://www.python.org/)

You bring a model behind an OpenAI-compatible endpoint (vLLM, llama.cpp,
anything with logprobs) or a small model that runs in-process. jevify sends it
a **state** once, then asks **typed questions** about it: a choice among
options, a score on ordered levels, a yes/no with a probability. Each answer
is read straight off the model's next-token distribution, no generation, no
parsing, one request per question. The result comes back in the exact shape
of TypeSafe's Jev API, so the unmodified `typesafe-sdk` works against it.

jevify owns no weights and trains nothing. A **recipe** says how a model is
asked, a **probe** proves what the deployment can do, and a **scorecard**
says how it did, with the command, hashes and raw file that produced every
number.

```text
   state (text, images)          one warm request, cached prefix
        │
        ├── "Which team handles this?"  ──►  {billing: 0.94, support: 0.05, sales: 0.01}
        ├── "How urgent?"               ──►  {low: 0.05, medium: 0.10, high: 0.85}
        └── "Threatens to cancel."      ──►  p(true) = 0.95
                                              each ≈ 30–140 ms on a local GPU
```

## Quickstart

```sh
uv tool install jevify                # or: pipx install jevify
git clone https://github.com/fidecastro/jevify && cd jevify

# 1. point a recipe at your server (copy the nearest one under recipes/)
jevify probe recipes/gemma-4-e4b-it.llamacpp.yaml

# 2. ask
jevify ask recipes/gemma-4-e4b-it.llamacpp.yaml \
  --state "Customer: I was charged twice. Fix it today or I cancel." \
  --choice "dept:Which team should handle this?=billing,support,sales" \
  --score  "urgency:How urgent is this?=low,medium,high" \
  --noul   "cancel:The customer threatens to cancel." \
  [--image screenshot.png]

# 3. serve Jev's API and use the stock SDK
jevify serve recipes/gemma-4-e4b-it.llamacpp.yaml --port 8600
```

Every answer carries Jev's fields (`choice`, `confidence`, `probabilities`,
`score`, `legend`, `noul`) plus an `x_jevify` block naming the readout method,
the cached tokens, the latency and the recipe hash that produced it.

## How it works

1. **Recipe.** A hashed YAML file: the endpoint, the prompt template (chat
   messages or a raw prompt with the model's own control tokens), the answer
   token spellings, and the readout policy. The only way a model is ever
   asked.
2. **Probe.** `jevify probe` measures the deployment: dialect (vLLM,
   llama.cpp, generic), context, which logprob rungs it honours, whether
   thinking is really off, whether an image really adds tokens, how much of a
   warmed prefix a question reuses. The findings are written into the recipe.
3. **Warm, then ask.** The state goes once; each question is a short suffix
   that reuses the server's prefix cache. A 32k-token state costs one warm and
   then a few hundred milliseconds per question.
4. **Readout ladder.** Named-token logprobs, grammar with post-sampling
   probabilities, top-k, equal bias, and a floored top-k as the last resort.
   Auto walks down the proven rungs; the answer names the rung that read it
   and whether it was degraded.
5. **Evidence.** `jevify eval` runs a frozen suite with controls (shuffled
   states, permuted options) and writes a scorecard. `jevify calibrate` fits
   temperatures and refuses tables it cannot defend.

## Model kinds

| Kind | Talks to | Proven live with |
|---|---|---|
| `endpoint` | Any OpenAI-compatible server; dialect extras for vLLM and llama.cpp | DeepSeek-V4-Flash on vLLM; Ternary Bonsai 2 27B, Gemma 4 E4B and Qwen3-VL-Reranker 8B/2B on llama.cpp; text and images |
| `embedding` | A `/v1/embeddings` route, or a model in-process | Qwen3-Embedding-0.6B in-process |
| `encoder` | An NLI sequence-classification head in-process | facebook/bart-large-mnli in-process |
| `rerank` | A `/v1/rerank` route | Fakes only |

The in-process kinds need `uv tool install 'jevify[encoder]'`.

## Results

Four frozen suites, every recipe on each, one run each, latencies per
question on one machine on one day. Every row links to the scorecard that
records the command, the recipe hash, the suite hash and the raw file digest.

**policy-29**, the diagnostic suite carried over from the author's earlier experiments (choice only):

| Recipe | Kind | Accuracy | Median latency | Scorecard |
|---|---|---|---|---|
| `gemma-4-e4b-it.llamacpp` | endpoint | 29/29 | 28 ms | [summary](https://github.com/fidecastro/jevify/blob/main/docs/evidence/gemma-4-e4b-it.llamacpp--policy-29.md) |
| `ternary-bonsai-2-27b.llamacpp` | endpoint | 29/29 | 99 ms | [summary](https://github.com/fidecastro/jevify/blob/main/docs/evidence/ternary-bonsai-2-27b.llamacpp--policy-29.md) |
| `deepseek-v4-flash-vision-exp-keys.vllm` | endpoint | 28/29 | 428 ms | [summary](https://github.com/fidecastro/jevify/blob/main/docs/evidence/deepseek-v4-flash-vision-exp-keys.vllm--policy-29.md) |
| `qwen3-vl-reranker-8b.llamacpp` | endpoint | 26/29 | 68 ms | [summary](https://github.com/fidecastro/jevify/blob/main/docs/evidence/qwen3-vl-reranker-8b.llamacpp--policy-29.md) |
| `qwen3-vl-reranker-2b.llamacpp` | endpoint | 21/29 | 49 ms | [summary](https://github.com/fidecastro/jevify/blob/main/docs/evidence/qwen3-vl-reranker-2b.llamacpp--policy-29.md) |
| `qwen3-embedding-0.6b.local` | embedding | 19/29 | 8 ms | [summary](https://github.com/fidecastro/jevify/blob/main/docs/evidence/qwen3-embedding-0.6b.local--policy-29.md) |
| `bart-large-mnli.local` | encoder | 15/29 | 7 ms | [summary](https://github.com/fidecastro/jevify/blob/main/docs/evidence/bart-large-mnli.local--policy-29.md) |

**policy-hard-52**, written for jevify before any model saw it: instruction
flips, negations, two-threshold policies, distractors, and long transcripts
with the decisive line buried (`tools/author_policy_hard.py`):

| Recipe | Accuracy | Flips | Thresholds | Long states | Scorecard |
|---|---|---|---|---|---|
| `deepseek-v4-flash-vision-exp-keys.vllm` | 47/52 | 6/9 | 16/18 | 12/12 | [summary](https://github.com/fidecastro/jevify/blob/main/docs/evidence/deepseek-v4-flash-vision-exp-keys.vllm--policy-hard-52.md) |
| `ternary-bonsai-2-27b.llamacpp` | 46/52 | 6/9 | 15/18 | 12/12 | [summary](https://github.com/fidecastro/jevify/blob/main/docs/evidence/ternary-bonsai-2-27b.llamacpp--policy-hard-52.md) |
| `gemma-4-e4b-it.llamacpp` | 45/52 | 5/9 | 16/18 | 12/12 | [summary](https://github.com/fidecastro/jevify/blob/main/docs/evidence/gemma-4-e4b-it.llamacpp--policy-hard-52.md) |
| `qwen3-vl-reranker-2b.llamacpp` | 34/52 | 3/9 | 9/18 | 12/12 | [summary](https://github.com/fidecastro/jevify/blob/main/docs/evidence/qwen3-vl-reranker-2b.llamacpp--policy-hard-52.md) |
| `bart-large-mnli.local` | 28/52 | 3/9 | 8/18 | 12/12 | [summary](https://github.com/fidecastro/jevify/blob/main/docs/evidence/bart-large-mnli.local--policy-hard-52.md) |
| `qwen3-embedding-0.6b.local` | 24/52 | 3/9 | 8/18 | 6/12 | [summary](https://github.com/fidecastro/jevify/blob/main/docs/evidence/qwen3-embedding-0.6b.local--policy-hard-52.md) |

Every generative model misses the same inverted instructions ("the action the
customer asked NOT to take"), a weakness the author's earlier experiments found across model
generations. Negations in the state and long transcripts are solved.

**doom-frames-41**, 164 questions over 41 ViZDoom screenshots at 640×480,
every label from the engine's labels buffer or a fixed expert rule
(`tools/record_doom_suite.py`; frames regenerate byte-identically and their
hashes are pinned in the manifest):

| Recipe | Enemy visible | Enemy count | Enemy side | Expert button | Median latency | Scorecard |
|---|---|---|---|---|---|---|
| `deepseek-v4-flash-vision-exp-keys.vllm` | 0.88 | 0.83 | 0.56 | 0.24 | 672 ms | [summary](https://github.com/fidecastro/jevify/blob/main/docs/evidence/deepseek-v4-flash-vision-exp-keys.vllm--doom-frames-41.md) |
| `ternary-bonsai-2-27b.llamacpp` | 0.93 | 0.85 | 0.37 | 0.24 | 97 ms | [summary](https://github.com/fidecastro/jevify/blob/main/docs/evidence/ternary-bonsai-2-27b.llamacpp--doom-frames-41.md) |
| `gemma-4-e4b-it.llamacpp` | 0.78 | 0.49 | 0.41 | 0.20 | 36 ms | [summary](https://github.com/fidecastro/jevify/blob/main/docs/evidence/gemma-4-e4b-it.llamacpp--doom-frames-41.md) |
| `qwen3-vl-reranker-2b.llamacpp` | 0.10 | 0.56 | 0.41 | 0.10 | 64 ms | [summary](https://github.com/fidecastro/jevify/blob/main/docs/evidence/qwen3-vl-reranker-2b.llamacpp--doom-frames-41.md) |

Presence and count are perception; the expert button asks the model to agree
with a rule it is not told, so that column is a floor for a scripted policy,
not a skill score. Enemy side is where the vision models diverge most.

**long-state-24**, one decisive customer line at the start, middle or end of
a filler transcript of about 2k, 8k, 16k or 32k tokens
(`tools/author_long_state.py`). This is the suite the warm step exists for:
the state is sent once, and each question then costs only its own tokens.

| Recipe | 2k | 8k | 16k | 32k | Warm at 32k | Per question after warm | Scorecard |
|---|---|---|---|---|---|---|---|
| `deepseek-v4-flash-vision-exp-keys.vllm` | 6/6 | 6/6 | 6/6 | 6/6 | 20 s | 359 ms, 31,232 tokens cached | [summary](https://github.com/fidecastro/jevify/blob/main/docs/evidence/deepseek-v4-flash-vision-exp-keys.vllm--long-state-24.md) |
| `ternary-bonsai-2-27b.llamacpp` | 6/6 | 6/6 | 6/6 | 6/6 | 13 s | 136 ms, 33,946 tokens cached | [summary](https://github.com/fidecastro/jevify/blob/main/docs/evidence/ternary-bonsai-2-27b.llamacpp--long-state-24.md) |
| `gemma-4-e4b-it.llamacpp` | 6/6 | context | context | context | | 37 ms at 2k | [summary](https://github.com/fidecastro/jevify/blob/main/docs/evidence/gemma-4-e4b-it.llamacpp--long-state-24.md) |

"context" means the launch's slot context was shorter than the state and the
case recorded a context error, which is the finding for that launch (Gemma
was launched with two 8k slots; Bonsai first ran with two 16k slots and
failed 16k and 32k the same way, then with one 64k slot for the row above). No model that could read the state missed
the decisive line at any position.

These suites rank models; they do not measure calibration, which is why
`jevify calibrate` refused to write a table for any of them.

## Play Doom

`tools/play_doom.py` lets a recipe play ViZDoom from screenshots: one frame
per decision, one question per frame, the chosen button pressed, a video with
the decision and its latency burned in. `tools/record_doom_suite.py` records
the frozen perception suite the same way. Both need `uv sync --extra games`
and ffmpeg.

## Work with your agent

jevify is written to be operated through a coding agent. Point yours at
[`SKILL.md`](https://github.com/fidecastro/jevify/blob/main/SKILL.md): the workflow, recipe anatomy per kind, how to read
`x_jevify`, and what a result may claim. Typical asks:

- "Write a recipe for the model behind `http://host:8000/v1`, probe it, and
  tell me which readout rung it landed on and why."
- "Run `policy-hard-52` against this recipe with controls and read me the
  scorecard against the claims contract."
- "Serve this recipe and show me an unmodified typesafe-sdk call against it."

## Repository map

| Path | What it holds |
|---|---|
| [`jevify/`](https://github.com/fidecastro/jevify/blob/main/jevify) | The package: domain, ports, adapters (endpoint, embedding, encoder, rerank), recipes, evaluation, Jev API, CLI |
| [`recipes/`](https://github.com/fidecastro/jevify/blob/main/recipes) | One recipe per model deployment, with its probe results and launch line |
| [`suites/`](https://github.com/fidecastro/jevify/blob/main/suites) | Frozen, hashed suites and their manifests |
| [`docs/evidence/`](https://github.com/fidecastro/jevify/blob/main/docs/evidence) | Committed scorecards, one per recipe and suite |
| [`docs/00-invariants.md`](https://github.com/fidecastro/jevify/blob/main/docs/00-invariants.md) | What jevify is and refuses to be; the claims contract. Ranks above everything else |
| [`docs/adr/`](https://github.com/fidecastro/jevify/blob/main/docs/adr/README.md) | The decision ledger |
| [`tools/`](https://github.com/fidecastro/jevify/blob/main/tools) | Suite authoring, the Doom recorder and player |
| [`AGENTS.md`](https://github.com/fidecastro/jevify/blob/main/AGENTS.md) | The engineering principles that bind every change |
| [`SKILL.md`](https://github.com/fidecastro/jevify/blob/main/SKILL.md) | How an agent operates jevify |

To work on jevify: `uv sync --group dev` (add `--extra encoder --extra
games` as needed) and `uv run pytest`. Live and in-process tests skip unless
you opt in.

## Not yet

Named so that nobody has to discover it: a direct-call control, so no
"faster than generation" claim exists yet; a Doom player that survives (both
models die within seconds of `deadly_corridor`); the rerank kind and the
embedding route against a live server; a marker-slot encoder layout; the
grammar rung on the PrismML llama.cpp fork, which accepts a grammar without
constraining the reported probabilities; Windows. Choice menus on the
endpoint kind are bounded by the single-token identifier alphabet the probe
verifies (26 on the DeepSeek recipe); Jev's 255-option ceiling is not
reachable there.

## Licence

[MIT](https://github.com/fidecastro/jevify/blob/main/LICENSE).
