---
name: jevify
description: Operate jevify, the bring-your-own-model harness that makes a pretrained model answer Jev-shaped typed questions (choice, score, noul) over a shared state and measures how well it does. Use when asked to connect a model behind an OpenAI-compatible endpoint (vLLM, llama.cpp, others) or an in-process encoder or embedding model, write or probe a recipe, ask questions, run a suite, read a scorecard, calibrate, or serve Jev's API.
---

# Operating jevify

jevify owns no weights and trains nothing. A **recipe** (YAML, hashed) is the
only way a model is asked; the **probe** measures what a deployment can do and
writes it into the recipe; **ask** answers questions; **eval** scores a frozen
suite; **serve** exposes Jev's API. Read `docs/00-invariants.md` once: its
vocabulary and its claims contract (§5) govern everything you say about a
result.

## The workflow

1. **Write a recipe** under `recipes/` (copy the nearest one; anatomy below).
2. **Probe it**: `jevify probe recipes/<name>.yaml`. Read the printed
   capabilities; the recipe now carries a `probe:` section and its status
   moves from `draft` to `probed`. `--dry-run` prints without writing.
3. **Ask one question** to see the shape of an answer:
   ```sh
   jevify ask recipes/<name>.yaml --state "..." \
     --choice "id:Which team?=billing,support,sales" \
     --noul "id2:The customer threatens to cancel." \
     --score "id3:How urgent?=low,medium,high" [--image photo.png] [--rung top_k]
   ```
4. **Score a suite**: `jevify eval recipes/<name>.yaml suites/policy-29.jsonl --controls`.
   Suites: `policy-29` (diagnostic), `policy-hard-52` (flips, negations,
   thresholds, distractors, long states), `doom-frames-41/` (screenshots;
   rows carry `images`, regenerate the frames with
   `uv run python tools/record_doom_suite.py` after `uv sync --extra games`),
   `long-state-24` (2k to 32k tokens). `tools/play_doom.py <recipe> --scenario
   defend_the_center --policy perception` records a video of a recipe playing.
   Raw decisions land in `runs/` (git-ignored); the summary in `docs/evidence/`
   (committed). Read the Markdown summary, then the `wrong` list in the JSON.
5. **Serve**: `jevify serve recipes/<name>.yaml --port 8600`. Jev's SDK works
   unmodified against it (`typesafe-sdk`, any key unless `--api-key-env` is set).
6. **Calibrate** only with a few hundred labeled decisions:
   `jevify calibrate recipes/<name>.yaml runs/<raw>.jsonl`. It refuses when a
   bucket is too small or error-free; `--force` writes anyway and says so.

Live and in-process tests skip by default. `JEVIFY_LIVE_RECIPES=a.yaml,b.yaml
uv run pytest -m live` runs the live checks; `uv run pytest -m encoder` runs
the in-process ones (needs the `encoder` extra and a model download).

## Recipe anatomy by kind

Every recipe has `schema_version: 1`, `model: {name, kind, revision}` and
`provenance: {author, created, status, notes, scorecards}`. The kind decides
the rest.

**`endpoint`** (any OpenAI-compatible server): `endpoint: {base_url,
api_key_env, dialect: auto|vllm|llamacpp|generic, timeout_s}`, `template`
and `answers`.
- `template.mode: messages` sends chat messages; put thinking off in
  `template.kwargs` (`{thinking: false}` for DeepSeek, `{enable_thinking:
  false}` for Qwen3-class models; names are model-specific, the probe checks
  that no `<think>` token opens the answer).
- `template.mode: raw` sends a completion prompt you write with the model's
  own control tokens; use it when the chat template would not put the model
  in its trained regime (the Qwen3-VL reranker recipe is the example).
- `template.questions.choice_strategy`: `identifiers` (one request, options
  labelled A, B, C) or `per_option` (one yes/no request per option; needed
  for a model that never emits letters, such as a reranker).
- `answers.noul.true/false` and `answers.identifiers` list token spellings;
  the probe fills their ids.
- `readout.rung: auto` walks the rungs the probe proved, strongest first:
  `named_token_logprobs` (vLLM with `logprob_token_ids`), `grammar`
  (llama.cpp), `top_k`, `equal_bias`, `top_k_floor` (degraded, missing
  labels floored and listed). A rung that cannot see every label hands the
  question to the next; the answer names the rung that read it. A pinned rung
  never walks: it fails loudly instead.

**`rerank`**: `endpoint` plus `rerank: {path, query, document, instruction,
score_field}`. Choice and noul only; semantics `relevance`.

**`embedding`**: `embedding: {query, option, scale}` plus either `endpoint`
(a `/v1/embeddings` route) or `local: {path, device, max_tokens, pooling}`.
Choice only; semantics `similarity`. The query template carries the
instruction for instruction-tuned models (`Instruct: {instructions}\nQuery:
{state}` for Qwen3-Embedding).

**`encoder`**: `local` plus `encoder: {layout: nli, premise, hypothesis,
noul_hypothesis, entailment_label, contradiction_label}`. An NLI
sequence-classification head; all three question types; semantics `readout`.

## Reading an answer

Jev's fields come first (`choice`, `confidence`, `probabilities`; `score`,
`legend`; `noul`). Everything jevify adds is under `x_jevify`:

| Field | Meaning |
|---|---|
| `semantics` | `readout`: the model's own distribution at the answer slot. `relevance`: independent scores, ranked. `similarity`: cosines, ranked. `calibrated`: a readout rescaled by a fitted temperature. Only `readout` and `calibrated` may be read as probabilities. |
| `rung` | Which readout method produced the distribution. |
| `degraded`, `missing` | True when a label was floored because the server did not return it; the labels are listed. |
| `off_menu_mass` | Probability the model put outside the answer set, when the rung can see it. |
| `cached_tokens`, `prompt_tokens`, `latency_ms`, `calls` | Evidence, per question. |
| `recipe_hash` | The recipe that asked. Quote it with any number you report. |

## What you may claim

- A number comes from a scorecard that names its command, recipe hash, suite
  hash and raw digest. Quote those with it. Twenty-nine cases rank models;
  they do not measure calibration.
- "Faster than generation" is not claimable until the direct-call control
  exists. Latencies are one machine's, one day's.
- A `similarity` or `relevance` answer is a ranking, not a probability; say so.
- A recipe whose status is `draft` has not been probed; do not report results
  from it as the model's.
- Name what was not exercised (a kind against fakes only, an unprobed recipe,
  a rung the probe could not prove).

## When something fails

- **Probe says thinking is on**: set the model's template kwarg, or switch to
  `raw` mode with the control tokens spelled out.
- **`prefill_honored: false`**: the template appends EOS after the prefill;
  do not rely on `template.prefill` for that deployment.
- **Context limit error**: the state is longer than the server's context; the
  message names the limit. Shorten the state or raise the server's context.
- **`logit_bias` rejected**: vLLM under speculative decoding; the ladder
  falls back to `top_k`.
- **llama.cpp refuses a GGUF type**: the build lacks that quantization. Ternary
  Bonsai's `PQ2_0` needs the PrismML fork (branch `prism`), launched with
  `--jinja` so template kwargs reach the template; the recipe records the
  launch line.
- **Multi-token identifiers**: the probe reports `multi_token_answers`; pick
  spellings that are single tokens for that tokenizer (the recipe lists them).
- **Images accepted but ignored on llama.cpp's native route**: the server draws
  a random media marker per process and exposes it in `/props`; the probe
  reads it, proves the image adds prompt tokens, and writes the marker into
  `template.image_marker`. Launch with `LLAMA_MEDIA_MARKER='<__media__>'` so
  the recipe survives restarts.
- **Warm buys nothing** (`cache.warm_reuse_tokens` near zero in the probe): a
  hybrid model with recurrent layers on llama.cpp resumes only from a
  checkpoint at the end of an earlier prompt, and a chat template closes the
  user turn right after the state. Use `template.mode: raw` so the warm prompt
  ends at the state boundary; the Bonsai recipe is the example.
- **Missing encoder extra**: the error names the install command.

## House rules that bind you here

Test first at a public seam; prove with a run you observed; one authoritative
path per process (`compose.py` builds backends, `render.py` renders prompts,
`distribution.py` computes confidence); secrets never enter the tree; `runs/`
and weights stay out of git. `AGENTS.md` is the full statement.
