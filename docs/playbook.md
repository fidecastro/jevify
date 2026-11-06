# jevify playbook

This guide shows how to use jevify day to day: ask questions about a state,
read the answer, serve the API, add a model and fix common problems. Each
step is one command. Terms follow [`00-invariants.md`](00-invariants.md) §2.

## 1. The words you need

| Word | Meaning |
|---|---|
| **State** | The input you ask about: a message, a document, a transcript, an image. |
| **Question** | What you ask about the state. Three types: **choice**, **score**, **noul**. |
| **Answer** | The result of one question: the selected value, the full distribution and a confidence. |
| **Recipe** | A YAML file under `recipes/` that says how one model is asked. Every command takes one. |
| **Probe** | A check that measures what the server and model can do, and writes the result into the recipe. |
| **Readout** | How jevify gets the answer: it reads the model's next-token probabilities. It never parses generated text. |

The three question types:

| Type | You give | You get |
|---|---|---|
| **choice** | A menu of options | The selected option and a probability for each option |
| **score** | Ordered levels, low to high | A number from 0 to the last level index, and a probability for each level |
| **noul** | A statement | The probability that the statement is true |

## 2. Set up once

1. Install [uv](https://docs.astral.sh/uv/).
2. In the repository root, install the dependencies:

   ```sh
   uv sync --group dev
   ```

3. Check the install:

   ```sh
   uv run jevify --version
   ```

`make` comes with the Xcode command-line tools on macOS and with most Linux
systems. `make help` lists every variable and every recipe.

## 3. Ask questions

Use `make ask`. Give a recipe, a state and at least one question:

```sh
make ask RECIPE=recipes/gemma-4-e4b-it.llamacpp.yaml \
  STATE="Customer: I was charged twice. Fix it today or I cancel." \
  CHOICE="dept:Which team should handle this?=billing,support,sales" \
  SCORE="urgency:How urgent is this?=low,medium,high" \
  NOUL="cancel:The customer threatens to cancel."
```

| Variable | Format | Example |
|---|---|---|
| `RECIPE` | Path to a recipe | `recipes/gemma-4-e4b-it.llamacpp.yaml` |
| `STATE` | The text, or `@path` to read it from a file | `STATE=@ticket.txt` |
| `CHOICE` | `id:question=option1,option2,...` | `dept:Which team?=billing,support` |
| `SCORE` | `id:question=level0,level1,...`, lowest level first | `urgency:How urgent?=low,medium,high` |
| `NOUL` | `id:statement` | `cancel:The customer threatens to cancel.` |
| `IMAGE` | A png, jpg or webp file, added after the text | `IMAGE=screenshot.png` |
| `RUNG` | Force one readout method (normally leave it out) | `RUNG=top_k` |

Rules:

- The `id` names the question in the answer. Use a short word.
- `make` reads `$` in a value. Write `$$49.99` for `$49.99`, or put the
  text in a file and use `STATE=@file`.
- To ask two questions of the same type, call jevify directly and repeat the
  flag:

  ```sh
  uv run jevify ask recipes/gemma-4-e4b-it.llamacpp.yaml --state @ticket.txt \
    --noul "cancel:The customer threatens to cancel." \
    --noul "refund:The customer asks for a refund."
  ```

## 4. Read the answer

`make ask` prints JSON. This is the shape of the answer to the example in
step 3, shortened; the numbers are illustrative, not a recorded run:

```json
{
  "model": "gemma-4-e4b-it",
  "answers": {
    "dept": {
      "type": "choice",
      "choice": "billing",
      "confidence": 0.9996,
      "probabilities": {"billing": 0.9997, "support": 0.0002, "sales": 0.0001},
      "x_jevify": {"semantics": "readout", "rung": "top_k", "degraded": false, "...": "..."}
    },
    "urgency": {
      "type": "score",
      "score": 1.9988,
      "confidence": 0.9988,
      "legend": {"0": "low", "1": "medium", "2": "high"},
      "probabilities": {"0": 0.0003, "1": 0.0005, "2": 0.9992}
    },
    "cancel": {"type": "noul", "noul": 0.9250}
  },
  "usage": {"input_tokens": 241, "output_tokens": 3},
  "x_jevify": {"state_id": "f832…", "warm_ms": 474.6, "elapsed_ms": 424.7, "recipe": "…", "recipe_hash": "ddfa…", "base_url": "…"}
}
```

### The answer fields (TypeSafe's Jev format)

| Field | Question type | Meaning |
|---|---|---|
| `type` | all | `choice`, `score` or `noul`. |
| `choice` | choice | The option with the highest probability. |
| `probabilities` | choice, score | One probability per option or level. They add up to 1. Score levels are keyed by index (`"0"`, `"1"`, ...). |
| `score` | score | The probability-weighted level index. 0 is the first level, 2 is the third. 1.9988 means "almost certainly level 2". |
| `legend` | score | Maps each level index to its text. |
| `noul` | noul | The probability that the statement is true. |
| `confidence` | choice, score | How peaked the distribution is: 0 when all options are equal, 1 when one option has all the probability. It is not the probability that the answer is correct. |

### The `x_jevify` fields in each answer

| Field | Meaning |
|---|---|
| `semantics` | What the numbers mean. `readout`: the model's own probabilities. `relevance` (reranker) and `similarity` (embedding): rankings, not probabilities. `calibrated`: a readout rescaled by a fitted table. |
| `rung` | The readout method that produced the answer, for example `top_k`. |
| `degraded` | `true` when the server did not return the probability of some option. jevify gave those options a floor value. Do not trust a degraded answer. |
| `missing` | Present when `degraded` is `true`: the options that got the floor value. |
| `off_menu_mass` | Probability the model put on tokens that are not an option. High values mean the model wanted to say something else, for example start reasoning. `null` when the readout method cannot see it. |
| `confidence` | On a noul answer only: how far `noul` is from 0.5, scaled to 0–1. |
| `calls` | Requests sent for this question. |
| `latency_ms` | Time for this question. |
| `prompt_tokens`, `cached_tokens` | Tokens sent, and tokens the server reused from its cache (`null` when the server does not say). |
| `permutation`, `temperature` | Option reordering policy, and the calibration temperature (`null` when the recipe has no calibration). |
| `recipe_hash` | The exact recipe that asked. Quote it with any number you report. |

### The top-level fields

| Field | Meaning |
|---|---|
| `model` | The model name from the recipe. |
| `usage.input_tokens` | Prompt tokens over all questions. |
| `usage.output_tokens` | One per question: jevify reads one token position. |
| `x_jevify.state_id` | A hash of the state as the recipe renders it. The same state with the same recipe gives the same id. |
| `x_jevify.warm_ms` | Time to send the state once (the warm). |
| `x_jevify.elapsed_ms` | Time to answer all questions after the warm. |
| `x_jevify.recipe`, `recipe_hash`, `base_url` | Which recipe and which server answered. |

## 5. Serve the API

Use `make serve`. It serves TypeSafe's Jev API for one recipe:

```sh
make serve RECIPE=recipes/gemma-4-e4b-it.llamacpp.yaml
```

It listens on `127.0.0.1:8600`. Use `PORT=8601` to pick another port.
Press Ctrl-C to stop it.

Check that it runs:

```sh
curl -s http://127.0.0.1:8600/health
```

Ask with `curl`. The body is TypeSafe's `system_one` request:

```sh
curl -s -X POST http://127.0.0.1:8600/v1/systemone \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-4-e4b-it",
    "state": "Customer: I was charged twice. Fix it today or I cancel.",
    "questions": {
      "dept":    {"type": "choice", "instructions": "Which team should handle this?",
                  "criteria": {"billing": "invoices and refunds", "support": null, "sales": null}},
      "urgency": {"type": "score", "instructions": "How urgent is this?",
                  "criteria": ["low", "medium", "high"]},
      "cancel":  {"type": "noul", "instructions": "The customer threatens to cancel."}
    }
  }'
```

Ask with the unmodified TypeSafe SDK (`typesafe-sdk`). Set `timeout`: the
SDK default is 10 seconds, and a busy server can take longer.

```python
import typesafe_sdk as typesafe

client = typesafe.TypeSafeClient(api_key="any", base_url="http://127.0.0.1:8600", timeout=60)
response = client.system_one(
    "Customer: I was charged twice. Fix it today or I cancel.",
    {
        "dept": typesafe.Choice(
            instructions="Which team should handle this?",
            criteria={"billing": "invoices and refunds", "support": None, "sales": None},
        ),
        "urgency": typesafe.Score(
            instructions="How urgent is this?", criteria=["low", "medium", "high"]
        ),
        "cancel": typesafe.Noul(instructions="The customer threatens to cancel."),
    },
)
print(response.answers["dept"].choice, response.answers["cancel"].noul)
```

The answers have the same fields as in step 4.

### Serve to other machines

By default only this machine can connect. To open the API to the network,
set a key first. Clients must then send it as `Authorization: Bearer <key>`.

```sh
export JEVIFY_API_KEY=<a long random value>
make serve RECIPE=recipes/gemma-4-e4b-it.llamacpp.yaml HOST=0.0.0.0
```

`make serve` refuses a `HOST` other than `127.0.0.1` when `JEVIFY_API_KEY`
is not set.

## 6. Add a new model

jevify needs a server that returns next-token log-probabilities, in one of
two ways:

- on the reply token (`logprobs` with `top_logprobs`), as vLLM and llama.cpp
  do; any recipe works;
- for the prompt (vLLM's `prompt_logprobs` on `/v1/completions`), as
  TensorFold's native server does for GLM-5.3-Flash, which refuses the
  first way; the probe proves the `prompt_logprobs` rung. A raw-mode recipe
  works on any such server. A chat-template recipe works where the server
  also takes a chat `messages` list on `/v1/completions` (TensorFold's does),
  and is the one that can send images there.

Many hosted APIs and proxies do neither; for those the probe stops with
`no readout is possible`.

1. Copy the nearest recipe under `recipes/`.
2. Set `model.name` to the id the server lists at `/v1/models`.
3. Set `endpoint.base_url` (it ends in `/v1`).
4. If the server needs a header, set `endpoint.headers`, for example
   `{x-client-id: agenticApp}`. Never put a secret there: a header whose
   value is a secret goes in `endpoint.headers_env` as header name to
   environment variable (`{X-Api-Key: GATEWAY_KEY}`), and a bearer key in the
   variable that `endpoint.api_key_env` names. A header name that looks like
   a credential (`key`, `token`, `secret`, `auth`, `cookie`, ...) is refused in
   `endpoint.headers`.
   `recipes/qwen38-27b.vllm.yaml` sends a header to a private router.
5. If the model reasons before it answers, turn reasoning off in
   `template.kwargs`. Try `{enable_thinking: false}` (Qwen3, GLM) or
   `{thinking: false}` (DeepSeek).
6. Set `provenance.status: draft` and delete the old `probe:` section.
7. Probe it. The probe fills in the token ids and sets the status to `probed`:

   ```sh
   uv run jevify probe recipes/<name>.yaml
   ```

8. Ask one question with `make ask`. Check that `degraded` is `false` and
   that the answer makes sense.

If no template switch turns reasoning off, write the prompt yourself with
`template.mode: raw` and the model's own control tokens.
`recipes/ternary-bonsai-2-27b.llamacpp.yaml` is an example.

## 7. Score a recipe

Run a recipe against a frozen suite of cases with known answers:

```sh
uv run jevify eval recipes/<name>.yaml suites/policy-29.jsonl --controls --concurrency 4
```

- The summary goes to `docs/evidence/<name>--policy-29.md`. Read it first.
- The raw decisions go to `runs/` (not committed).
- `--controls` adds two checks. **Shuffled state**: each case is asked
  against another case's state; lower agreement means the answers depend
  on the state. **Option permutation**: each case is asked with its options
  reversed; 1.00 means the choice never moved.
- Other suites: `policy-hard-52` (harder cases), `long-state-24` (2k to 32k
  tokens), `doom-frames-41/` (images).

A small suite ranks models. It does not prove that the probabilities are
calibrated.

## 8. Fix common problems

| You see | Cause | Do this |
|---|---|---|
| `missing required header 'x-client-id'` | The server needs a header the recipe does not send. | Add `endpoint.headers: {x-client-id: <your product id>}`. |
| `no readout is possible on this backend: it returned no logprobs` | The server or proxy returns no log-probabilities for the reply token, and the recipe could not read them through `prompt_logprobs`. | If the probe notes `reply-token logprobs refused`, write a raw-mode recipe so it can prove `prompt_logprobs` (step 6); a chat-template recipe proves it only where the server takes `messages` on `/v1/completions`. Otherwise use a server that returns them (vLLM, llama.cpp). |
| `endpoint.headers names X-Api-Key, which looks like a credential` | A secret would be written into the recipe. | Move it to `endpoint.headers_env: {X-Api-Key: <VARIABLE>}` and export the variable. |
| `endpoint.headers_env sends X-Api-Key from $GATEWAY_KEY, which is not set` | The variable the recipe names is not exported in this shell. | `export GATEWAY_KEY=...`, then run again. |
| `degraded: true`, options listed in `missing` | The options were not among the model's most likely tokens. Often the model starts to reason instead of answering. | Turn reasoning off (step 6, item 5), then probe again. |
| `off_menu_mass` close to 1 | The model wants to write something that is not an option. | Same as above. Check the question text too. |
| A context-limit error, or HTTP 413 from `serve` | The state is longer than the model's context. | Shorten the state. |
| `TypeSafeAPITimeoutError: Request timed out (timeout=10.0)` | The SDK's 10-second default is too short for a busy server. | Pass `timeout=60` to `TypeSafeClient`. |
| `address already in use` | Another server uses the port. | Use `PORT=8601`, or stop the other server. |
| A `$` amount vanishes from `STATE` | `make` read it as a variable. | Write `$$`, or use `STATE=@file`. |
| `make serve` refuses `HOST=0.0.0.0` | No `JEVIFY_API_KEY` is set. | Export a key first (step 5). |
| HTTP 401 `invalid api key` from `serve` | The server has a key and the client did not send it. | Send `Authorization: Bearer <key>`. |
