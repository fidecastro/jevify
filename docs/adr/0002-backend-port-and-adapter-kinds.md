# ADR-0002 — One backend port, four adapter kinds, a capability probe, a readout ladder

- **Status:** accepted (2026-09-19).
- **Extends:** ADR-0001 D2 (scope). Implements INV-2, INV-3, INV-5, INV-9.

## Context

The models a user may bring differ in what they expose. An endpoint-served
decoder shows a distribution only at generated positions and only its top
few tokens. An in-process encoder classifier such as Laya or GLiClass has a
head that scores option slots directly. A reranker returns one independent
relevance score per pair. An embedding model returns vectors. The core must
not know which one it is talking to, and the user must not write code to
switch.

Three facts about the endpoint route, verified on 2026-09-19, determine its
design.

- The plain OpenAI-compatible surface supports a readout: one generated token
  with `logprobs` and `top_logprobs`, and `logit_bias`. Adding the **same**
  bias to every answer token surfaces them into the returned list without
  changing their ratios, because softmax is shift-invariant within the biased
  set. Some servers report logprobs before bias is applied, which the probe
  must detect.
- vLLM's sampling parameters include `logprob_token_ids`, which returns exact
  logprobs for named token ids regardless of top-k
  ([source](https://github.com/vllm-project/vllm/blob/main/vllm/sampling_params.py)).
  vLLM also caches prefixes automatically and can share a prefix's attention
  read across a batch of branches.
- llama.cpp's server exposes `n_probs`, `post_sampling_probs` (probabilities
  after the sampling chain, so a grammar restricted to the answer tokens yields
  a renormalized distribution), `logit_bias`, GBNF grammars, a `/apply-template`
  endpoint that renders the chat template without inference, a `tokens_cached`
  count in every response, and slot save/restore to disk
  ([README](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)).
  Slots do not share KV, so fan-out is sequential per slot or N copies.
  Prompt-token logprobs are pending in
  [PR #27537](https://github.com/ggml-org/llama.cpp/pull/27537), and a
  prefill-time masked distribution is proposed in
  [issue #29022](https://github.com/ggml-org/llama.cpp/issues/29022).

The precursor measured that relevance-trained models fail instruction flips
at every size tested, and that rerank endpoints re-encode the state once per
pair, so a reranker served that way cannot be sub-second at long context.

## Decision

### D1 — One port

`jevify.ports.backend.Backend` is a Protocol with three operations:

- `probe() -> Capabilities` — measure and return what this backend and model
  can do (D3). Called once per recipe and recorded in the recipe.
- `warm(state) -> StateHandle` — encode the state so later questions reuse it.
  For backends without a reusable encoding this returns a handle that only
  carries the rendered state.
- `evaluate(handle, questions) -> list[RawAnswer]` — answer a batch of
  questions against one state. A `RawAnswer` is a distribution over the answer
  set plus the semantics label, the readout method used and any off-menu mass
  observed.

The core calls nothing else. Domain code depends on this Protocol and never
on an adapter module; a structure guard enforces the import ban.

### D2 — Four adapter kinds

| Kind | Talks to | Readout | Semantics |
|---|---|---|---|
| `endpoint` | Any OpenAI-compatible HTTP server | One generated token; distribution from logprobs at the answer position | `readout` |
| `encoder` | An in-process encoder classifier loaded through Transformers | The head's scores at option marker slots (Laya's layout) or label logits (GLiClass, NLI) | `readout` |
| `rerank` | A rerank or score endpoint, or a reranker model in-process | One independent relevance score per option | `relevance` |
| `embedding` | An embeddings endpoint or model | Cosine between instruction-conditioned state vector and cached option vectors | `similarity` |

The `endpoint` kind carries **dialects**, detected by the probe and never
chosen by the core: `vllm` unlocks named-token logprobs, template arguments and
assistant prefill; `llamacpp` unlocks server-side template rendering, grammar
with post-sampling probabilities, the cached-token count and slot persistence;
`generic` uses only the OpenAI surface.

A reranker that is a causal model, such as the Qwen3-VL rerankers, is by
default driven through the `endpoint` kind with the reranker's own template
so that the state becomes a shared prefix and each option a suffix. The
`rerank` kind, which calls the pair-scoring endpoint, remains for rerankers
that are not causal models and as a control.

### D3 — The capability probe

The probe is a measurement, not a lookup table. For an endpoint it sends
requests and records:

- whether each answer token under the recipe's template is a single token
  (via the tokenizer or the server's tokenize endpoint);
- whether all answer tokens come back in the logprobs unbiased, biased, or
  only through named-token logprobs or a grammar;
- whether assistant prefill and template arguments (thinking off) are honored;
- whether a second request over the same state is measurably faster than the
  first, which is the only proof that the prefix is cached;
- whether image and video content parts are accepted;
- the context limit the server reports or the first length at which it fails;
- whether N concurrent branches complete faster than N sequential ones, which
  decides fan-out.

For in-process kinds it records the model's maximum length, the option budget,
the modalities, and the device. The result is written into the recipe and
into every scorecard.

### D4 — The readout ladder

For the `endpoint` kind the core requests a distribution over the answer set
and the adapter obtains it by the strongest method the capability list
allows, in this order:

1. named-token logprobs (exact, no truncation);
2. a grammar or allowed-token mask with post-sampling probabilities
   (exact over the answer set, off-menu mass hidden);
3. an unbiased top-k request when every answer token is present (exact, and
   off-menu mass visible);
4. equal bias on all answer tokens (exact ratios, off-menu mass hidden);
5. top-k with a floor for missing tokens, marked degraded in the answer.

The method used is recorded on every answer. A recipe may pin a rung; it may
not skip the ladder's recording.

Amended 2026-09-20, after the first hybrid model on llama.cpp: `auto` walks
the proven rungs rather than picking one. When the strongest rung reads the
answer slot but not every label (a label outside the top-k), the next proven
rung takes the question, down to the floor; the answer names the rung that
read it. A pinned rung never walks and fails loudly, as before. The probe
also proves two things it used to assume: that an image adds prompt tokens
(a server can accept an image and drop it), and how much of a warm prefix a
question request reuses (a recurrent model resumes only from a checkpoint at
the end of an earlier prompt, so a chat-template warm can buy nothing).

### D5 — Choice questions on decoders

Options are labeled with single-token identifiers and the distribution is
read over the identifiers. Multi-token option strings are never generated
(INV-3). Position bias is handled by a permutation policy declared in the
recipe: `none`, `cyclic` with k calls averaged, or `full` for small k. The
policy and the number of calls are recorded on the answer.

### D6 — Two-phase execution and fan-out

`warm` encodes the state once. `evaluate` issues one request per question
(INV-5). If the probe found that concurrent branches are faster and the
backend caches prefixes, branches are issued concurrently; otherwise
sequentially on one connection. Both paths are one code path with a
concurrency parameter, not two implementations.

### D7 — Deferred adapters, with the port shaped for them

- **In-process decoder engine** (vLLM's offline engine): gives exact
  multi-token option scoring and guaranteed shared-state batching. Deferred
  until a scorecard shows the endpoint kind cannot meet a latency or exactness
  requirement. The port's batch-evaluate signature already fits it.
- **Diffusion answer slots** (vLLM PR #57250, OpenJev): native multi-question
  single pass. Deferred until merged upstream. `evaluate` taking a batch of
  questions is what lets it plug in.

## Consequences

- The probe is where the engineering cost of "any endpoint" lives. It is
  tested against live servers in CI, not against fakes alone.
- Evaluations are not bit-reproducible across backends, and on llama.cpp not
  across batch sizes. Scorecards compare at the level of decisions and record
  the backend and its flags.
- A head-based encoder model can only be served in-process, which is
  acceptable because such models are small.
- The generic dialect on a server that reports pre-bias logprobs and lacks
  grammar support degrades to rung 5 and says so.

## Rejected alternatives

- **In-process first, endpoints later.** Would honor every property on day one
  but makes jevify the server, forfeits llama.cpp and remote models, and is not
  where the founder's use is. Deferred, not rejected (D7).
- **A separate adapter per server dialect.** Three implementations of one
  process; rejected under the chokepoint principle. Dialects are capability
  sets inside one adapter.
- **Scoring multi-token options by generating them under a grammar.** Gives a
  valid argmax and no distribution over the options; violates INV-3.
- **Driving rerankers only through the rerank endpoint.** Re-encodes the state
  per option; kept only as a control.
