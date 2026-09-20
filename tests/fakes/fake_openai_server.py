"""A switchable fake OpenAI-compatible server for adapter tests.

It is the CI stand-in for vLLM and llama.cpp: a Starlette app served to the
adapter through `httpx.ASGITransport`, with a `Behaviour` that flips every
server quirk the probe and the readout ladder must handle. The "model" is a
table of logits per answer token plus a fixed background of filler tokens,
so every expected number in a test is a literal derived from that table.
"""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass, field
from typing import Any, Literal

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

FILLER_TOKENS = ("the", ",", "\n", "I", " it")
THINK_OPEN = "<think>"


@dataclass
class Behaviour:
    dialect: Literal["vllm", "llamacpp", "generic"] = "vllm"
    top_k_cap: int = 5
    logprobs_mode: Literal["raw", "processed"] = "raw"
    supports_logprob_token_ids: bool = True
    supports_tokenize: bool = True
    thinking_kwarg: str | None = "thinking"
    thinking_default_on: bool = True
    prefill: Literal["honored", "eos_appended", "rejected"] = "eos_appended"
    max_context: int = 512
    block_tokens: int = 256
    report_cached_tokens: bool = True
    background_logit: float = -3.0
    scores: dict[str, float] = field(default_factory=dict)
    # (substring, scores): the first substring found in the prompt selects that score table.
    scores_when: list[tuple[str, dict[str, float]]] = field(default_factory=list)
    rate_limit_first: int = 0
    version: str = "0.99.0-fake"
    # answer texts that tokenize to two tokens on this "model" (probe must catch them)
    multi_token_answers: set[str] = field(default_factory=set)


class FakeTokenizer:
    """Deterministic: known answer tokens get fixed ids, everything else a stable hash id."""

    KNOWN = {
        " yes": 14452,
        "yes": 16520,
        " no": 1119,
        "no": 3567,
        " A": 334,
        " B": 406,
        " C": 356,
        " D": 423,
        "A": 32,
        "B": 33,
        " Yes": 11608,
        " No": 3011,
        THINK_OPEN: 99001,
        "</think>": 99002,
    }

    def id_of(self, text: str) -> int:
        if text in self.KNOWN:
            return self.KNOWN[text]
        return 100_000 + int.from_bytes(hashlib.sha1(text.encode()).digest()[:3], "big")

    @staticmethod
    def pieces(text: str) -> list[str]:
        """Word pieces that keep their leading space, like a byte-level BPE would."""
        return re.findall(r"\n|\s?[^\s]+", text)

    def encode(self, text: str) -> list[int]:
        return [self.id_of(p) for p in self.pieces(text)]


class FakeOpenAIServer:
    def __init__(self, behaviour: Behaviour | None = None) -> None:
        self.behaviour = behaviour or Behaviour()
        self.tokenizer = FakeTokenizer()
        self.requests: list[dict[str, Any]] = []
        self.seen_blocks: set[tuple[int, ...]] = set()
        self._served = 0
        routes = [
            Route("/v1/chat/completions", self.chat, methods=["POST"]),
            Route("/v1/completions", self.completions, methods=["POST"]),
            Route("/v1/models", self.models, methods=["GET"]),
            Route("/tokenize", self.tokenize, methods=["POST"]),
        ]
        if self.behaviour.dialect == "vllm":
            routes.append(Route("/version", self.version, methods=["GET"]))
        if self.behaviour.dialect == "llamacpp":
            routes.append(Route("/props", self.props, methods=["GET"]))
            routes.append(Route("/apply-template", self.apply_template, methods=["POST"]))
        self.app = Starlette(routes=routes)

    # ---------------------------------------------------------------- the "model"
    def _distribution(self, prompt_tail: str, body: dict[str, Any]) -> dict[str, float]:
        """Log-probabilities over the whole (tiny) vocabulary at the answer position."""
        b = self.behaviour
        if prompt_tail.endswith(THINK_OPEN):
            logits = {"\n": 0.0, "Okay": -0.5}
        else:
            logits = dict(b.scores)
            for marker, table in b.scores_when:
                if marker in prompt_tail:
                    logits = dict(table)
                    break
            for filler in FILLER_TOKENS:
                logits.setdefault(filler, b.background_logit)
        if b.logprobs_mode == "processed":
            for key, bias in (body.get("logit_bias") or {}).items():
                text = self._text_for_bias_key(key)
                if text in logits:
                    logits[text] += float(bias)
        peak = max(logits.values())
        total = sum(math.exp(v - peak) for v in logits.values())
        return {t: (v - peak) - math.log(total) for t, v in logits.items()}

    def _text_for_bias_key(self, key: str) -> str:
        try:
            wanted = int(key)
        except ValueError:
            return key
        for text, ident in self.tokenizer.KNOWN.items():
            if ident == wanted:
                return text
        return key

    def _entries(self, logprobs: dict[str, float], body: dict[str, Any]) -> list[dict[str, Any]]:
        b = self.behaviour
        ranked = sorted(logprobs.items(), key=lambda kv: kv[1], reverse=True)
        requested = int(body.get("top_logprobs") or body.get("logprobs") or 0)
        wanted_ids = body.get("logprob_token_ids") if b.supports_logprob_token_ids else None
        if wanted_ids and b.dialect == "vllm":
            chosen = [(t, lp) for t, lp in ranked if self.tokenizer.id_of(t) in set(wanted_ids)]
        else:
            chosen = ranked[: min(requested, b.top_k_cap)]
        entries = []
        for text, lp in chosen:
            entry: dict[str, Any] = {"token": text, "logprob": lp, "bytes": list(text.encode())}
            if b.dialect == "llamacpp":
                entry["id"] = self.tokenizer.id_of(text)
            if b.dialect == "vllm" and body.get("return_as_token_id"):
                entry["token"] = f"token_id:{self.tokenizer.id_of(text)}"
            entries.append(entry)
        return entries

    # ---------------------------------------------------------------- helpers
    def _rendered_chat(self, body: dict[str, Any]) -> str:
        parts = []
        for message in body["messages"]:
            content = message["content"]
            if isinstance(content, list):
                content = "".join(p.get("text", "<image>") for p in content)
            parts.append(f"<|{message['role']}|>{content}")
        kwargs = body.get("chat_template_kwargs") or {}
        thinking_on = self.behaviour.thinking_default_on
        if self.behaviour.thinking_kwarg and kwargs.get(self.behaviour.thinking_kwarg) is False:
            thinking_on = False
        last = body["messages"][-1]
        if last["role"] == "assistant" and body.get("continue_final_message"):
            if self.behaviour.prefill == "eos_appended":
                parts.append("<eos>")
            return "".join(parts)
        parts.append("<|assistant|>" + (THINK_OPEN if thinking_on else ""))
        return "".join(parts)

    def _cache_and_usage(self, prompt: str) -> dict[str, Any]:
        ids = self.tokenizer.encode(prompt)
        n = len(ids)
        cached = 0
        b = self.behaviour
        blocks = [tuple(ids[i : i + b.block_tokens]) for i in range(0, n, b.block_tokens)]
        full_blocks = [blk for blk in blocks if len(blk) == b.block_tokens]
        for blk in full_blocks:
            if blk in self.seen_blocks:
                cached += b.block_tokens
            else:
                break
        self.seen_blocks.update(full_blocks)
        usage: dict[str, Any] = {"prompt_tokens": n, "completion_tokens": 1, "total_tokens": n + 1}
        if b.report_cached_tokens and b.dialect == "vllm":
            usage["prompt_tokens_details"] = {"cached_tokens": cached, "multimodal_tokens": None}
        return usage

    def _gate(self, body: dict[str, Any], path: str) -> JSONResponse | None:
        self.requests.append({"path": path, **body})
        self._served += 1
        if self._served <= self.behaviour.rate_limit_first:
            return JSONResponse(
                {"error": {"message": "rate limited", "type": "rate_limit"}},
                status_code=429,
                headers={"retry-after": "0"},
            )
        return None

    def _too_long(self, prompt: str) -> JSONResponse | None:
        n = len(self.tokenizer.encode(prompt))
        if n > self.behaviour.max_context:
            return JSONResponse(
                {
                    "error": {
                        "message": (
                            f"This model's maximum context length is {self.behaviour.max_context} "
                            f"tokens. However, you requested {n + 1} tokens."
                        ),
                        "type": "BadRequestError",
                    }
                },
                status_code=400,
            )
        return None

    # ---------------------------------------------------------------- routes
    async def chat(self, request: Request) -> JSONResponse:
        body = await request.json()
        if (gated := self._gate(body, "/v1/chat/completions")) is not None:
            return gated
        prompt = self._rendered_chat(body)
        if (bad := self._too_long(prompt)) is not None:
            return bad
        logprobs = self._distribution(prompt, body)
        entries = self._entries(logprobs, body)
        sampled = max(logprobs.items(), key=lambda kv: kv[1])
        usage = self._cache_and_usage(prompt)
        content_entry = {"token": sampled[0], "logprob": sampled[1], "top_logprobs": entries}
        if self.behaviour.dialect == "llamacpp":
            content_entry["id"] = self.tokenizer.id_of(sampled[0])
        return JSONResponse(
            {
                "id": "chatcmpl-fake",
                "object": "chat.completion",
                "model": body.get("model", "fake"),
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": sampled[0]},
                        "logprobs": {"content": [content_entry]} if body.get("logprobs") else None,
                        "finish_reason": "length",
                    }
                ],
                "usage": usage,
                **(
                    {
                        "tokens_cached": usage.get("prompt_tokens_details", {}).get(
                            "cached_tokens", 0
                        )
                    }
                    if self.behaviour.dialect == "llamacpp"
                    else {}
                ),
            }
        )

    async def completions(self, request: Request) -> JSONResponse:
        body = await request.json()
        if (gated := self._gate(body, "/v1/completions")) is not None:
            return gated
        prompt = body["prompt"]
        if isinstance(prompt, list):
            prompt = " ".join(str(t) for t in prompt)
        if (bad := self._too_long(prompt)) is not None:
            return bad
        logprobs = self._distribution(prompt, body)
        entries = self._entries(logprobs, body)
        sampled = max(logprobs.items(), key=lambda kv: kv[1])
        usage = self._cache_and_usage(prompt)
        return JSONResponse(
            {
                "id": "cmpl-fake",
                "object": "text_completion",
                "model": body.get("model", "fake"),
                "choices": [
                    {
                        "index": 0,
                        "text": sampled[0],
                        "logprobs": {
                            "tokens": [sampled[0]],
                            "token_logprobs": [sampled[1]],
                            "top_logprobs": [{e["token"]: e["logprob"] for e in entries}],
                        }
                        if body.get("logprobs") is not None
                        else None,
                        "finish_reason": "length",
                    }
                ],
                "usage": usage,
            }
        )

    async def models(self, request: Request) -> JSONResponse:
        return JSONResponse(
            {
                "object": "list",
                "data": [
                    {
                        "id": "fake-model",
                        "object": "model",
                        "owned_by": self.behaviour.dialect,
                        **(
                            {"meta": {"n_ctx_train": 10 * self.behaviour.max_context}}
                            if self.behaviour.dialect == "llamacpp"
                            else {"max_model_len": self.behaviour.max_context}
                        ),
                    }
                ],
            }
        )

    async def version(self, request: Request) -> JSONResponse:
        return JSONResponse({"version": self.behaviour.version})

    async def props(self, request: Request) -> JSONResponse:
        return JSONResponse(
            {
                "build_info": f"b{self.behaviour.version}",
                "total_slots": 2,
                "default_generation_settings": {"n_ctx": self.behaviour.max_context},
            }
        )

    async def tokenize(self, request: Request) -> JSONResponse:
        if not self.behaviour.supports_tokenize:
            return JSONResponse({"detail": "Not Found"}, status_code=404)
        body = await request.json()
        if self.behaviour.dialect == "llamacpp":
            ids = self.tokenizer.encode(body["content"])
            if body["content"] in self.behaviour.multi_token_answers:
                ids = ids + [self.tokenizer.id_of(body["content"] + "#2")]
            if body.get("with_pieces"):
                return JSONResponse(
                    {
                        "tokens": [
                            {"id": i, "piece": p}
                            for i, p in zip(ids, body["content"].split(" "), strict=False)
                        ]
                    }
                )
            return JSONResponse({"tokens": ids})
        text = body.get("prompt")
        if text is None:
            text = self._rendered_chat(body)
        ids = self.tokenizer.encode(text)
        if body.get("prompt") is not None and self.behaviour.multi_token_answers:
            # the switch: a chosen answer text splits into two tokens on this "model"
            if text in self.behaviour.multi_token_answers:
                ids = ids + [self.tokenizer.id_of(text + "#2")]
        return JSONResponse(
            {
                "count": len(ids),
                "max_model_len": self.behaviour.max_context,
                "tokens": ids,
                "token_strs": text.replace("\n", " \n ").split(" ")
                if body.get("return_token_strs")
                else None,
            }
        )

    async def apply_template(self, request: Request) -> JSONResponse:
        body = await request.json()
        return JSONResponse({"prompt": self._rendered_chat(body)})
