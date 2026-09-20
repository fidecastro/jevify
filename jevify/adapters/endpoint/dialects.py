"""The dialect registry: how each server spells the OpenAI surface and its extras.

The only file where the dialect names appear as behaviour. The adapter asks a
dialect to shape a request and to parse a response; it never branches on the
name itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jevify.ports.backend import Dialect
from jevify.recipes.render import RenderedPrefix


@dataclass(frozen=True)
class Entry:
    """One token at the answer position: decoded text, id when the server gives one, logprob."""

    text: str
    logprob: float
    id: int | None = None


@dataclass(frozen=True)
class Parsed:
    entries: tuple[Entry, ...]
    prompt_tokens: int | None
    cached_tokens: int | None
    completion_tokens: int | None


def base_request(
    model: str, prefix: RenderedPrefix, suffix: str, dialect: Dialect
) -> tuple[str, dict[str, Any]]:
    """The readout request every rung starts from: one token, no sampling, logprobs on.

    Returns the route key and the body: "chat/completions" and "completions" live under
    /v1; "completion" is llama.cpp's native root route, used for raw multimodal input.
    """
    if prefix.mode == "raw":
        if prefix.images:
            if dialect is not Dialect.LLAMACPP:
                raise ValueError("raw-mode images need the llama.cpp native completion route")
            # Verified live on build 10809: top-level prompt string with one media marker per
            # image, plus a top-level multimodal_data list of raw base64 payloads.
            return "completion", {
                "prompt": prefix.text + suffix,
                "multimodal_data": [_base64_payload(i.data_uri) for i in prefix.images],
                "n_predict": 1,
                "temperature": 0,
                "stream": False,
            }
        body: dict[str, Any] = {
            "model": model,
            "prompt": prefix.text + suffix,
            "max_tokens": 1,
            "temperature": 0,
            "stream": False,
        }
        return "completions", body
    content: str | list[dict[str, Any]]
    if prefix.images:
        content = [{"type": "text", "text": prefix.text}]
        content.extend(
            {"type": "image_url", "image_url": {"url": image.data_uri}} for image in prefix.images
        )
        content.append({"type": "text", "text": suffix})
    else:
        content = prefix.text + suffix
    messages: list[dict[str, Any]] = []
    if prefix.system:
        messages.append({"role": "system", "content": prefix.system})
    messages.append({"role": "user", "content": content})
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": 1,
        "temperature": 0,
        "stream": False,
        "logprobs": True,
    }
    if prefix.prefill:
        messages.append({"role": "assistant", "content": prefix.prefill})
        body["continue_final_message"] = True
        body["add_generation_prompt"] = False
    if prefix.kwargs and dialect is not Dialect.GENERIC:
        body["chat_template_kwargs"] = dict(prefix.kwargs)
    return "chat/completions", body


def _base64_payload(data_uri: str) -> str:
    return data_uri.split(",", 1)[1] if data_uri.startswith("data:") else data_uri


def neutral_samplers(body: dict[str, Any]) -> None:
    """llama.cpp only: post-sampling probabilities are read after the sampler chain, so the
    chain must not truncate or reshape the distribution (the server's defaults do)."""
    body.update({"temperature": 1.0, "top_k": 0, "top_p": 1.0, "min_p": 0.0})
    body["post_sampling_probs"] = True


def grammar_for(texts: list[str]) -> str:
    """A GBNF root that admits exactly the answer spellings."""
    escaped = " | ".join('"' + t.replace("\\", "\\\\").replace('"', '\\"') + '"' for t in texts)
    return f"root ::= {escaped}"


def parse_response(path: str, response: dict[str, Any], dialect: Dialect) -> Parsed:
    entries: list[Entry] = []
    if path == "completion":
        # llama.cpp native route: completion_probabilities plus top-level counters.
        items = response.get("completion_probabilities") or []
        if items:
            for item in items[0].get("top_logprobs") or items[0].get("top_probs") or []:
                entries.append(_entry_from(item, dialect))
        return Parsed(
            entries=tuple(entries),
            prompt_tokens=response.get("tokens_evaluated"),
            cached_tokens=response.get("tokens_cached"),
            completion_tokens=response.get("tokens_predicted"),
        )
    choice = response["choices"][0]
    logprobs = choice.get("logprobs") or {}
    # Shape-driven, not dialect-driven: llama.cpp answers the completions route with the
    # chat-style `content` list (ids included); OpenAI and vLLM use the dict form there.
    if "content" in logprobs:
        content = logprobs.get("content") or []
        if content:
            for item in content[0].get("top_logprobs") or content[0].get("top_probs") or []:
                entries.append(_entry_from(item, dialect))
    else:
        top = (logprobs.get("top_logprobs") or [{}])[0] or {}
        entries = [Entry(text=text, logprob=float(lp)) for text, lp in top.items()]
    usage = response.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens")
    if cached is None and "tokens_cached" in response:
        cached = response.get("tokens_cached")
    return Parsed(
        entries=tuple(entries),
        prompt_tokens=usage.get("prompt_tokens"),
        cached_tokens=cached,
        completion_tokens=usage.get("completion_tokens"),
    )


def _entry_from(item: dict[str, Any], dialect: Dialect) -> Entry:
    text = str(item.get("token", ""))
    ident = item.get("id")
    if ident is None and text.startswith("token_id:"):
        try:
            ident = int(text[len("token_id:") :])
        except ValueError:
            ident = None
    value = item.get("logprob")
    if value is None and "prob" in item:
        import math

        value = math.log(item["prob"]) if item["prob"] > 0 else -1e4
    return Entry(text=text, logprob=float(value), id=ident)
