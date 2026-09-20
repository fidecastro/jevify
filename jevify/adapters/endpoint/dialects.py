"""The dialect registry: how each server spells the OpenAI surface and its extras.

The only file where the dialect names appear as behaviour. The adapter asks a
dialect to shape a request and to parse a response; it never branches on the
name itself.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jevify.ports.backend import Dialect
from jevify.recipes.render import RenderedPrefix, RenderedQuestion


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
    model: str, prefix: RenderedPrefix, question: RenderedQuestion, dialect: Dialect
) -> tuple[str, dict[str, Any]]:
    """The readout request every rung starts from: one token, no sampling, logprobs on."""
    if prefix.mode == "raw":
        body: dict[str, Any] = {
            "model": model,
            "prompt": prefix.text + question.text,
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
        content.append({"type": "text", "text": question.text})
    else:
        content = prefix.text + question.text
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


def parse_response(path: str, response: dict[str, Any], dialect: Dialect) -> Parsed:
    choice = response["choices"][0]
    logprobs = choice.get("logprobs") or {}
    entries: list[Entry] = []
    if path == "completions":
        top = (logprobs.get("top_logprobs") or [{}])[0] or {}
        entries = [Entry(text=text, logprob=float(lp)) for text, lp in top.items()]
    else:
        content = logprobs.get("content") or []
        if content:
            for item in content[0].get("top_logprobs") or []:
                entries.append(_entry_from(item, dialect))
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
