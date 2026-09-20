# ADR-0006 — Toolchain, dependency budget and portability

- **Status:** accepted (2026-09-19).
- **Extends:** ADR-0001 D4 (Python floor, "nothing heavier than an HTTP
  client"); makes that sentence precise and records what the core may depend
  on. Supports the founder's requirement that deployment is easy and the
  installed CLI is light.

## Context

The founder wants a tinkerer to install one command and try a model. That
rules out a core that pulls PyTorch, and it rules out a server that needs a
compiler. At the same time the project ships a server, a client, YAML
recipes and schemas that must match Jev's SDK field for field, which is
pydantic territory. The machine this starts on runs Python 3.14 with no
`uv` or `pip` on the path; every dependency below has wheels for 3.12 through
3.14 on Linux and macOS, and PyTorch 2.14 already runs on 3.14 here.

## Decision

### D1 — Core dependencies, and nothing else

| Package | Why it is in core |
|---|---|
| `httpx` | The one HTTP client. Async and sync, per-request timeouts, and an ASGI transport that lets every adapter test run against the fake server in-process without sockets. |
| `pydantic>=2.12` | Three schemas need validation with field-named errors: the recipe, Jev's wire format, the scorecard. Jev's SDK is pydantic, and the pinned-mirror test compares model fields directly. Strict mode reproduces the SDK's no-coercion behaviour locally. |
| `PyYAML` | Recipes are YAML (ADR-0004) and the probe writes them back. Hashing is over canonical JSON of the parsed document, so YAML formatting and comments never affect identity. |
| `starlette` | The HTTP server for `jevify serve`. Serving is a primary command, so it is not an extra. Pydantic validation is called directly; the FastAPI-style error body is a few lines. |
| `uvicorn` | The ASGI server, plain build with no `uvloop` or `httptools`, so nothing compiles at install time. |

The CLI uses `argparse`. Metrics are computed with the standard library. No
NumPy, no web framework beyond Starlette, no tokenizer library, no model
runtime in core.

### D2 — Extras

- `encoder`: `torch`, `transformers`, `safetensors`. Needed only by the
  in-process encoder, reranker and embedding kinds.
- `dev`: `pytest`, `ruff`, `mypy`, `typesafe-sdk==0.7.0`. The SDK depends on
  `httpx2`, a different distribution from `httpx`, so both coexist.

### D3 — Python versions

`requires-python = ">=3.12"`. The 3.12 floor from ADR-0001 stands for
DevCake alignment; 3.14 satisfies it and is where local development happens.
CI runs 3.12 and 3.14 on Linux and macOS. The `encoder` extra is exercised
in CI on Linux 3.12 only until its 3.14 wheels are confirmed there. No
3.14-only syntax enters the tree; ruff's target version enforces it.

### D4 — Build and distribution

`hatchling` builds the package. The console script is `jevify`. Users
install with `uv tool install jevify` or `pipx install jevify`; either gives
an isolated environment and the command on the path regardless of the system
Python. Developers use `uv sync --all-extras` and `uv run pytest`; the lock
file is committed. On this host `uv` comes from `mise`, which is a convenience
of this machine and not a project requirement.

### D5 — Portability rules

- Core runs wherever Python 3.12 runs: Linux x86_64 and arm64, macOS on
  Apple silicon and Intel, Windows best-effort. No GPU, compiler, CUDA or
  vendor SDK in core.
- No backend lock-in: the generic endpoint dialect uses only the OpenAI
  surface; server-specific extras are detected by the probe, never assumed.
- No tokenizer in core: servers without a tokenize endpoint get text matching
  and `token_ids_verified: false`; an optional extra may add exact ids later.
- No host coupling: paths use `pathlib`, run outputs are relative to the
  working directory, the handle store is in-process, nothing assumes `/tmp`,
  symlinks or a shell.

## Consequences

- Five light core packages. A fresh `uv tool install jevify` is seconds, not
  minutes.
- Starlette rather than FastAPI means request validation is explicit code
  in `api/app.py`, which is also where the SDK's error shape is produced.
- Anyone who wants the in-process kinds installs the extra and accepts
  PyTorch's footprint; the endpoint path never pays it.
- The dev extra pins the SDK; bumping it is a deliberate change that reruns
  the mirror test.

## Rejected alternatives

- **FastAPI.** Convenient, DevCake uses it, but it adds nothing the plan
  needs and one more dependency on the request path.
- **Typer or Click for the CLI.** Autocompletion and colours were not asked
  for; `argparse` has no dependencies.
- **JSON or TOML recipes.** JSON has no comments and tinkerers edit recipes
  by hand; TOML has no writer in the standard library. YAML with canonical
  JSON hashing keeps both hand-editing and stable identity.
- **A single mandatory `torch` install.** Would make the light CLI heavy for
  the majority who only talk to endpoints.
