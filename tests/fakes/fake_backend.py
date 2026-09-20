"""A scripted Backend for engine and API tests: no HTTP, no prompts, known numbers."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from jevify.domain.questions import Question, State
from jevify.ports.backend import Capabilities, RawAnswer, Rung, StateHandle
from jevify.recipes.render import RenderedPrefix


@dataclass
class FakeBackend:
    """Answers each question id with the logprobs scripted for it."""

    scripted: dict[str, dict[str, float]]
    prompt_tokens: int = 100
    rung: Rung = Rung.TOP_K
    warmed: list[State] = field(default_factory=list)
    evaluated: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)

    async def probe(self) -> Capabilities:
        return Capabilities(kind="endpoint", model="fake", rungs=frozenset({self.rung}))

    async def warm(self, state: State) -> StateHandle:
        self.warmed.append(state)
        prefix = RenderedPrefix(mode="messages", text=state.text)
        return StateHandle(state_id=f"state-{len(self.warmed)}", prefix=prefix, warm_ms=1.5)

    async def evaluate(self, handle: StateHandle, questions: Sequence[Question]) -> list[RawAnswer]:
        answers = []
        for question in questions:
            self.evaluated.append((handle.state_id, tuple(q.id for q in questions)))
            logprobs = self.scripted[question.id]
            answers.append(
                RawAnswer(
                    question_id=question.id,
                    logprobs=dict(logprobs),
                    semantics="readout",
                    rung=self.rung,
                    latency_ms=2.0,
                    prompt_tokens=self.prompt_tokens,
                    off_menu_mass=0.01,
                )
            )
        return answers
