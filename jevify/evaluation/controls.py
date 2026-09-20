"""Scorecard controls (ADR-0005 D3): the checks that make a main result interpretable.

- shuffled context: every case is answered against another case's state; agreement with the
  original label should collapse if the answers depend on the state at all.
- option permutation: every choice case is asked again with its options reversed; the share
  of unchanged selections measures position stability.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from typing import Any

from jevify.domain.engine import Engine
from jevify.evaluation.scorecard import Decision, run_case
from jevify.evaluation.suites import Suite, SuiteRow


async def _run(engine: Engine, rows: list[SuiteRow], concurrency: int) -> list[Decision]:
    gate = asyncio.Semaphore(max(1, concurrency))

    async def one(row: SuiteRow) -> Decision:
        async with gate:
            return await run_case(engine, row)

    return list(await asyncio.gather(*(one(row) for row in rows)))


async def shuffled_context(engine: Engine, suite: Suite, *, concurrency: int = 1) -> dict[str, Any]:
    rows = list(suite.rows)
    swapped = [
        replace(row, context=rows[(index + 1) % len(rows)].context)
        for index, row in enumerate(rows)
    ]
    decisions = await _run(engine, swapped, concurrency)
    scored = [d for d in decisions if d.label is not None and not d.error]
    agreement = sum(d.selected == d.label for d in scored) / max(1, len(scored))
    return {
        "cases": len(scored),
        "agreement_with_label": agreement,
        "failed": [{"case_id": d.case_id, "error": d.error} for d in decisions if d.error],
        "note": "each case answered against the next case's state; low agreement means the "
        "answers depend on the state",
    }


async def option_permutation(
    engine: Engine, suite: Suite, main: list[Decision], *, concurrency: int = 1
) -> dict[str, Any]:
    rows = [row for row in suite.rows if row.question_type == "choice"]
    reversed_rows = [
        replace(
            row,
            options=tuple(reversed(row.options)),
            label=(len(row.options) - 1 - row.label) if row.label is not None else None,
        )
        for row in rows
    ]
    decisions = await _run(engine, reversed_rows, concurrency)
    by_id = {d.case_id: d for d in main}
    pairs = [(by_id[d.case_id], d) for d in decisions if d.case_id in by_id and not d.error]
    same = sum(original.selected == permuted.selected for original, permuted in pairs)
    return {
        "cases": len(pairs),
        "same_choice": same / max(1, len(pairs)),
        "failed": [{"case_id": d.case_id, "error": d.error} for d in decisions if d.error],
        "note": "each choice case asked again with its options reversed; the share of "
        "unchanged selections measures position stability",
    }
