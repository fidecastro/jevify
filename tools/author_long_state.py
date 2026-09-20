"""Author suites/long-state-*.jsonl: one decisive customer line buried in a long support
transcript, at three positions, at four transcript lengths (about 2k, 8k, 16k and 32k
tokens). Deterministic filler; written before any model saw it; the manifest hash freezes it.

Run: uv run python tools/author_long_state.py
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "suites"
INSTR = "Select the action that satisfies the customer's stated request without making unwanted changes."
OPTIONS = [
    "Cancel the subscription.",
    "Keep the subscription active and change the plan to monthly.",
    "Issue a refund for the last invoice.",
]
DECISIVE = {
    "monthly": (
        "Customer: To be clear, I do not want to cancel anything. I want to move from annual to monthly billing.",
        1,
    ),
    "cancel": (
        "Customer: I have decided. Please cancel the subscription today; I do not want another plan or a refund.",
        0,
    ),
}
AGENT = [
    "Agent: Thanks for waiting, I am checking the account now.",
    "Agent: I can see the account was created on the annual plan.",
    "Agent: There are two devices registered, a laptop and a phone.",
    "Agent: Your last invoice was paid on the first of the month.",
    "Agent: Do you use the mobile app or mostly the website?",
    "Agent: I also see a support ticket from last spring about a password reset.",
    "Agent: Let me pull the billing history for the last twelve months.",
    "Agent: Everything looks consistent, one charge per year, same card.",
    "Agent: The notification settings show email and push both enabled.",
    "Agent: Your storage usage is well within the plan's allowance.",
    "Agent: I have added a note to the account about this conversation.",
    "Agent: The outage on the fourth affected the region you are in; it is resolved.",
]
CUSTOMER = [
    "Customer: Sure, take your time.",
    "Customer: Yes, that sounds right.",
    "Customer: The phone is my old one, you can ignore it.",
    "Customer: I remember, the receipt came through fine.",
    "Customer: Mostly the website, the app only for notifications.",
    "Customer: That got sorted, no problems since.",
    "Customer: Okay.",
    "Customer: Right.",
    "Customer: Fine by me.",
    "Customer: I had not noticed, thanks for checking.",
    "Customer: Good to know.",
    "Customer: Understood.",
]
# words per transcript, roughly 0.75 tokens per word on these tokenizers: 2k, 8k, 16k, 32k tokens
WORD_TARGETS = {"2k": 1500, "8k": 6000, "16k": 12000, "32k": 24000}

rng = random.Random(20260920)


def transcript(words: int, decisive: str, position: str) -> str:
    lines: list[str] = []
    count = 0
    while count < words:
        line = rng.choice(AGENT) if len(lines) % 2 == 0 else rng.choice(CUSTOMER)
        lines.append(line)
        count += len(line.split())
    index = {"start": 0, "middle": len(lines) // 2, "end": len(lines)}[position]
    lines.insert(index, decisive)
    return "Ticket transcript:\n" + "\n".join(lines)


cases = []
for length, words in WORD_TARGETS.items():
    for position in ("start", "middle", "end"):
        for key, (line, label) in DECISIVE.items():
            cases.append(
                {
                    "instruction": INSTR,
                    "context": transcript(words, line, position),
                    "options": OPTIONS,
                    "label": label,
                    "case_id": f"{length}_{position}_{key}",
                    "group": length,
                }
            )
name = f"long-state-{len(cases)}"
path = OUT / f"{name}.jsonl"
path.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases), encoding="utf-8")
manifest = {
    "name": name,
    "file": path.name,
    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    "cases": len(cases),
    "groups": {k: sum(1 for c in cases if c["group"] == k) for k in WORD_TARGETS},
    "question_type": "choice",
    "source": "authored for jevify on 2026-09-20 by tools/author_long_state.py (deterministic, seed 20260920)",
    "construction": (
        "one of two decisive customer lines (switch to monthly; cancel) inserted at the start, middle or "
        "end of a filler support transcript of about 2k, 8k, 16k or 32k tokens; groups are the lengths. "
        "A backend whose context is shorter than the state records an error, which is the finding."
    ),
    "expected_answers_by": "human (the author of the cases)",
    "measures": "whether a decisive line survives a long state at each position and length; not calibration",
    "known_results": "none at authoring time",
}
(OUT / f"{name}.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
sizes = {k: len(transcript(w, "x", "end").split()) for k, w in WORD_TARGETS.items()}
print(name, manifest["groups"], manifest["sha256"][:16], "words", sizes)
