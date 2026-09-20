"""Author suites/policy-hard-*.jsonl: instruction flips, negations, two-threshold policies,
distractors and long states with the decisive fact buried. Written before any model saw
it; append-only after the first scored run (the manifest hash freezes it).

Run: uv run python tools/author_policy_hard.py
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "suites"
cases: list[dict] = []


def case(group: str, instruction: str, context: str, options: list[str], label: int) -> None:
    assert 0 <= label < len(options) and len(set(options)) == len(options)
    cases.append(
        {
            "instruction": instruction,
            "context": context,
            "options": options,
            "label": label,
            "case_id": f"{group}_{sum(1 for c in cases if c['group'] == group)}",
            "group": group,
        }
    )


# ---------------------------------------------------------------- flips: the instruction inverts the natural reading
FLIP_STATE = "Customer: Keep my subscription active, but stop the promotional emails."
FLIP_OPTS = [
    "Cancel the subscription.",
    "Disable promotional emails and keep the subscription active.",
    "Keep promotional emails enabled.",
]
case("flip", "Select the action that satisfies the customer's request.", FLIP_STATE, FLIP_OPTS, 1)
case("flip", "Select the action the customer explicitly asked NOT to take.", FLIP_STATE, FLIP_OPTS, 0)
case("flip", "Select the action that would most clearly violate the customer's request.", FLIP_STATE, FLIP_OPTS, 0)
case("flip", "Select the option that changes nothing the customer complained about.", FLIP_STATE, FLIP_OPTS, 2)

TEAM_STATE = "I was charged twice for my subscription. I want the extra payment returned."
TEAM_OPTS = ["Billing and refunds", "Technical support", "Sales"]
case("flip", "Identify the team that should handle this request.", TEAM_STATE, TEAM_OPTS, 0)
case("flip", "Identify a team that should NOT handle this request.", TEAM_STATE, TEAM_OPTS, 1)
case("flip", "Identify the team least relevant to a refund complaint.", TEAM_STATE, TEAM_OPTS, 2)

MOOD_STATE = "Thanks so much, the fix worked first time and the agent was lovely."
MOOD_OPTS = ["Angry", "Grateful", "Confused"]
case("flip", "Identify the customer's tone.", MOOD_STATE, MOOD_OPTS, 1)
case("flip", "Identify the tone that is the opposite of the customer's.", MOOD_STATE, MOOD_OPTS, 0)

# ---------------------------------------------------------------- negation: the state, not the instruction, carries the twist
INSTR_ACT = "Select the action that satisfies the customer's stated request without making unwanted changes."
case("negation", INSTR_ACT, "Do not cancel my subscription. I only want the annual plan switched to monthly.",
     ["Cancel the subscription.", "Switch the plan to monthly and keep the subscription.", "Keep the annual plan unchanged."], 1)
case("negation", INSTR_ACT, "I never said I wanted a refund; I want the duplicate charge reversed, not the original one.",
     ["Refund both charges.", "Reverse the duplicate charge only.", "Refund the original charge only."], 1)
case("negation", INSTR_ACT, "Unless the delivery arrives by Friday, cancel the order. It is now Saturday and nothing has arrived.",
     ["Cancel the order.", "Keep the order open.", "Reschedule delivery for Monday."], 0)
case("negation", INSTR_ACT, "Unless the delivery arrives by Friday, cancel the order. It arrived Thursday.",
     ["Cancel the order.", "Keep the order as delivered.", "Reschedule delivery for Monday."], 1)
case("negation", INSTR_ACT, "Please do not withdraw my cancellation; I still want the account closed.",
     ["Withdraw the cancellation and keep the account open.", "Proceed with closing the account.", "Pause the account for a month."], 1)
case("negation", INSTR_ACT, "I said no to the upgrade twice. Stop offering it and just fix the login problem.",
     ["Apply the upgrade.", "Fix the login problem without upgrading.", "Offer the upgrade with a discount."], 1)
case("negation", INSTR_ACT, "It is not that I want to leave; the price increase is what I object to.",
     ["Cancel the subscription.", "Review the price increase with the customer.", "Sell a higher tier."], 1)
case("negation", "Identify the customer's tone.", "I am not angry, just disappointed that this took three weeks.",
     ["Angry", "Disappointed", "Pleased"], 1)

# ---------------------------------------------------------------- two thresholds: two conditions interact
EXP = (
    "Expense policy: an expense is approved automatically when the amount is 75 or less. "
    "Between 76 and 500 a manager approves it. Above 500 a director approves it. Regardless of "
    "amount, any expense in the category Alcohol or Gifts needs a director."
)
EXP_OPTS = ["Approve automatically.", "Request manager approval.", "Request director approval."]
INSTR_EXP = "Apply the expense policy in the state to the expense described and select the required action."
for amount, category, label in [
    (40, "Meals", 0), (75, "Meals", 0), (76, "Meals", 1), (500, "Travel", 1), (501, "Travel", 2),
    (40, "Gifts", 2), (75, "Alcohol", 2), (300, "Gifts", 2), (12, "Office supplies", 0), (499, "Software", 1),
]:
    case("threshold", INSTR_EXP, f"{EXP}\n\nExpense: amount {amount}, category {category}.", EXP_OPTS, label)

REF = (
    "Refund policy: a refund is granted when the request comes within 30 days of purchase AND the "
    "item is unused. Used items within 30 days get store credit. Anything after 30 days is declined, "
    "unless the item was faulty on arrival, which is refunded at any time."
)
REF_OPTS = ["Grant a refund.", "Issue store credit.", "Decline the request."]
INSTR_REF = "Apply the refund policy in the state to the request described and select the outcome."
for days, used, faulty, label in [
    (10, False, False, 0), (10, True, False, 1), (45, False, False, 2), (45, True, False, 2),
    (45, True, True, 0), (30, False, False, 0), (31, False, False, 2), (3, True, True, 0),
]:
    desc = f"Request: {days} days after purchase, item {'used' if used else 'unused'}, {'faulty on arrival' if faulty else 'not faulty'}."
    case("threshold", INSTR_REF, f"{REF}\n\n{desc}", REF_OPTS, label)

# ---------------------------------------------------------------- distractors: the wrong option's keyword is loud in the state
case("distractor", "Identify the team that should handle this request.",
     "Your sales rep promised me a discount in the sales call, but my real problem is that the app crashes every time I open the invoices tab.",
     ["Billing and refunds", "Technical support", "Sales"], 1)
case("distractor", "Identify the team that should handle this request.",
     "Technical question, sort of: which of your plans includes the API? I am not a customer yet and want to buy.",
     ["Billing and refunds", "Technical support", "Sales"], 2)
case("distractor", INSTR_ACT,
     "Cancel, cancel, cancel: that is what your form kept saying. I do NOT want to cancel. I want the failed payment retried.",
     ["Cancel the subscription.", "Retry the failed payment and keep the subscription.", "Issue a refund."], 1)
case("distractor", "Identify the customer's tone.",
     "FURIOUS is what I would be if this had happened again, but you fixed it in an hour, so honestly, thank you.",
     ["Angry", "Grateful", "Confused"], 1)
case("distractor", INSTR_EXP,
     f"{EXP}\n\nExpense: amount 60, category Meals. Note from submitter: this was a team dinner, no alcohol was ordered although the venue is a wine bar.",
     EXP_OPTS, 0)

# ---------------------------------------------------------------- long states: the decisive fact is buried in a long ticket history
rng = random.Random(20260920)
FILLER = [
    "Agent: Thanks for waiting, I am checking the account now.",
    "Customer: Sure, take your time.",
    "Agent: I can see the account was created in 2023 on the annual plan.",
    "Customer: Yes, that sounds right.",
    "Agent: There are two devices registered, a laptop and a phone.",
    "Customer: The phone is my old one, you can ignore it.",
    "Agent: Noted. Your last invoice was paid on the first of the month.",
    "Customer: I remember, the receipt came through fine.",
    "Agent: Do you use the mobile app or mostly the website?",
    "Customer: Mostly the website, the app only for notifications.",
    "Agent: Understood. I also see a support ticket from last spring about a password reset.",
    "Customer: That got sorted, no problems since.",
    "Agent: Good. Let me pull the billing history for the last twelve months.",
    "Customer: Okay.",
    "Agent: Everything looks consistent, one charge per year, same card.",
    "Customer: Right.",
]


def long_state(decisive: str, position: str, turns: int) -> str:
    lines = [rng.choice(FILLER) for _ in range(turns)]
    if position == "start":
        lines.insert(0, decisive)
    elif position == "middle":
        lines.insert(len(lines) // 2, decisive)
    else:
        lines.append(decisive)
    return "Ticket transcript:\n" + "\n".join(lines)


LONG_OPTS = ["Cancel the subscription.", "Keep the subscription active and change the plan to monthly.", "Issue a refund for the last invoice."]
for position in ("start", "middle", "end"):
    for turns in (40, 120):
        case("long", INSTR_ACT,
             long_state("Customer: To be clear, I do not want to cancel anything. I want to move from annual to monthly billing.", position, turns),
             LONG_OPTS, 1)
        case("long", INSTR_ACT,
             long_state("Customer: I have decided. Please cancel the subscription today; I do not want another plan.", position, turns),
             LONG_OPTS, 0)

# ---------------------------------------------------------------- write
name = f"policy-hard-{len(cases)}"
path = OUT / f"{name}.jsonl"
path.write_text("".join(json.dumps(c, ensure_ascii=False) + "\n" for c in cases), encoding="utf-8")
digest = hashlib.sha256(path.read_bytes()).hexdigest()
groups: dict[str, int] = {}
for c in cases:
    groups[c["group"]] = groups.get(c["group"], 0) + 1
manifest = {
    "name": name,
    "file": path.name,
    "sha256": digest,
    "cases": len(cases),
    "groups": groups,
    "question_type": "choice",
    "source": "authored for jevify on 2026-09-20 by tools/author_policy_hard.py (deterministic, seed 20260920)",
    "construction": (
        "five groups written before any model saw them: instruction flips (the instruction inverts "
        "the natural reading of a fixed state), negations in the state, two-threshold policies with "
        "interacting conditions, distractors (the wrong option's keyword is loud in the state), and "
        "long ticket transcripts with the decisive line at the start, middle or end of 40 or 120 "
        "filler turns. Append-only after the first scored run; the hash freezes the file."
    ),
    "expected_answers_by": "human (the author of the cases)",
    "measures": "agreement with the intended action per group; not calibration",
    "known_results": "none at authoring time",
}
(OUT / f"{name}.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
print(name, groups, digest[:16])
