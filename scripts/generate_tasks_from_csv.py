from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


ROOT = Path(__file__).resolve().parents[1]
TASKS_DIR = ROOT / "tasks"
CSV_PATH = TASKS_DIR / "customer_support_tickets_200k.csv"


CATEGORY_TO_DEPARTMENT = {
    "Payment Problem": "billing",
    "Refund Request": "billing",
    "Subscription Cancellation": "billing",
    "Feature Request": "general",
    "Account Suspension": "technical",
    "Login Issue": "technical",
    "Performance Issue": "technical",
    "Bug Report": "technical",
    "Data Sync Issue": "technical",
    "Security Concern": "technical",
}


@dataclass(frozen=True)
class Candidate:
    source_ticket_id: int
    category: str
    product: str
    region: str
    payment_method: str
    operating_system: str
    browser: str
    subscription_type: str
    customer_segment: str
    status: str
    escalated: bool
    sla_breached: bool
    department: str
    priority: str
    action_type: str
    customer_tier: str
    urgency: int
    time_waiting: int
    description: str
    ground_truth_reason: str


@dataclass(frozen=True)
class SlotSpec:
    id: str
    department: str
    priority: str
    action_type: str
    categories: tuple[str, ...]
    hint: Optional[str] = None
    urgency: Optional[int] = None
    time_waiting: Optional[int] = None


def normalize_tier(subscription_type: str) -> str:
    return "premium" if subscription_type in {"Premium", "Enterprise"} else "free"


def derive_priority(row: dict[str, str], department: str) -> str:
    raw = row["priority"]

    if department == "general":
        return "low" if raw == "Low" else "medium"

    if raw in {"Urgent", "High"}:
        return "high"
    if raw == "Medium":
        return "medium"
    if department == "billing" and row["sla_breached"] == "Yes":
        return "medium"
    if department == "technical" and int(row["issue_complexity_score"]) >= 6:
        return "medium"
    return "low"


def derive_action(row: dict[str, str], department: str, priority: str) -> str:
    category = row["category"]
    status = row["status"]
    escalated = row["escalated"] == "Yes"
    sla_breached = row["sla_breached"] == "Yes"

    if department == "general":
        return "resolve"

    if department == "billing":
        if status == "Pending Customer":
            return "request_info"
        if category == "Payment Problem" and (priority == "high" or escalated or sla_breached):
            return "escalate"
        if category == "Subscription Cancellation" and priority == "high" and escalated:
            return "escalate"
        return "resolve"

    if category in {"Performance Issue", "Data Sync Issue"} and status == "Pending Customer":
        return "request_info"
    if priority == "high" or escalated or category in {"Account Suspension", "Login Issue", "Security Concern", "Bug Report"}:
        return "escalate"
    if status == "Pending Customer":
        return "request_info"
    return "request_info" if category in {"Performance Issue", "Data Sync Issue"} else "escalate"


def derive_urgency(row: dict[str, str], priority: str, department: str) -> int:
    raw_priority = row["priority"]
    premium = normalize_tier(row["subscription_type"]) == "premium"
    escalated = row["escalated"] == "Yes"
    sla_breached = row["sla_breached"] == "Yes"

    if priority == "high":
        if raw_priority == "Urgent" or (premium and (escalated or sla_breached)):
            return 5
        return 4
    if priority == "medium":
        if department == "general":
            return 2
        return 3
    return 1 if department == "general" else 2


def derive_time_waiting(row: dict[str, str]) -> int:
    hours = float(row["first_response_time_hours"])
    waiting = round(hours * 1.5)
    return max(5, min(120, waiting))


def build_description(row: dict[str, str], department: str, action_type: str) -> str:
    category = row["category"]
    product = row["product"]
    payment_method = row["payment_method"]
    browser = row["browser"]
    operating_system = row["operating_system"]
    region = row["region"]
    segment = row["customer_segment"].lower()
    tier = row["subscription_type"].lower()

    if category == "Feature Request":
        return (
            f"Does {product} support scheduled summary exports for {segment} teams, "
            "or is that only available on another plan?"
        )

    if category == "Refund Request":
        return f"I need a refund for the recent {product} charge on my {payment_method} statement."

    if category == "Subscription Cancellation":
        if action_type == "escalate":
            return (
                f"Our {tier} subscription for {product} was cancelled unexpectedly and the whole team lost access."
            )
        return f"My {tier} subscription for {product} was cancelled unexpectedly and I need clarification."

    if category == "Payment Problem":
        if action_type == "request_info":
            return (
                f"My payment for {product} looks wrong, but I am not sure which invoice or workspace is affected yet. "
                f"It was billed through {payment_method}."
            )
        if action_type == "escalate":
            return (
                f"The payment for {product} through {payment_method} was deducted, "
                "but the transaction still shows failed and access is blocked."
            )
        return (
            f"The payment for {product} through {payment_method} went through, "
            "but the workspace still shows unpaid."
        )

    if category == "Account Suspension":
        return (
            f"Users can no longer access {product}; the account appears suspended after sign-in "
            f"on {browser} for {operating_system}."
        )

    if category == "Login Issue":
        return (
            f"We cannot log in to {product} with valid credentials on {browser} for {operating_system}."
        )

    if category == "Security Concern":
        return f"Two-factor authentication codes are not arriving, so the team cannot complete sign-in to {product}."

    if category == "Bug Report":
        return f"After the latest update, {product} crashes while generating reports on {operating_system}."

    if category == "Performance Issue":
        if action_type == "request_info":
            return (
                f"{product} feels very slow in {region}, but I do not know yet if it is specific to "
                f"{browser} on {operating_system}."
            )
        return (
            f"{product} is extremely slow in {region} and the dashboard times out for multiple users on {browser}."
        )

    if category == "Data Sync Issue":
        if action_type == "request_info":
            return (
                f"{product} is not syncing data across devices, but I do not know yet whether the issue is limited "
                f"to {browser} on {operating_system}."
            )
        return f"{product} is not syncing data across devices and recent updates are missing for the whole team."

    raise ValueError(f"Unsupported category: {category}")


def build_reason(row: dict[str, str], department: str, priority: str, action_type: str) -> str:
    return (
        f"Generated from CSV ticket {row['ticket_id']} ({row['category']} on {row['product']}). "
        f"Route to {department}; priority is {priority} from the source severity and action is {action_type} "
        f"from the source status and escalation signals."
    )


def load_candidates() -> list[Candidate]:
    candidates: list[Candidate] = []
    with CSV_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            department = CATEGORY_TO_DEPARTMENT[row["category"]]
            priority = derive_priority(row, department)
            action_type = derive_action(row, department, priority)
            candidates.append(
                Candidate(
                    source_ticket_id=int(row["ticket_id"]),
                    category=row["category"],
                    product=row["product"],
                    region=row["region"],
                    payment_method=row["payment_method"],
                    operating_system=row["operating_system"],
                    browser=row["browser"],
                    subscription_type=row["subscription_type"],
                    customer_segment=row["customer_segment"],
                    status=row["status"],
                    escalated=row["escalated"] == "Yes",
                    sla_breached=row["sla_breached"] == "Yes",
                    department=department,
                    priority=priority,
                    action_type=action_type,
                    customer_tier=normalize_tier(row["subscription_type"]),
                    urgency=derive_urgency(row, priority, department),
                    time_waiting=derive_time_waiting(row),
                    description=build_description(row, department, action_type),
                    ground_truth_reason=build_reason(row, department, priority, action_type),
                )
            )
    return sorted(candidates, key=lambda candidate: candidate.source_ticket_id)


def choose_candidate(candidates: Iterable[Candidate], slot: SlotSpec, used_ids: set[int]) -> Candidate:
    for candidate in candidates:
        if candidate.source_ticket_id in used_ids:
            continue
        if candidate.department != slot.department:
            continue
        if candidate.priority != slot.priority:
            continue
        if candidate.action_type != slot.action_type:
            continue
        if candidate.category not in slot.categories:
            continue
        used_ids.add(candidate.source_ticket_id)
        return candidate
    raise RuntimeError(f"Could not find a CSV candidate for slot {slot.id}.")


def ticket_from_candidate(candidate: Candidate, slot: SlotSpec) -> dict[str, object]:
    return {
        "id": slot.id,
        "category_hint": slot.hint or slot.department,
        "description": candidate.description,
        "urgency": slot.urgency if slot.urgency is not None else candidate.urgency,
        "customer_tier": candidate.customer_tier,
        "time_waiting": slot.time_waiting if slot.time_waiting is not None else candidate.time_waiting,
        "ground_truth": {
            "department": slot.department,
            "priority": slot.priority,
            "action_type": slot.action_type,
        },
        "ground_truth_reason": candidate.ground_truth_reason,
    }


def slot(
    id: str,
    department: str,
    priority: str,
    action_type: str,
    *categories: str,
    hint: Optional[str] = None,
    urgency: Optional[int] = None,
    time_waiting: Optional[int] = None,
) -> SlotSpec:
    return SlotSpec(
        id=id,
        department=department,
        priority=priority,
        action_type=action_type,
        categories=tuple(categories),
        hint=hint,
        urgency=urgency,
        time_waiting=time_waiting,
    )


SCENARIOS = {
    "easy": {
        "scenario_id": "support-triage-easy-v1",
        "difficulty": "easy",
        "description": "Dataset-backed triage with mostly accurate hints, no arrivals, and enough clarity to test basic routing.",
        "start_time": 0,
        "max_steps": 8,
        "sla_targets_steps": {"high": 2, "medium": 5, "low": 999},
        "initial": [
            slot("E001", "billing", "medium", "resolve", "Refund Request", "Subscription Cancellation"),
            slot("E002", "technical", "high", "escalate", "Login Issue", "Security Concern", "Account Suspension", hint="technical", urgency=4),
            slot("E003", "general", "low", "resolve", "Feature Request", hint="general", urgency=1),
            slot("E004", "billing", "low", "resolve", "Refund Request", "Subscription Cancellation", hint="billing"),
            slot("E005", "billing", "medium", "request_info", "Payment Problem", hint="billing"),
            slot("E006", "technical", "high", "escalate", "Bug Report", "Login Issue", "Security Concern", "Account Suspension", hint="technical", urgency=5, time_waiting=85),
        ],
        "arrivals": [],
    },
    "medium": {
        "scenario_id": "support-triage-medium-v1",
        "difficulty": "medium",
        "description": "Dataset-backed triage with noisier hints, staged arrivals, and enough urgent work to force prioritization.",
        "start_time": 0,
        "max_steps": 12,
        "sla_targets_steps": {"high": 2, "medium": 5, "low": 999},
        "initial": [
            slot("M001", "billing", "medium", "resolve", "Refund Request", "Subscription Cancellation", hint="general"),
            slot("M002", "technical", "high", "escalate", "Login Issue", "Security Concern", "Account Suspension", hint="billing"),
            slot("M003", "general", "low", "resolve", "Feature Request", hint="technical"),
            slot("M004", "technical", "high", "escalate", "Bug Report", "Account Suspension", "Security Concern", hint="general"),
            slot("M005", "billing", "medium", "request_info", "Payment Problem", hint="billing"),
            slot("M006", "billing", "medium", "resolve", "Refund Request", "Subscription Cancellation", "Payment Problem", hint="technical"),
            slot("M007", "general", "low", "resolve", "Feature Request", hint="general"),
            slot("M008", "technical", "medium", "request_info", "Performance Issue", "Data Sync Issue", hint="billing"),
        ],
        "arrivals": [
            {
                "time": 2,
                "tickets": [
                    slot("M009", "technical", "high", "escalate", "Bug Report", "Login Issue", "Security Concern", "Account Suspension", hint="general", urgency=5),
                    slot("M010", "billing", "high", "escalate", "Payment Problem", "Subscription Cancellation", hint="technical"),
                ],
            },
            {
                "time": 4,
                "tickets": [
                    slot("M011", "general", "medium", "resolve", "Feature Request", hint="billing"),
                    slot("M012", "technical", "medium", "request_info", "Performance Issue", "Data Sync Issue", hint="general"),
                ],
            },
        ],
    },
    "hard": {
        "scenario_id": "support-triage-hard-v1",
        "difficulty": "hard",
        "description": "Dataset-backed triage with high hint noise, continuous arrivals, and fewer steps than tickets so the agent must trade off impact.",
        "start_time": 0,
        "max_steps": 16,
        "sla_targets_steps": {"high": 2, "medium": 5, "low": 999},
        "initial": [
            slot("H001", "technical", "high", "escalate", "Bug Report", "Login Issue", "Security Concern", "Account Suspension", hint="billing", urgency=5),
            slot("H002", "billing", "high", "resolve", "Refund Request", "Subscription Cancellation", hint="general"),
            slot("H003", "general", "low", "resolve", "Feature Request", hint="technical"),
            slot("H004", "technical", "high", "escalate", "Login Issue", "Security Concern", "Account Suspension", "Bug Report", hint="general", urgency=5),
            slot("H005", "billing", "medium", "request_info", "Payment Problem", hint="billing"),
            slot("H006", "technical", "medium", "escalate", "Bug Report", "Login Issue", "Account Suspension", hint="technical"),
            slot("H007", "general", "low", "resolve", "Feature Request", hint="general"),
            slot("H008", "billing", "medium", "resolve", "Refund Request", "Subscription Cancellation", hint="billing"),
            slot("H009", "billing", "high", "escalate", "Payment Problem", hint="technical"),
            slot("H010", "technical", "medium", "request_info", "Performance Issue", "Data Sync Issue", hint="general"),
        ],
        "arrivals": [
            {
                "time": 1,
                "tickets": [
                    slot("H011", "technical", "high", "escalate", "Bug Report", "Security Concern", "Login Issue", hint="general", urgency=5),
                    slot("H012", "general", "low", "resolve", "Feature Request", hint="billing"),
                ],
            },
            {
                "time": 2,
                "tickets": [
                    slot("H013", "billing", "high", "resolve", "Refund Request", "Subscription Cancellation", hint="technical"),
                    slot("H014", "technical", "high", "escalate", "Account Suspension", "Login Issue", "Security Concern", "Bug Report", hint="general"),
                ],
            },
            {
                "time": 3,
                "tickets": [
                    slot("H015", "technical", "high", "escalate", "Bug Report", "Login Issue", "Account Suspension", "Security Concern", hint="billing", urgency=5),
                    slot("H016", "billing", "low", "resolve", "Refund Request", "Subscription Cancellation", hint="general"),
                ],
            },
            {
                "time": 4,
                "tickets": [
                    slot("H017", "technical", "medium", "request_info", "Performance Issue", "Data Sync Issue", hint="billing"),
                    slot("H018", "billing", "medium", "request_info", "Payment Problem", hint="general"),
                ],
            },
            {
                "time": 5,
                "tickets": [
                    slot("H019", "technical", "high", "escalate", "Bug Report", "Login Issue", "Security Concern", hint="general"),
                    slot("H020", "billing", "high", "escalate", "Payment Problem", hint="technical"),
                ],
            },
            {
                "time": 6,
                "tickets": [
                    slot("H021", "general", "medium", "resolve", "Feature Request", hint="technical"),
                    slot("H022", "technical", "medium", "escalate", "Bug Report", "Login Issue", "Account Suspension", hint="billing"),
                ],
            },
            {
                "time": 7,
                "tickets": [
                    slot("H023", "billing", "medium", "resolve", "Refund Request", "Subscription Cancellation", hint="general"),
                    slot("H024", "technical", "high", "escalate", "Bug Report", "Login Issue", "Security Concern", "Account Suspension", hint="billing"),
                ],
            },
        ],
    },
}


def build_scenario(name: str, candidates: list[Candidate], used_ids: set[int]) -> dict[str, object]:
    config = SCENARIOS[name]

    initial_tickets = [
        ticket_from_candidate(choose_candidate(candidates, slot_spec, used_ids), slot_spec)
        for slot_spec in config["initial"]
    ]

    arrival_schedule = []
    for wave in config["arrivals"]:
        arrival_schedule.append(
            {
                "time": wave["time"],
                "tickets": [
                    ticket_from_candidate(choose_candidate(candidates, slot_spec, used_ids), slot_spec)
                    for slot_spec in wave["tickets"]
                ],
            }
        )

    return {
        "scenario_id": config["scenario_id"],
        "difficulty": config["difficulty"],
        "description": config["description"],
        "start_time": config["start_time"],
        "max_steps": config["max_steps"],
        "sla_targets_steps": config["sla_targets_steps"],
        "initial_tickets": initial_tickets,
        "arrival_schedule": arrival_schedule,
    }


def main() -> None:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV dataset not found: {CSV_PATH}")

    candidates = load_candidates()
    used_ids: set[int] = set()

    for scenario_name in ("easy", "medium", "hard"):
        scenario = build_scenario(scenario_name, candidates, used_ids)
        output_path = TASKS_DIR / f"{scenario_name}.json"
        output_path.write_text(json.dumps(scenario, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {output_path.name}")


if __name__ == "__main__":
    main()
