"""
Judge simulation script — tests your deployed HF Space exactly like the hackathon judges do.
No API key needed — uses a simple rule-based agent built into this script.

Usage:
    python test_deployed.py
    python test_deployed.py --url https://tensura81-openenv.hf.space
    python test_deployed.py --scenario easy
"""

import argparse
import sys
import requests

SPACE_URL = "https://tensura81-openenv.hf.space"
SCENARIOS = ["easy", "medium", "hard"]


def decide_action(ticket: dict) -> dict:
    """Simple rule-based agent — mirrors the heuristic baseline logic."""
    desc = ticket.get("description", "").lower()
    hint = ticket.get("category_hint", "").lower()
    urgency = ticket.get("urgency", 1)

    general_words = ["how do i", "does ", "is there a way", "do you offer",
                     "available on another plan", "support scheduled", "discounts for"]
    billing_words = ["refund", "charge", "invoice", "payment", "billed",
                     "subscription", "renewal", "cancelled"]
    technical_words = ["login", "otp", "suspended", "access", "crash",
                       "error", "timeout", "slow", "bug", "sync", "2fa"]

    if any(w in desc for w in general_words):
        department = "general"
    elif any(w in desc for w in billing_words):
        department = "billing"
    elif any(w in desc for w in technical_words):
        department = "technical"
    else:
        department = hint if hint in ["billing", "technical", "general"] else "general"

    # Priority
    if urgency >= 4:
        priority = "high"
    elif urgency >= 2:
        priority = "medium"
    else:
        priority = "low"

    if department == "general":
        priority = "low"

    # Action
    vague_words = ["not sure which", "don't know", "i think", "maybe", "not sure if"]
    if any(w in desc for w in vague_words):
        action_type = "request_info"
    elif department == "technical" and any(w in desc for w in ["login", "otp", "suspended", "access", "error"]):
        action_type = "escalate"
    else:
        action_type = "resolve"

    return {"department": department, "priority": priority, "action_type": action_type}


def run_episode(base_url: str, scenario: str) -> dict:
    print(f"\n{'='*60}")
    print(f"  SCENARIO: {scenario.upper()}")
    print(f"{'='*60}")

    # 1. Reset — same as judge does first
    r = requests.post(f"{base_url}/reset", json={"scenario_name": scenario}, timeout=30)
    r.raise_for_status()
    reset_data = r.json()
    session_id = reset_data["session_id"]
    print(f"  Session ID : {session_id}")
    print(f"  Pending    : {reset_data['observation']['pending_count']} tickets")

    done = False
    step = 0

    while not done:
        step += 1

        # 2. Read state — get current_ticket
        state_r = requests.get(f"{base_url}/state/{session_id}", timeout=30)
        state_r.raise_for_status()
        state = state_r.json()
        current_ticket = state["current_ticket"]

        # 3. Decide action
        action = decide_action(current_ticket)
        action["ticket_id"] = current_ticket["id"]  # ALWAYS use current_ticket id

        # 4. Submit step
        step_r = requests.post(f"{base_url}/step/{session_id}", json=action, timeout=30)
        step_r.raise_for_status()
        step_data = step_r.json()

        reward = step_data["reward"]["step_score"]
        done = step_data["done"]
        error = step_data.get("info", {}).get("error")

        status = "✅" if not error else f"❌ {error}"
        print(f"  Step {step:02d} | {current_ticket['id']:6s} | reward={reward:+.2f} | {status}")

    # 5. Grade — what judges collect as your final score
    grade_r = requests.get(f"{base_url}/grade/{session_id}", timeout=30)
    grade_r.raise_for_status()
    grade = grade_r.json()["grade"]

    print(f"\n  ── Final Grade ──────────────────────────────")
    print(f"  Routing accuracy : {grade['routing_accuracy']:.2%}")
    print(f"  Priority accuracy: {grade['priority_accuracy']:.2%}")
    print(f"  SLA score        : {grade['sla_score']:.2%}")
    print(f"  Action accuracy  : {grade['action_accuracy']:.2%}")
    print(f"  ─────────────────────────────────────────────")
    print(f"  FINAL SCORE      : {grade['final_score']:.4f}")
    return grade


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default=SPACE_URL)
    parser.add_argument("--scenario", choices=SCENARIOS)
    args = parser.parse_args()
    base_url = args.url.rstrip("/")

    # Health check
    print(f"\nTesting: {base_url}")
    try:
        health = requests.get(f"{base_url}/health", timeout=15)
        health.raise_for_status()
        print(f"Health  : {health.json()['status']} ✅")
    except Exception as e:
        print(f"❌ Space unreachable: {e}")
        sys.exit(1)

    scenarios = [args.scenario] if args.scenario else SCENARIOS
    grades = {}

    for scenario in scenarios:
        try:
            grade = run_episode(base_url, scenario)
            grades[scenario] = grade["final_score"]
        except Exception as e:
            print(f"\n❌ Error on {scenario}: {e}")
            grades[scenario] = 0.0

    # Summary
    print(f"\n{'='*60}")
    print(f"  OVERALL RESULTS")
    print(f"{'='*60}")
    for scenario, score in grades.items():
        bar = "█" * int(score * 20)
        print(f"  {scenario:8s} : {score:.4f}  {bar}")
    if len(grades) == 3:
        avg = sum(grades.values()) / 3
        print(f"  {'AVERAGE':8s} : {avg:.4f}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
