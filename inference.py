import argparse
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from agent.baseline import choose_action as heuristic_choose_action
from agent.baseline import choose_action_from_observation as heuristic_choose_action_from_observation
from env.core import TicketEnv
from env.models import Action, Observation
from env.tasks import scenario_names

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover - handled at runtime for local setup failures
    OpenAI = None


<<<<<<< Updated upstream
DEFAULT_BASE_URL = "https://api.openai.com/v1"
=======
DEFAULT_API_BASE_URL = "https://router.huggingface.co/v1"
DEFAULT_MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
BENCHMARK = os.getenv("OPENENV_BENCHMARK") or "support-triage-env"
SUCCESS_SCORE_THRESHOLD = 0.1

>>>>>>> Stashed changes
SYSTEM_PROMPT = """You are a support triage agent. Choose ONE pending ticket to handle next.
Return a JSON object with exactly these keys: ticket_id, department, priority, action_type.

## TICKET SELECTION — THIS IS CRITICAL
- HIGH-PRIORITY tickets (urgency >= 4) have a strict SLA: they MUST be handled within 3 steps of appearing.
- Always handle ALL urgency 5 tickets first, then urgency 4, then the rest.
- Among equal urgency, prefer tickets with higher time_waiting.
- NEVER skip an urgency >= 4 ticket to handle a lower one — this incurs a -0.5 penalty.

## DEPARTMENT RULES
**billing**: The issue involves money — refunds, charges, invoices, payments, subscriptions, renewals, billing statements, pricing disputes, plan downgrades with billing impact.
**technical**: The issue involves system failures — login/access failures, crashes, bugs, HTTP errors (500/401/504), slow performance, timeouts, sync failures, authentication (2FA/OTP), suspended accounts, webhooks, API issues.
**general**: The issue is a question or inquiry — "how do I", "does X support", "is there a way to", "do you offer", feature questions, plan comparisons.

Key rules:
- Route from the SYMPTOM, not the product name. "Billing System crashes" → technical.
- "cancelled unexpectedly and I need clarification" → billing (subscription issue).
- Feature/plan/how-to questions → general, UNLESS they mention refund/payment → billing.
- Use specialist_team as a tiebreaker when text is ambiguous: payments_ops/refunds/subscription_ops → billing; account_access/security/product_bug/platform_reliability → technical; sales_ops → general.

## PRIORITY RULES
- urgency >= 4 → "high"
- urgency 2-3 → "medium"
- urgency 1 → "low"
- EXCEPTION: general inquiry questions → always "low"
- EXCEPTION: simple billing (refund, invoice inquiry) with urgency <= 2 → "low"

## ACTION TYPE RULES
**request_info**: Ticket is VAGUE — missing key details. Look for: "not sure which", "don't know", "I think", "I guess", "maybe", very short descriptions (4 words or fewer), hedging language. Example: "payment looks wrong, not sure which invoice or workspace".
**escalate**: Access is blocked, login/2FA/OTP failures, account suspension, outages, HTTP errors, crashes, data sync issues. For billing: only when payment is actively blocking access ("whole team lost access", "workspace shows unpaid", "payment failed and can't access").
**resolve**: Everything else — general questions, clear billing issues (refunds, invoices), straightforward technical fixes.

Return JSON only. No markdown fences, no explanation."""


def build_client() -> Tuple[Any, str]:
    api_key = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY") or os.getenv("API_KEY")
    api_base_url = os.getenv("API_BASE_URL") or DEFAULT_API_BASE_URL
    model_name = os.getenv("MODEL_NAME") or DEFAULT_MODEL_NAME

    if OpenAI is None:
<<<<<<< Updated upstream
        raise RuntimeError("The openai package is not installed. Run `pip install -r requirements.txt` first.")

    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("HF_TOKEN")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY or HF_TOKEN before running inference.py.")

    model_name = os.getenv("MODEL_NAME")
    if not model_name:
        raise RuntimeError("Set MODEL_NAME before running inference.py.")

    base_url = os.getenv("API_BASE_URL", DEFAULT_BASE_URL)
    return OpenAI(api_key=api_key, base_url=base_url), model_name
=======
        raise RuntimeError("The openai package is not installed.")
    if not api_key:
        raise RuntimeError("HF_TOKEN or OPENAI_API_KEY is not set.")
    return OpenAI(api_key=api_key, base_url=api_base_url), model_name
>>>>>>> Stashed changes


def build_observation_payload(observation: Observation) -> Dict[str, Any]:
    return {
        "current_time": observation.current_time,
        "step_number": observation.step_number,
        "pending_count": observation.pending_count,
        "resolved_count": observation.resolved_count,
        "current_ticket": observation.current_ticket.model_dump() if observation.current_ticket else None,
        "pending_tickets": [ticket.model_dump() for ticket in observation.pending_tickets],
    }


def choose_action_with_llm(client: Any, model_name: str, observation: Observation) -> Tuple[Action, bool, str]:
    if observation.current_ticket is None:
        raise RuntimeError("Cannot choose an action without a current ticket.")

    user_payload = build_observation_payload(observation)
    response = client.chat.completions.create(
        model=model_name,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": json.dumps(user_payload, indent=2)},
        ],
    )
    content = response.choices[0].message.content or ""

    try:
        action = Action.model_validate(json.loads(extract_json_object(content)))
        pending_ids = {ticket.id for ticket in observation.pending_tickets}
        if action.ticket_id not in pending_ids:
            raise ValueError("Model selected a ticket that is not pending.")
        return action, False, content
    except Exception:
        # Keep the baseline reproducible and non-brittle even if the model emits malformed JSON.
        fallback = heuristic_choose_action_from_observation(observation)
        return fallback, True, content


def extract_json_object(text: str) -> str:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        cleaned = cleaned.replace("json", "", 1).strip()

    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError("No JSON object found in model response.")
    return cleaned[start : end + 1]


def run_episode(
    scenario_name: str = "easy",
    heuristic_only: bool = False,
    max_steps: Optional[int] = None,
    client: Any = None,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    config: Dict[str, Any] = {"scenario_name": scenario_name}
    if max_steps is not None:
        config["max_steps"] = max_steps

    env = TicketEnv(config)
    state = env.reset()
    done = False
    llm_calls = 0
    fallback_actions = 0
    trace: List[Dict[str, Any]] = []
    first_seen_at: Dict[str, int] = {}

    while not done:
        for pending_ticket in state.pending_tickets:
            first_seen_at.setdefault(pending_ticket.id, state.current_time)

        ticket = state.current_ticket
        if ticket is None:
            break

        if heuristic_only:
            action = heuristic_choose_action_from_observation(state, first_seen_at)
            raw_response = None
            used_fallback = False
        else:
            assert model_name is not None
            action, used_fallback, raw_response = choose_action_with_llm(client, model_name, state)
            llm_calls += 1
            fallback_actions += int(used_fallback)

        state, reward, done, info = env.step(action)
        trace.append(
            {
                "ticket_id": action.ticket_id,
                "reward": reward.step_score,
                "total_score": reward.total_score,
                "done": done,
                "used_fallback": used_fallback,
                "raw_response": raw_response,
            }
        )

    grade = env.grade()
    return {
        "scenario": scenario_name,
        "model_name": model_name if not heuristic_only else "heuristic-baseline",
        "steps_taken": len(trace),
        "llm_calls": llm_calls,
        "fallback_actions": fallback_actions,
        "grade": grade.model_dump(),
        "trace": trace,
    }


def run_all_tasks(
    heuristic_only: bool = False,
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    client = None
    model_name = None
    if not heuristic_only:
        client, model_name = build_client()

    reports = [
        run_episode(
            scenario_name=name,
            heuristic_only=heuristic_only,
            max_steps=max_steps,
            client=client,
            model_name=model_name,
        )
        for name in scenario_names()
    ]

    average_score = round(
        sum(report["grade"]["final_score"] for report in reports) / len(reports),
        4,
    )
    return {
        "runner": "openai-client" if not heuristic_only else "heuristic",
        "model_name": model_name if not heuristic_only else "heuristic-baseline",
        "average_final_score": average_score,
        "tasks": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the support-triage baseline across one or all tasks.")
    parser.add_argument("--scenario", choices=["all", *scenario_names()], default="all")
    parser.add_argument("--heuristic-only", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    if args.scenario == "all":
        report = run_all_tasks(heuristic_only=args.heuristic_only, max_steps=args.max_steps)
    else:
        client = None
        model_name = None
        if not args.heuristic_only:
            client, model_name = build_client()
        report = run_episode(
            scenario_name=args.scenario,
            heuristic_only=args.heuristic_only,
            max_steps=args.max_steps,
            client=client,
            model_name=model_name,
        )

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
