import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

from agent.baseline import choose_action_from_observation as heuristic_choose_action_from_observation
from env.core import TicketEnv
from env.models import Action, Observation
from env.tasks import scenario_names

try:
    from openai import OpenAI  # type: ignore
except ImportError:  # pragma: no cover
    OpenAI = None


HF_TOKEN = os.getenv("HF_TOKEN")
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "meta-llama/Llama-3.3-70B-Instruct"
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")
BENCHMARK = os.getenv("OPENENV_BENCHMARK") or "support-triage-env"
SUCCESS_SCORE_THRESHOLD = 0.1

SYSTEM_PROMPT = """You are a support triage agent. Choose ONE pending ticket to handle next.
Return a JSON object with exactly these keys: ticket_id, department, priority, action_type.

Rules:
- Prefer the most urgent pending ticket.
- Use billing for payment, charges, invoices, renewals, refunds.
- Use technical for login issues, crashes, bugs, sync failures, API problems.
- Use general for informational or how-to requests.
- Use high priority for urgency >= 4, medium for urgency 2-3, else low.
- Use request_info for vague issues, escalate for severe blocked access or technical failures, resolve otherwise.

Return JSON only. No markdown fences, no explanation."""


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_value = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def build_client() -> Tuple[Any, str]:
    if OpenAI is None:
        raise RuntimeError("The openai package is not installed.")
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN is not set.")
    return OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL), MODEL_NAME


def build_observation_payload(observation: Observation) -> Dict[str, Any]:
    return {
        "current_time": observation.current_time,
        "step_number": observation.step_number,
        "pending_count": observation.pending_count,
        "resolved_count": observation.resolved_count,
        "current_ticket": observation.current_ticket.model_dump() if observation.current_ticket else None,
        "pending_tickets": [ticket.model_dump() for ticket in observation.pending_tickets],
    }


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


def choose_action_with_llm(client: Any, model_name: str, observation: Observation) -> Tuple[Action, bool]:
    if observation.current_ticket is None:
        raise RuntimeError("Cannot choose an action without a current ticket.")

    payload = build_observation_payload(observation)
    try:
        response = client.chat.completions.create(
            model=model_name,
            temperature=0,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(payload, indent=2)},
            ],
        )
        content = response.choices[0].message.content or ""
        action = Action.model_validate(json.loads(extract_json_object(content)))
        pending_ids = {ticket.id for ticket in observation.pending_tickets}
        if action.ticket_id not in pending_ids:
            raise ValueError("Model selected a ticket that is not pending.")
        return action, False
    except Exception:
        return heuristic_choose_action_from_observation(observation), True


def run_episode(
    scenario_name: str,
    heuristic_only: bool = False,
    max_steps: Optional[int] = None,
    client: Any = None,
    model_name: Optional[str] = None,
) -> Dict[str, Any]:
    config: Dict[str, Any] = {"scenario_name": scenario_name}
    if max_steps is not None:
        config["max_steps"] = max_steps

    env = TicketEnv(config)
    rewards: List[float] = []
    trace: List[Dict[str, Any]] = []
    first_seen_at: Dict[str, int] = {}
    llm_calls = 0
    fallback_actions = 0
    success = False
    final_score = 0.0
    steps_taken = 0
    model_label = model_name if not heuristic_only and model_name else "heuristic-baseline"

    log_start(task=scenario_name, env=BENCHMARK, model=model_label)

    try:
        state = env.reset()
        done = False

        while not done:
            for pending_ticket in state.pending_tickets:
                first_seen_at.setdefault(pending_ticket.id, state.current_time)

            if state.current_ticket is None:
                break

            if heuristic_only or client is None or model_name is None:
                action = heuristic_choose_action_from_observation(state, first_seen_at)
                used_fallback = False
            else:
                action, used_fallback = choose_action_with_llm(client, model_name, state)
                llm_calls += 1
                fallback_actions += int(used_fallback)

            next_state, reward, done, info = env.step(action)
            steps_taken += 1
            rewards.append(reward.step_score)

            action_str = json.dumps(action.model_dump(), separators=(",", ":"))
            last_error = info.get("error") if isinstance(info, dict) else None
            log_step(
                step=steps_taken,
                action=action_str,
                reward=reward.step_score,
                done=done,
                error=last_error,
            )

            trace.append(
                {
                    "ticket_id": action.ticket_id,
                    "reward": reward.step_score,
                    "total_score": reward.total_score,
                    "done": done,
                    "used_fallback": used_fallback,
                    "error": last_error,
                }
            )
            state = next_state

        grade = env.grade().model_dump()
        final_score = max(0.0, min(1.0, float(grade["final_score"])))
        success = final_score >= SUCCESS_SCORE_THRESHOLD
        return {
            "scenario": scenario_name,
            "model_name": model_label,
            "steps_taken": steps_taken,
            "llm_calls": llm_calls,
            "fallback_actions": fallback_actions,
            "grade": grade,
            "trace": trace,
        }
    except Exception as exc:
        print(f"Warning: scenario '{scenario_name}' failed: {exc}", file=sys.stderr)
        return {
            "scenario": scenario_name,
            "model_name": model_label,
            "steps_taken": steps_taken,
            "llm_calls": llm_calls,
            "fallback_actions": fallback_actions,
            "grade": {
                "routing_accuracy": 0.0,
                "priority_accuracy": 0.0,
                "sla_score": 0.0,
                "action_accuracy": 0.0,
                "final_score": 0.0,
                "total_tickets": 0,
                "processed_tickets": 0,
                "late_high_priority": 0,
                "total_high_priority": 0,
            },
            "trace": trace,
            "error": str(exc),
        }
    finally:
        close_method = getattr(env, "close", None)
        if callable(close_method):
            try:
                close_method()
            except Exception as exc:
                print(f"Warning: env.close() failed: {exc}", file=sys.stderr)
        log_end(success=success, steps=steps_taken, score=final_score, rewards=rewards)


def run_single_task(
    scenario_name: str,
    heuristic_only: bool = False,
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    if heuristic_only:
        return run_episode(scenario_name=scenario_name, heuristic_only=True, max_steps=max_steps)

    try:
        client, model_name = build_client()
        return run_episode(
            scenario_name=scenario_name,
            heuristic_only=False,
            max_steps=max_steps,
            client=client,
            model_name=model_name,
        )
    except Exception as exc:
        print(
            f"Warning: falling back to heuristic inference for scenario '{scenario_name}': {exc}",
            file=sys.stderr,
        )
        return run_episode(scenario_name=scenario_name, heuristic_only=True, max_steps=max_steps)


def run_all_tasks(
    heuristic_only: bool = False,
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    requested_runner = "heuristic" if heuristic_only else "openai-client"
    runner = requested_runner
    client = None
    model_name = None

    if not heuristic_only:
        try:
            client, model_name = build_client()
        except Exception as exc:
            heuristic_only = True
            runner = "heuristic-fallback"
            print(f"Warning: falling back to heuristic inference because LLM setup failed: {exc}", file=sys.stderr)

    reports = []
    for name in scenario_names():
        if heuristic_only:
            reports.append(run_episode(scenario_name=name, heuristic_only=True, max_steps=max_steps))
        else:
            reports.append(
                run_episode(
                    scenario_name=name,
                    heuristic_only=False,
                    max_steps=max_steps,
                    client=client,
                    model_name=model_name,
                )
            )

    scores = [float(report["grade"]["final_score"]) for report in reports]
    average_score = round(sum(scores) / len(scores), 4) if scores else 0.0
    return {
        "runner": runner,
        "requested_runner": requested_runner,
        "model_name": model_name if model_name else "heuristic-baseline",
        "average_final_score": average_score,
        "tasks": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the support-triage baseline across one or all tasks.")
    parser.add_argument("--scenario", choices=["all", *scenario_names()], default="all")
    parser.add_argument("--heuristic-only", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    args = parser.parse_args()

    try:
        if args.scenario == "all":
            run_all_tasks(heuristic_only=args.heuristic_only, max_steps=args.max_steps)
        else:
            run_single_task(
                scenario_name=args.scenario,
                heuristic_only=args.heuristic_only,
                max_steps=args.max_steps,
            )
    except Exception as exc:
        print(f"Fatal inference error: {exc}", file=sys.stderr)
        run_all_tasks(heuristic_only=True, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
