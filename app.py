import argparse
import uuid
from typing import Any, Dict, Optional

from fastapi import Body, FastAPI, HTTPException
from pydantic import BaseModel

from agent.baseline import choose_action
from env.core import TicketEnv
from env.models import Action, GradeResult, Observation, Reward
from env.tasks import scenario_names


app = FastAPI(
    title="Support Triage OpenEnv",
    description="Deterministic support-ticket triage benchmark with OpenEnv-style reset/step/state APIs.",
    version="1.0.0",
)

_SESSIONS: Dict[str, TicketEnv] = {}


class ResetRequest(BaseModel):
    scenario_name: str = "easy"
    max_steps: Optional[int] = None
    session_id: Optional[str] = None


class ResetResponse(BaseModel):
    session_id: str
    observation: Observation
    available_tasks: list[str]


class StepResponse(BaseModel):
    observation: Observation
    reward: Reward
    done: bool
    info: Dict[str, Any]


class GradeResponse(BaseModel):
    session_id: str
    grade: GradeResult


def _build_env(scenario_name: str, max_steps: Optional[int]) -> TicketEnv:
    if scenario_name not in scenario_names():
        raise HTTPException(
            status_code=400,
            detail={"error": "unknown_scenario", "available_tasks": scenario_names()},
        )

    config: Dict[str, Any] = {"scenario_name": scenario_name}
    if max_steps is not None:
        config["max_steps"] = max_steps

    return TicketEnv(config)


def _reset_session(request: ResetRequest) -> ResetResponse:
    env = _build_env(request.scenario_name, request.max_steps)
    observation = env.reset()
    session_id = request.session_id or uuid.uuid4().hex
    _SESSIONS[session_id] = env
    return ResetResponse(
        session_id=session_id,
        observation=observation,
        available_tasks=scenario_names(),
    )


def _get_session_env(session_id: str) -> TicketEnv:
    env = _SESSIONS.get(session_id)
    if env is None:
        raise HTTPException(
            status_code=404,
            detail={"error": "unknown_session", "session_id": session_id},
        )
    return env


@app.get("/")
def root() -> Dict[str, Any]:
    return {
        "name": "support-triage-env",
        "status": "ok",
        "available_tasks": scenario_names(),
        "routes": ["/health", "/tasks", "/reset", "/state/{session_id}", "/step/{session_id}", "/grade/{session_id}"],
    }


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/tasks")
def tasks() -> Dict[str, list[str]]:
    return {"tasks": scenario_names()}


@app.get("/reset", response_model=ResetResponse)
def reset_get(
    scenario_name: str = "easy",
    max_steps: Optional[int] = None,
) -> ResetResponse:
    return _reset_session(ResetRequest(scenario_name=scenario_name, max_steps=max_steps))


@app.post("/reset", response_model=ResetResponse)
def reset_post(request: Optional[ResetRequest] = Body(default=None)) -> ResetResponse:
    return _reset_session(request or ResetRequest())


@app.get("/state/{session_id}", response_model=Observation)
def state(session_id: str) -> Observation:
    return _get_session_env(session_id).state()


@app.post("/step/{session_id}", response_model=StepResponse)
def step(session_id: str, action: Action) -> StepResponse:
    env = _get_session_env(session_id)
    observation, reward, done, info = env.step(action)
    return StepResponse(observation=observation, reward=reward, done=done, info=info)


@app.get("/grade/{session_id}", response_model=GradeResponse)
def grade(session_id: str) -> GradeResponse:
    env = _get_session_env(session_id)
    return GradeResponse(session_id=session_id, grade=env.grade())


@app.delete("/session/{session_id}")
def delete_session(session_id: str) -> Dict[str, str]:
    _get_session_env(session_id)
    del _SESSIONS[session_id]
    return {"status": "deleted", "session_id": session_id}


def run_cli(scenario_name: str = "easy") -> None:
    env = TicketEnv({"scenario_name": scenario_name})
    state = env.reset()
    print("Scenario", scenario_name, "loaded with", state.pending_count, "tickets")

    done = False
    while not done:
        ticket = state.current_ticket
        if ticket is None:
            break
        action = choose_action(ticket)
        state, reward, done, info = env.step(action)
        print(
            "Handled",
            action.ticket_id,
            "step_score=",
            reward.step_score,
            "total_score=",
            reward.total_score,
            "done=",
            done,
        )
        if reward.breakdown:
            print("  reward breakdown:", reward.breakdown)

    grade_result = env.grade()
    print("Final score:", grade_result.final_score)
    print("Routing accuracy:", grade_result.routing_accuracy)
    print("Priority accuracy:", grade_result.priority_accuracy)
    print("SLA score:", grade_result.sla_score)
    print("Action accuracy:", grade_result.action_accuracy)


def main() -> None:
    parser = argparse.ArgumentParser(description="Local smoke runner for the support triage environment.")
    parser.add_argument("scenario", nargs="?", default="easy", choices=scenario_names())
    args = parser.parse_args()
    run_cli(args.scenario)


if __name__ == "__main__":
    main()
