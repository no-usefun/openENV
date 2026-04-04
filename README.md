# Support Triage OpenEnv

Support Triage OpenEnv is a deterministic customer-support routing benchmark built for OpenEnv-style RL and agent evaluation. The environment simulates the work of a support triage operator who must decide which ticket to handle next and assign the correct department, priority, and action under time pressure.

## Why this environment

Real support teams do not just classify tickets. They choose what to handle first, decide whether a case is billing or technical, and balance fast resolutions against escalation and SLA risk. This benchmark captures those tradeoffs in a simple, reproducible environment.

## Core Objective

Given an incoming support queue, the agent must choose one ticket and assign:

- `department`: `billing`, `technical`, or `general`
- `priority`: `low`, `medium`, or `high`
- `action_type`: `resolve`, `escalate`, or `request_info`

The goal is to maximize correct routing, prioritize urgent tickets, and avoid SLA violations.

## Observation Space

The environment exposes a typed Pydantic `Observation` with:

- `current_ticket`: the next ticket surfaced by the queue ordering
- `pending_count`: number of tickets still waiting
- `resolved_count`: number of tickets already handled
- `current_time`: simulated environment time
- `step_number`: number of actions taken so far
- `pending_tickets`: visible ticket queue
- `resolved_tickets`: previously taken decisions

Each visible `Ticket` contains:

- `id`
- `category_hint`
- `description`
- `urgency`
- `customer_tier`
- `time_waiting`

Ground-truth labels are stored internally in the task files and are not shown to the agent.

## Action Space

The typed `Action` model contains:

- `ticket_id`
- `department`
- `priority`
- `action_type`

An action is valid only if the values match the allowed literals in [env/models.py](env/models.py).

## Reward Function

Dense step reward is implemented in [env/core.py](env/core.py).

Positive signals:

- correct department: `+0.2`
- correct priority: `+0.15`
- correct action: `+0.1`

Negative signals:

- wrong department: `-0.3`
- ignoring an urgent ticket while handling a less urgent one: `-0.5`
- delay penalty: `-0.05 * current_time`

Each step returns a typed `Reward` with:

- `step_score` in `[-1, 1]`
- `total_score` normalized to `[0, 1]`
- `breakdown` for debugging

## Grader

Final scoring is deterministic and normalized to `[0, 1]`.

Metrics:

- routing accuracy = correct department / total tickets
- priority accuracy = correct priority / total tickets
- SLA score = `1 - (late_high_priority / total_high_priority)`
- action accuracy = correct action / total tickets

Final score:

```text
0.35 * routing_accuracy +
0.25 * priority_accuracy +
0.25 * sla_score +
0.15 * action_accuracy
```

This logic is implemented in [env/grader.py](env/grader.py).

## Tasks

The benchmark ships with three deterministic tasks:

- `easy`: 6 tickets, mostly clean category hints, no new arrivals
- `medium`: 12 total tickets, noisy hints, scheduled arrivals, visible SLA pressure
- `hard`: 24 total tickets, continuous arrivals, limited steps, forced tradeoffs

Task definitions live in [tasks/easy.json](tasks/easy.json), [tasks/medium.json](tasks/medium.json), and [tasks/hard.json](tasks/hard.json).

## Project Structure

```text
support-triage-env/
├── agent/
│   └── baseline.py
├── env/
│   ├── core.py
│   ├── environment.py
│   ├── grader.py
│   ├── models.py
│   ├── tasks.py
│   └── tickets.py
├── tasks/
│   ├── easy.json
│   ├── medium.json
│   └── hard.json
├── tests/
│   ├── test_environment.py
│   ├── test_grader.py
│   └── test_tasks.py
├── app.py
├── inference.py
├── openenv.yaml
├── Dockerfile
└── requirements.txt
```

## Setup

### Local Python setup

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Run tests

```bash
python -m unittest discover -s tests -v
```

## Local Usage

### CLI smoke test

Run the heuristic baseline against a single task:

```bash
python app.py
python app.py medium
python app.py hard
```

### HTTP server

Run the FastAPI app locally:

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Useful endpoints:

- `GET /`
- `GET /health`
- `GET /tasks`
- `GET /reset`
- `POST /reset`
- `GET /state/{session_id}`
- `POST /step/{session_id}`
- `GET /grade/{session_id}`

Example reset call:

```bash
curl -X POST http://127.0.0.1:8000/reset \
  -H "Content-Type: application/json" \
  -d '{"scenario_name":"easy"}'
```

## Baseline Inference

`inference.py` runs a reproducible baseline over one task or all tasks.

Required environment variables:

- `MODEL_NAME`
- `API_BASE_URL` (optional for native OpenAI, defaults to `https://api.openai.com/v1`)
- `OPENAI_API_KEY` or `HF_TOKEN`

Run all tasks with the OpenAI client:

```bash
python inference.py
```

Run a single task:

```bash
python inference.py --scenario medium
```

Run the heuristic fallback only:

```bash
python inference.py --heuristic-only
```

The baseline prints JSON with per-task scores, fallback counts, and the average final score.

## Verified Local Smoke Scores

These are the current heuristic smoke-test scores from the local environment:

- `easy`: `0.9583`
- `medium`: `0.8667`
- `hard`: `0.55`

Model-based scores depend on the configured `MODEL_NAME` and API endpoint.

## Docker

Build:

```bash
docker build -t support-triage-env .
```

Run:

```bash
docker run -p 8000:8000 support-triage-env
```

## Hugging Face Spaces

The repository includes:

- a Dockerfile for containerized deployment
- a FastAPI entrypoint exposed through [app.py](app.py)
- [openenv.yaml](openenv.yaml) configured for `runtime: fastapi`

This makes the project ready to package as a Docker-based Hugging Face Space tagged for OpenEnv.
