---
title: OpenENV Support Triage
emoji: ":ticket:"
colorFrom: blue
colorTo: indigo
sdk: docker
app_file: app.py
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - support-triage
  - fastapi
---

# Support Triage OpenEnv

A deterministic customer-support ticket triage benchmark for evaluating agents on three decisions at once: routing, priority assignment, and next-action selection. The project follows an OpenEnv-style setup and includes a FastAPI app, local runners, scenario files, and a baseline policy.

## What This Project Simulates

Support teams need to decide, for every incoming ticket:

1. Which department should handle it.
2. How urgent it is.
3. What action should happen next.

This environment turns that workflow into a reproducible benchmark with hidden labels, realistic ticket text, staged arrivals, and SLA pressure.

## Action and Observation Format

Each step expects an action like this:

```json
{
  "ticket_id": "E006",
  "department": "billing",
  "priority": "high",
  "action_type": "escalate"
}
```

The environment returns an observation shaped like this:

```json
{
  "current_ticket": {
    "id": "E006",
    "category_hint": "technical",
    "specialist_team": "account_access",
    "description": "Users cannot log in and the account appears suspended.",
    "urgency": 5,
    "customer_tier": "premium",
    "time_waiting": 85
  },
  "pending_tickets": [],
  "pending_count": 9,
  "resolved_count": 1,
  "current_time": 2,
  "step_number": 2,
  "resolved_tickets": []
}
```

Ground-truth labels are kept hidden from the acting agent.

## Available Scenarios

The repo currently exposes the canonical scenarios discovered from `tasks/*.json`:

| Scenario | Tickets | Arrival Pattern | Default Max Steps |
|---|---:|---|---:|
| `easy` | 10 | No new arrivals | 12 |
| `medium` | 15 | Two staged waves | 16 |
| `hard` | 25 | Continuous arrivals | 22 |

Difficulty increases through more ambiguous ticket language, more interruptions, and stronger SLA pressure.

## Scoring

### Step reward

| Event | Score |
|---|---:|
| Correct department | `+0.20` |
| Correct priority | `+0.15` |
| Correct action | `+0.10` |
| Wrong department | `-0.30` |
| Skip urgent ticket (`urgency >= 4`) | `-0.50` |
| Time delay penalty | `-0.05 * current_time` |

### Final grade

```text
final_score = 0.35 * routing_accuracy
            + 0.25 * priority_accuracy
            + 0.25 * sla_score
            + 0.15 * action_accuracy
```

High-priority tickets are expected to be handled within 3 steps of appearing.

## API Endpoints

Base URL when running locally with Uvicorn: `http://localhost:8000`

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Root metadata, routes, and scenario list |
| `GET` | `/health` | Health check |
| `GET` | `/tasks` | List available scenarios |
| `GET` | `/reset` | Create a session using query parameters |
| `POST` | `/reset` | Create a session using a JSON body |
| `GET` | `/state/{session_id}` | Fetch the current observation |
| `POST` | `/step/{session_id}` | Submit one action |
| `GET` | `/grade/{session_id}` | Get the final grade summary |
| `DELETE` | `/session/{session_id}` | Delete an in-memory session |

## Local Setup

### Requirements

- Python 3.10+
- `pip`

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the API locally

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Run the local smoke runner

```bash
python app.py easy
python app.py medium
python app.py hard
```

## Inference Runner

`inference.py` can run either:

- the built-in heuristic baseline
- an OpenAI-compatible chat-completions model

### Heuristic-only mode

```bash
python inference.py --heuristic-only
python inference.py --scenario medium --heuristic-only
```

### OpenAI-compatible mode

PowerShell:

```powershell
$env:OPENAI_API_KEY="your_api_key"
$env:MODEL_NAME="gpt-4.1-mini"
$env:API_BASE_URL="https://api.openai.com/v1"
```

Bash:

```bash
export OPENAI_API_KEY=your_api_key
export MODEL_NAME=gpt-4.1-mini
export API_BASE_URL=https://api.openai.com/v1
```

Then run:

```bash
python inference.py
python inference.py --scenario hard
```

If the model response is malformed or selects a ticket that is not pending, the runner falls back to the heuristic policy.

## Docker

Build and run the container with:

```bash
docker build -t support-triage-env .
docker run -p 7860:7860 support-triage-env
```

The Docker image serves the app on `http://localhost:7860`.

## Test Location

Main test folder:

- `./tests` (absolute path: `G:\\TraingingLLM\\tests`)

Additional root-level test file:

- `./test_deployed.py` (absolute path: `G:\\TraingingLLM\\test_deployed.py`)

## Project Structure

```text
.
|-- app.py
|-- inference.py
|-- openenv.yaml
|-- Dockerfile
|-- requirements.txt
|-- agent/
|   `-- baseline.py
|-- env/
|   |-- core.py
|   |-- environment.py
|   |-- grader.py
|   |-- models.py
|   |-- tasks.py
|   `-- tickets.py
|-- tasks/
|   |-- easy.json
|   |-- medium.json
|   `-- hard.json
|-- tests/
|   |-- test_api.py
|   |-- test_deployed.py
|   |-- test_environment.py
|   |-- test_grader.py
|   |-- test_inference.py
|   `-- test_tasks.py
`
```

## Notes

- Sessions are stored in memory, so restarting the server clears active runs.
- `openenv.yaml` declares the app entrypoint as `app:app` on port `8000`.
- The Dockerfile starts Uvicorn on port `7860`.
- The repo currently depends on FastAPI, Pydantic, Uvicorn, and the OpenAI Python client.

## Attribution

Built for the Meta PyTorch OpenEnv Hackathon by Team Tensura.
