---
title: OpenENV Support Triage
emoji: "🎫"
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - support-triage
  - fastapi
---

# Support Triage OpenEnv

A deterministic customer-support ticket triage benchmark for evaluating routing, prioritization, and next-action selection under time pressure.

## What This Project Simulates

Support teams need to decide, for every incoming ticket:

1. Which department should handle it.
2. How urgent it is.
3. What action should happen next.

This environment turns that workflow into a reproducible benchmark with hidden labels, realistic ticket text, staged arrivals, and SLA pressure.

## Action And Observation Format

Each step expects an action like this:

```json
{
  "ticket_id": "E006",
  "department": "technical",
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

Ground-truth labels remain hidden from the acting agent.

## Available Scenarios

The repo currently exposes three task files from `tasks/*.json`:

| Scenario | Tickets | Arrival Pattern | Default Max Steps |
|---|---:|---|---:|
| `easy` | 10 | No new arrivals | 12 |
| `medium` | 15 | Two staged waves | 16 |
| `hard` | 25 | Continuous arrivals | 22 |

Difficulty increases through noisier hints, more interruptions, and stronger SLA pressure.

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

## API Endpoints

Base URL when running locally with Docker or Uvicorn: `http://localhost:7860`

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

## Live Deployment

Public deployed API URL:

- `https://tensura81-openenv.hf.space`

Quick checks:

```text
https://tensura81-openenv.hf.space/health
https://tensura81-openenv.hf.space/reset
```

## Local Setup

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the API locally

```bash
uvicorn server.app:app --host 0.0.0.0 --port 7860 --reload
```

### Run the Docker image locally

```bash
docker build -t support-triage-env .
docker run -p 7860:7860 support-triage-env
```

### Run the local smoke runner

```bash
python app.py easy
python app.py medium
python app.py hard
```

## Inference Runner

`inference.py` supports:

- heuristic-only mode
- OpenAI-client mode against an OpenAI-compatible endpoint

### Heuristic-only mode

```bash
python inference.py --heuristic-only
python inference.py --scenario medium --heuristic-only
```

### OpenAI-compatible mode

Required environment variables for hackathon evaluation:

- `HF_TOKEN` or `OPENAI_API_KEY`
- `API_BASE_URL`
- `MODEL_NAME`

PowerShell:

```powershell
$env:HF_TOKEN="your_api_key"
$env:MODEL_NAME="meta-llama/Llama-3.3-70B-Instruct"
$env:API_BASE_URL="https://router.huggingface.co/v1"
python inference.py
```

Bash:

```bash
export HF_TOKEN=your_api_key
export MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1
python inference.py
```

If credentials are missing or the model response is malformed, the script falls back to the heuristic policy instead of exiting with an error.

## Deployment Files

- [openenv.yaml](C:/project/openEnv/openENV2/openENV/openenv.yaml): OpenEnv runtime metadata
- [Dockerfile](C:/project/openEnv/openENV2/openENV/Dockerfile): container entrypoint and dependencies
- [server/app.py](C:/project/openEnv/openENV2/openENV/server/app.py): FastAPI app used by Docker/HF Space
- [inference.py](C:/project/openEnv/openENV2/openENV/inference.py): baseline runner used by validation

## Project Structure

```text
.
|-- app.py
|-- inference.py
|-- openenv.yaml
|-- Dockerfile
|-- requirements.txt
|-- pyproject.toml
|-- uv.lock
|-- agent/
|   |-- __init__.py
|   `-- baseline.py
|-- env/
|   |-- core.py
|   |-- environment.py
|   |-- grader.py
|   |-- models.py
|   |-- tasks.py
|   `-- tickets.py
|-- server/
|   |-- __init__.py
|   `-- app.py
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
```

## Notes

- Sessions are stored in memory, so restarting the server clears active runs.
- `openenv.yaml` declares the app entrypoint as `server.app:app` on port `7860`.
- The Dockerfile starts Uvicorn on port `7860`.
- The local Docker verification path is:
  - `GET /health`
  - `GET /reset`
  - `POST /step/{session_id}`
  - `GET /grade/{session_id}`

## Attribution

Built for the Meta PyTorch OpenEnv Hackathon by Team Tensura.
