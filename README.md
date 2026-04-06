---
title: OpenENV Support Triage
emoji: 🤖
colorFrom: blue
colorTo: green
sdk: docker
app_file: app.py
pinned: false
---

# Support Triage OpenEnv

A deterministic support-ticket triage benchmark for evaluating routing, prioritization, and action selection.

## What The Agent Does

For one chosen ticket, the agent predicts:

- `department`: `billing`, `technical`, or `general`
- `priority`: `low`, `medium`, or `high`
- `action_type`: `resolve`, `escalate`, or `request_info`

The environment scores both correctness and queue management under time pressure.

## Scenario Overview

- `easy`: `10` tickets, no arrivals, forgiving step budget
- `medium`: `15` total tickets, staged arrivals, moderate pressure
- `hard`: `25` total tickets, continuous arrivals, fewer steps than tickets

Task files:

- [easy.json](G:\TraingingLLM\tasks\easy.json)
- [medium.json](G:\TraingingLLM\tasks\medium.json)
- [hard.json](G:\TraingingLLM\tasks\hard.json)

## Observation

Each visible ticket includes:

- `id`
- `category_hint`
- `description`
- `urgency` from `1` to `5`
- `customer_tier`: `free` or `premium`
- `time_waiting`

The environment also returns:

- `current_ticket`
- `pending_tickets`
- `pending_count`
- `resolved_count`
- `current_time`
- `step_number`

Ground truth stays hidden from the agent during play.

## Scoring

Step reward:

- correct department: `+0.2`
- correct priority: `+0.15`
- correct action: `+0.1`
- wrong department: `-0.3`
- ignoring an urgent ticket: `-0.5`
- delay penalty: `-0.05 * current_time`

Final score:

```text
0.35 * routing_accuracy +
0.25 * priority_accuracy +
0.25 * sla_score +
0.15 * action_accuracy
```

## Quick Start

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Run tests:

```powershell
python -m unittest discover -s tests -v
```

Run the local CLI:

```powershell
python app.py
python app.py medium
python app.py hard
```

Run the heuristic benchmark:

```powershell
python inference.py --heuristic-only
```

Run the API:

```powershell
uvicorn app:app --host 0.0.0.0 --port 8000
```

Useful endpoints:

- `GET /health`
- `GET /tasks`
- `POST /reset`
- `GET /state/{session_id}`
- `POST /step/{session_id}`
- `GET /grade/{session_id}`

## Current Heuristic Scores

- `easy`: `0.9375`
- `medium`: `0.89`
- `hard`: `0.7758`

## Repo Layout

- [app.py](G:\TraingingLLM\app.py): FastAPI app and local CLI entrypoint
- [inference.py](G:\TraingingLLM\inference.py): heuristic and model runner
- [agent/baseline.py](G:\TraingingLLM\agent\baseline.py): baseline policy
- [env/core.py](G:\TraingingLLM\env\core.py): environment step/reset logic
- [env/grader.py](G:\TraingingLLM\env\grader.py): final grading
- [tasks/README.md](G:\TraingingLLM\tasks\README.md): task format and classification guide
