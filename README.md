# Support Triage OpenEnv

Support Triage OpenEnv is a deterministic customer-support routing benchmark built for OpenEnv-style RL and agent evaluation. The agent must decide which ticket to handle next and assign:

- `department`: `billing`, `technical`, or `general`
- `priority`: `low`, `medium`, or `high`
- `action_type`: `resolve`, `escalate`, or `request_info`

The benchmark is grounded in the local dataset [customer_support_tickets_200k.csv](tasks/customer_support_tickets_200k.csv), but evaluation uses three reproducible tasks: `easy`, `medium`, and `hard`.

## Quick Start

### 1. Install dependencies

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

### 2. Run tests

```bash
python -m unittest discover -s tests -v
```

### 3. Run the local heuristic benchmark

```bash
python inference.py --heuristic-only
```

### 4. Run only the hard task

```bash
python inference.py --heuristic-only --scenario hard
```

### 5. Run the API locally

```bash
uvicorn app:app --host 0.0.0.0 --port 8000
```

Useful endpoints:

- `GET /health`
- `GET /tasks`
- `GET /reset`
- `POST /reset`
- `GET /state/{session_id}`
- `POST /step/{session_id}`
- `GET /grade/{session_id}`

## Current Baseline

Current heuristic smoke scores:

- `easy`: `0.9583`
- `medium`: `0.9875`
- `hard`: `0.9318`
- overall average: `0.9592`

The hard task is intentionally harder because several high-priority tickets overlap under a tight SLA window.

## Daily Commands

Run tests:

```bash
python -m unittest discover -s tests -v
```

Run all tasks with the heuristic baseline:

```bash
python inference.py --heuristic-only
```

Run a single task:

```bash
python inference.py --heuristic-only --scenario medium
python inference.py --heuristic-only --scenario hard
```

Run the CLI smoke app:

```bash
python app.py
python app.py medium
python app.py hard
```

Build Docker:

```bash
docker build -t support-triage-env .
```

Run Docker:

```bash
docker run -p 8000:8000 support-triage-env
```

## Environment Contract

### Observation

The environment exposes a typed Pydantic `Observation` with:

- `current_ticket`
- `pending_count`
- `resolved_count`
- `current_time`
- `step_number`
- `pending_tickets`
- `resolved_tickets`

Each visible `Ticket` contains:

- `id`
- `category_hint`
- `specialist_team`
- `description`
- `urgency`
- `customer_tier`
- `time_waiting`

`specialist_team` is a realistic queue hint such as `payments_ops`, `security`, or `product_bug`. It is visible to the agent, but the scored routing target remains the coarser 3-way `department` label.

### Action

The typed `Action` model contains:

- `ticket_id`
- `department`
- `priority`
- `action_type`

The benchmark is intentionally judged on exactly these three routing labels:

- `billing`
- `technical`
- `general`

### Reward

Dense step reward is implemented in `env/core.py`.

Positive:

- correct department: `+0.2`
- correct priority: `+0.15`
- correct action: `+0.1`

Negative:

- wrong department: `-0.3`
- ignoring an urgent ticket while handling a less urgent one: `-0.5`
- delay penalty: `-0.05 * current_time`

Each step returns a typed `Reward` with:

- `step_score` in `[-1, 1]`
- `total_score` normalized to `[0, 1]`
- `breakdown`

### Grader

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

## Tasks

The benchmark ships with three deterministic tasks:

- `easy`: 6 tickets, clean-ish hints, no arrivals
- `medium`: 12 total tickets, noisy hints, scheduled arrivals
- `hard`: 24 total tickets, continuous arrivals, high overlap

Task files:

- `tasks/easy.json`
- `tasks/medium.json`
- `tasks/hard.json`

The task files are generated from the CSV source, not hand-maintained line by line.

## Task Generation Workflow

Source dataset:

- `tasks/customer_support_tickets_200k.csv`

Generator:

- `scripts/generate_tasks_from_csv.py`

Regenerate tasks:

```bash
python scripts/generate_tasks_from_csv.py
```

After regeneration, always run:

```bash
python -m unittest discover -s tests -v
python inference.py --heuristic-only
```

## What To Edit

### If you want to tune ticket wording or scenario composition

Edit:

- `scripts/generate_tasks_from_csv.py`
- optional: `tasks/README.md`

Then regenerate:

```bash
python scripts/generate_tasks_from_csv.py
```

### If you want to add more realism without changing the scored action space

Good place to add fields like `specialist_team`, `queue`, or other visible hints:

- `env/models.py`
- `scripts/generate_tasks_from_csv.py`
- `agent/baseline.py`
- `inference.py`
- `tests/test_tasks.py`
- `README.md`

This is the safest kind of realism upgrade.

### If you want to change the scored department space

Do not change only the JSON.

To expand or rename scored departments, you must update all of:

- `env/models.py`
- `scripts/generate_tasks_from_csv.py`
- `agent/baseline.py`
- `inference.py`
- `tests/`
- `README.md`
- generated task files in `tasks/`

Otherwise the environment will fail validation or silently mis-score actions.

### If you want to tune hard-task difficulty

The main knobs are:

- `max_steps`
- `arrival_schedule`
- `sla_targets_steps`
- which tickets are marked high priority

Those are all controlled through `scripts/generate_tasks_from_csv.py`.

## Model-Backed Inference

`inference.py` can run either:

- heuristic-only mode
- OpenAI-client mode

Required environment variables for model-backed mode:

- `MODEL_NAME`
- `API_BASE_URL`
- `OPENAI_API_KEY` or `HF_TOKEN`

Examples:

```powershell
$env:HF_TOKEN="your_token_here"
$env:API_BASE_URL="https://router.huggingface.co/v1"
$env:MODEL_NAME="deepseek-ai/DeepSeek-V3-0324"
python inference.py
```

## Docker

Build:

```bash
docker build -t support-triage-env .
```

Run:

```bash
docker run -p 8000:8000 support-triage-env
```

Container verification:

```powershell
Invoke-RestMethod http://127.0.0.1:8000/health
Invoke-RestMethod http://127.0.0.1:8000/reset
```

## Hugging Face Space Deploy

This repo is set up to deploy as a Docker Space.

Key files:

- `Dockerfile`
- `app.py`
- `openenv.yaml`

Typical deploy flow:

1. Create a Docker Space on Hugging Face.
2. Add secrets like `HF_TOKEN`.
3. Add variables like `API_BASE_URL` and `MODEL_NAME`.
4. Push this repo to the Space remote.
5. Verify `/health` and `/reset` on the deployed URL.

## Repo Map

```text
support-triage-env/
|-- agent/
|   `-- baseline.py
|-- env/
|   |-- core.py
|   |-- environment.py
|   |-- grader.py
|   |-- models.py
|   |-- tasks.py
|   `-- tickets.py
|-- scripts/
|   `-- generate_tasks_from_csv.py
|-- tasks/
|   |-- customer_support_tickets_200k.csv
|   |-- easy.json
|   |-- medium.json
|   |-- hard.json
|   `-- README.md
|-- tests/
|   |-- test_api.py
|   |-- test_environment.py
|   |-- test_grader.py
|   |-- test_inference.py
|   `-- test_tasks.py
|-- app.py
|-- inference.py
|-- openenv.yaml
|-- Dockerfile
`-- requirements.txt
```

## Troubleshooting

### Score suddenly dropped after regenerating tasks

Check whether `scripts/generate_tasks_from_csv.py` changed:

- `hard.max_steps`
- `sla_targets_steps`
- arrival timing
- priority/action derivation rules

Then rerun:

```bash
python scripts/generate_tasks_from_csv.py
python -m unittest discover -s tests -v
python inference.py --heuristic-only
```

### Docker works locally but score changed

Docker only proves packaging and runtime health. It does not prove the benchmark logic stayed the same. Always rerun tests and the heuristic benchmark after task or generator changes.

### Want more realism but do not want to break validation

Add non-scored visible fields like `specialist_team` instead of changing the scored `department` action space.
