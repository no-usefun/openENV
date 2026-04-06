---
title: OpenENV Support Triage
emoji: 🎫
colorFrom: blue
colorTo: purple
sdk: docker
app_file: app.py
pinned: false
tags:
  - openenv
  - reinforcement-learning
  - support-triage
  - fastapi
---

# 🎫 Support Triage OpenEnv

> **Meta PyTorch OpenEnv Hackathon — Round 1 Submission**  
> Team **Tensura** · [Live Space](https://huggingface.co/spaces/Tensura81/openENV)

A deterministic, real-world **customer support ticket triage** benchmark for evaluating AI agents on routing, prioritization, and action selection — built on the [OpenEnv](https://github.com/meta-pytorch/OpenEnv) framework.

---

## 🌍 Real-World Task

Support teams receive hundreds of tickets daily. Humans must:
1. **Route** each ticket to the right department (billing / technical / general)
2. **Prioritize** urgency-weighted tickets within SLA deadlines
3. **Take action** — resolve, escalate, or request more info

This environment simulates exactly that workflow, with realistic ticket descriptions, staged arrival waves, and SLA time pressure.

---

## 🎮 Action & Observation Spaces

### Action (per step)
```json
{
  "ticket_id": "E006",
  "department": "billing | technical | general",
  "priority":   "low | medium | high",
  "action_type": "resolve | escalate | request_info"
}
```

### Observation (returned after each step)
```json
{
  "current_ticket": {
    "id": "E006",
    "category_hint": "technical",
    "description": "Users cannot log in — account appears suspended.",
    "urgency": 5,
    "customer_tier": "premium",
    "time_waiting": 85
  },
  "pending_tickets": [...],
  "pending_count": 9,
  "resolved_count": 1,
  "current_time": 2,
  "step_number": 2
}
```

> Ground truth labels stay **hidden** from the agent during play.

---

## 📋 Tasks (Easy → Medium → Hard)

| Scenario | Tickets | Arrivals | Max Steps | Avg Heuristic Score |
|---|---|---|---|---|
| `easy` | 10 | None | 12 | **0.9375** |
| `medium` | 15 | 2 staged waves | 16 | **0.8900** |
| `hard` | 25 | Continuous waves | 22 | **0.7758** |

### Difficulty progression
- **Easy**: Simple tickets with accurate category hints, no time pressure
- **Medium**: Ambiguous tickets, staged arrivals that interrupt workflow, moderate SLA pressure
- **Hard**: Misleading hints, continuous new arrivals, SLA deadlines for 11 high-priority tickets, fewer steps than total tickets

---

## 📊 Scoring

### Step Reward
| Event | Score |
|---|---|
| Correct department | `+0.20` |
| Correct priority | `+0.15` |
| Correct action | `+0.10` |
| Wrong department | `−0.30` |
| Skip urgent ticket (urgency ≥ 4) | `−0.50` |
| Time delay penalty | `−0.05 × current_time` |

### Final Score (weighted average)
```
final_score = 0.35 × routing_accuracy
            + 0.25 × priority_accuracy
            + 0.25 × sla_score
            + 0.15 × action_accuracy
```

SLA targets: high-priority tickets must be handled within **3 steps** of appearing.

---

## 🚀 API Endpoints

Base URL: `https://tensura81-openenv.hf.space`

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Health check + available routes |
| `GET` | `/tasks` | List available scenarios |
| `POST` | `/reset?scenario_name=easy` | Start a new episode |
| `GET` | `/state/{session_id}` | Get current environment state |
| `POST` | `/step/{session_id}` | Submit action, get next state + reward |
| `GET` | `/grade/{session_id}` | Get final episode grade |

### Example Full Episode

```python
import requests

BASE = "https://tensura81-openenv.hf.space"

# 1. Start episode
r = requests.post(f"{BASE}/reset", params={"scenario_name": "easy"})
session_id = r.json()["session_id"]

done = False
while not done:
    # 2. Get current ticket
    state = requests.get(f"{BASE}/state/{session_id}").json()
    ticket = state["current_ticket"]

    # 3. Submit action (always use current_ticket id)
    action = {
        "ticket_id": ticket["id"],
        "department": "technical",
        "priority": "high",
        "action_type": "escalate"
    }
    result = requests.post(f"{BASE}/step/{session_id}", json=action).json()
    done = result["done"]

# 4. Get final score
grade = requests.get(f"{BASE}/grade/{session_id}").json()
print(grade["grade"]["final_score"])  # e.g. 0.9375
```

---

## 🏃 Running the Baseline

```bash
# Clone & install
git clone https://huggingface.co/spaces/Tensura81/openENV
cd openENV
pip install -r requirements.txt

# Set LLM credentials (judges provide their own)
export HF_TOKEN=your_token
export MODEL_NAME=meta-llama/Llama-3.3-70B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1

# Run baseline inference (all 3 scenarios)
python inference.py

# Run heuristic-only (no API key needed)
python inference.py --heuristic-only

# Run unit tests
python -m unittest discover -s tests -v
```

### Baseline Scores (Heuristic Agent)

| Scenario | Routing | Priority | SLA | Action | **Final** |
|---|---|---|---|---|---|
| Easy | 100% | 90% | 100% | 80% | **0.9375** |
| Medium | 86.7% | 86.7% | 100% | 80% | **0.8900** |
| Hard | 80% | 80% | 72.7% | 76% | **0.7758** |
| **Average** | | | | | **0.8678** |

---

## 🐳 Docker

```bash
docker build -t support-triage-env .
docker run -p 7860:7860 support-triage-env
# API available at http://localhost:7860
```

---

## 📁 Project Structure

```
├── app.py              # FastAPI server (reset/step/state/grade endpoints)
├── inference.py        # Baseline LLM + heuristic inference runner
├── openenv.yaml        # OpenEnv environment manifest
├── Dockerfile          # Container configuration
├── requirements.txt    # Dependencies
├── agent/
│   └── baseline.py     # Rule-based heuristic policy
├── env/
│   ├── core.py         # Environment step/reset logic + SLA tracking
│   ├── grader.py       # Final scoring and grade computation
│   ├── models.py       # Typed Pydantic models (Action, Observation, Ticket)
│   └── tasks.py        # Scenario loader
└── tasks/
    ├── easy.json        # 10 tickets, no arrivals
    ├── medium.json      # 15 tickets, 2 arrival waves
    └── hard.json        # 25 tickets, continuous arrivals
```

---

## 🏆 Hackathon

Built for the **[Meta PyTorch OpenEnv Hackathon](https://www.scaler.com/school-of-technology/meta-pytorch-hackathon)** by Team **Tensura**.