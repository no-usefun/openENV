# Task Data Notes

These files are the scenario source of truth for the support-triage benchmark.

The current `easy.json`, `medium.json`, and `hard.json` files are generated from
`customer_support_tickets_200k.csv` with the deterministic script
`scripts/generate_tasks_from_csv.py`.

## Scenario shape

Each scenario file contains:

- `scenario_id`: stable identifier for the task.
- `difficulty`: `easy`, `medium`, or `hard`.
- `description`: plain-language summary of what the task is testing.
- `start_time`: initial environment clock value.
- `max_steps`: hard cap on agent actions for the scenario.
- `sla_targets_steps`: suggested per-priority handling windows for future grader logic.
- `initial_tickets`: tickets visible immediately after `reset()`.
- `arrival_schedule`: deterministic ticket waves to inject later.

## Ticket shape

Each ticket contains:

- `id`
- `category_hint`: noisy classifier signal exposed to the agent.
- `description`
- `urgency`: integer from 1 to 5.
- `customer_tier`: `free` or `premium`.
- `time_waiting`: age of the ticket when it becomes visible.
- `ground_truth.department`
- `ground_truth.priority`
- `ground_truth.action_type`
- `ground_truth_reason`: debugging-only note that explains the label choice.

## Regeneration

If the source CSV changes, regenerate the benchmark tasks with:

```powershell
python scripts/generate_tasks_from_csv.py
```

