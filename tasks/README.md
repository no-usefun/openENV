# Task Data Guide

These JSON files are the source of truth for the benchmark scenarios.

## Included Scenarios

- `easy.json`: `10` tickets, no arrivals
- `medium.json`: `15` total tickets with staged arrivals
- `hard.json`: `25` total tickets with continuous arrivals and a tight step budget

## Scenario Fields

Each scenario file contains:

- `scenario_id`: stable task identifier
- `difficulty`: `easy`, `medium`, or `hard`
- `description`: short summary of the scenario
- `start_time`: starting clock value
- `max_steps`: maximum number of agent actions allowed
- `sla_targets_steps`: target response windows by priority
- `initial_tickets`: visible at reset
- `arrival_schedule`: future ticket waves

## Ticket Fields

Each ticket contains:

- `id`
- `category_hint`
- `description`
- `urgency`: `1` to `5`
- `customer_tier`: `free` or `premium`
- `time_waiting`
- `ground_truth.department`
- `ground_truth.priority`
- `ground_truth.action_type`
- `ground_truth_reason`

## Allowed Classification Values

Department values:

- `billing`
- `technical`
- `general`

Priority values:

- `low`
- `medium`
- `high`

Action values:

- `resolve`
- `escalate`
- `request_info`

Customer tier values:

- `free`
- `premium`

## Quick Classification Guide

Use `billing` for:

- refunds
- charges
- invoices
- renewals
- subscription or payment disputes
- unpaid workspace or failed transaction issues

Use `technical` for:

- login or sign-in failures
- 2FA or authentication issues
- account lock or suspension
- bugs, crashes, errors, sync failures
- outages or severe performance problems

Use `general` for:

- feature questions
- plan questions
- how-to questions
- product capability questions

Use `resolve` when:

- the request is clear and can be handled directly

Use `escalate` when:

- the issue is severe, blocking, or specialist-owned
- examples: outages, access failures, security issues

Use `request_info` when:

- the ticket is vague or missing important details
- examples: "not sure", "maybe", "something is wrong", unclear invoice or workspace

## Notes

- `category_hint` is noisy and should not be treated as ground truth
- `ground_truth_reason` is for debugging and dataset inspection
- the benchmark may intentionally prevent full completion in harder scenarios
