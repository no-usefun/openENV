from typing import List

from .models import DecisionRecord, GradeResult, LabeledTicket, Scenario


def grade_episode(
    scenario: Scenario,
    decisions: List[DecisionRecord],
    pending_tickets: List[LabeledTicket],
) -> GradeResult:
    all_tickets = _flatten_tickets(scenario)
    total_tickets = len(all_tickets)
    decision_map = {decision.ticket_id: decision for decision in decisions}

    correct_department = 0
    correct_priority = 0
    correct_action = 0
    total_high_priority = 0
    late_high_priority = 0

    for ticket in all_tickets:
        decision = decision_map.get(ticket.id)
        if decision and decision.department == ticket.ground_truth.department:
            correct_department += 1
        if decision and decision.priority == ticket.ground_truth.priority:
            correct_priority += 1
        if decision and decision.action_type == ticket.ground_truth.action_type:
            correct_action += 1

        if ticket.ground_truth.priority == "high":
            total_high_priority += 1
            if decision is None or _is_late(ticket, decision, scenario):
                late_high_priority += 1

    routing_accuracy = _safe_divide(correct_department, total_tickets)
    priority_accuracy = _safe_divide(correct_priority, total_tickets)
    action_accuracy = _safe_divide(correct_action, total_tickets)
    sla_score = 1.0 if total_high_priority == 0 else 1.0 - (
        late_high_priority / total_high_priority
    )

    final_score = (
        0.35 * routing_accuracy
        + 0.25 * priority_accuracy
        + 0.25 * sla_score
        + 0.15 * action_accuracy
    )

    return GradeResult(
        routing_accuracy=round(routing_accuracy, 4),
        priority_accuracy=round(priority_accuracy, 4),
        sla_score=round(max(0.0, sla_score), 4),
        action_accuracy=round(action_accuracy, 4),
        final_score=round(max(0.0, min(1.0, final_score)), 4),
        total_tickets=total_tickets,
        processed_tickets=len(decisions),
        late_high_priority=late_high_priority,
        total_high_priority=total_high_priority,
    )


def _flatten_tickets(scenario: Scenario) -> List[LabeledTicket]:
    tickets = []
    for ticket in scenario.initial_tickets:
        cloned = _clone_ticket(ticket)
        cloned.visible_at = scenario.start_time
        tickets.append(cloned)

    for arrival in scenario.arrival_schedule:
        for ticket in arrival.tickets:
            cloned = _clone_ticket(ticket)
            cloned.visible_at = arrival.time
            tickets.append(cloned)
    return tickets


def _is_late(ticket: LabeledTicket, decision: DecisionRecord, scenario: Scenario) -> bool:
    sla_limit = scenario.sla_targets_steps["high"]
    return (decision.handled_at - ticket.visible_at) > sla_limit


def _safe_divide(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _clone_ticket(ticket: LabeledTicket) -> LabeledTicket:
    return ticket.model_copy(deep=True)
