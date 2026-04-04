from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


Department = Literal["billing", "technical", "general"]
Priority = Literal["low", "medium", "high"]
ActionType = Literal["resolve", "escalate", "request_info"]
CustomerTier = Literal["free", "premium"]


class GroundTruth(BaseModel):
    department: Department
    priority: Priority
    action_type: ActionType


class Ticket(BaseModel):
    id: str
    category_hint: Department
    description: str = ""
    urgency: int = Field(ge=1, le=5)
    customer_tier: CustomerTier
    time_waiting: int = Field(ge=0)

    model_config = {"extra": "ignore"}


class LabeledTicket(Ticket):
    ground_truth: GroundTruth
    ground_truth_reason: Optional[str] = None
    visible_at: int = 0


class DecisionRecord(BaseModel):
    ticket_id: str
    department: Department
    priority: Priority
    action_type: ActionType
    handled_at: int
    reward: float


class Observation(BaseModel):
    current_ticket: Optional[Ticket] = None
    pending_count: int = Field(default=0, ge=0)
    resolved_count: int = Field(default=0, ge=0)
    current_time: int = Field(default=0, ge=0)
    step_number: int = Field(default=0, ge=0)
    pending_tickets: List[Ticket] = Field(default_factory=list)
    resolved_tickets: List[DecisionRecord] = Field(default_factory=list)

    @property
    def tickets(self) -> List[Ticket]:
        return self.pending_tickets


class Action(BaseModel):
    ticket_id: str
    department: Department
    priority: Priority
    action_type: ActionType


class Reward(BaseModel):
    step_score: float = Field(ge=-1.0, le=1.0)
    total_score: float = Field(default=0.0, ge=0.0, le=1.0)
    breakdown: Dict[str, float] = Field(default_factory=dict)

    @property
    def value(self) -> float:
        return self.step_score

    @property
    def components(self) -> Dict[str, float]:
        return self.breakdown


class ArrivalWave(BaseModel):
    time: int
    tickets: List[LabeledTicket]


class Scenario(BaseModel):
    scenario_id: str
    difficulty: Literal["easy", "medium", "hard"]
    description: str
    start_time: int
    max_steps: int
    sla_targets_steps: Dict[str, int]
    initial_tickets: List[LabeledTicket]
    arrival_schedule: List[ArrivalWave]


class GradeResult(BaseModel):
    routing_accuracy: float
    priority_accuracy: float
    sla_score: float
    action_accuracy: float
    final_score: float
    total_tickets: int
    processed_tickets: int
    late_high_priority: int
    total_high_priority: int
