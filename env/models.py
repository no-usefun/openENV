from pydantic import BaseModel
from typing import List, Literal


# -------------------------
# Ticket Model
# -------------------------
class Ticket(BaseModel):
    id: int
    text: str
    urgency: int  # 1–5
    customer_tier: Literal["free", "premium"]
    time_waiting: int


# -------------------------
# Observation
# -------------------------
class Observation(BaseModel):
    tickets: List[Ticket]
    current_time: int


# -------------------------
# Action
# -------------------------
class Action(BaseModel):
    ticket_id: int
    department: Literal["billing", "technical", "general"]
    priority: Literal["low", "medium", "high"]
    action_type: Literal["resolve", "escalate", "request_info"]


# -------------------------
# Reward
# -------------------------
class Reward(BaseModel):
    value: float