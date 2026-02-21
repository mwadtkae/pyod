# group_rule.py
from dataclasses import dataclass
from typing import Optional

@dataclass
class GroupRule:
    name: str
    project: str
    direction: str               # "LOW" or "HIGH"
    threshold: float             # threshold for flagging
    mean_hours: float
    mode_hours: float
    min_hours: float
    max_hours: float
    n_rows: int
    notes: Optional[str] = None

    def is_flagged(self, hours: float) -> bool:
        if self.direction == "LOW":
            return hours < self.threshold
        if self.direction == "HIGH":
            return hours > self.threshold
        return False
