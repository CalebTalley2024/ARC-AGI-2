# arc/poe.py
from __future__ import annotations

from typing import List

# PoE is sum of log-probs across views

def poe_sum(logps: List[float]) -> float:
    return float(sum(logps))