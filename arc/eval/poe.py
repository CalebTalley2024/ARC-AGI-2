# arc/poe.py
from __future__ import annotations

# PoE is sum of log-probs across views


def poe_sum(logps: list[float]) -> float:
    return float(sum(logps))
