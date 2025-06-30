"""
Utility functions for calculating and categorizing theorem difficulties.
"""

import math
from typing import List, Union

from database.models import Theorem


def calculate_difficulty(theorem: Theorem) -> Union[float, None]:
    """Calculates the difficulty of a theorem."""
    proof_steps = theorem.traced_tactics
    if any("sorry" in step.tactic for step in proof_steps):
        return float("inf")  # Hard (no proof)
    if len(proof_steps) == 0:
        return None  # To be distributed later
    return math.exp(len(proof_steps))


def categorize_difficulty(
    difficulty: Union[float, None], percentiles: List[float]
) -> str:
    """Categorizes the difficulty of a theorem."""
    if difficulty is None:
        return "To_Distribute"
    if difficulty == float("inf"):
        return "Hard (No proof)"
    elif difficulty <= percentiles[0]:
        return "Easy"
    elif difficulty <= percentiles[1]:
        return "Medium"
    else:
        return "Hard"
