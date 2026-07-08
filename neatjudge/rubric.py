"""Multi-axis judging rubrics.

The default judging task in NEATJudge scores two axes (safety + quality). A
:class:`Rubric` generalizes this to an arbitrary set of numeric axes, each *owned*
by a specialist archetype. This is what lets a multi-agent graph genuinely beat a
single judge: when several independent axes must be assessed, a specialist that
focuses on one axis can rate it better than a generalist juggling all of them.

``HELPSTEER_RUBRIC`` maps three NVIDIA HelpSteer2 human-labeled axes to distinct
specialists so each owns a real, separable piece of the task.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass(frozen=True)
class Axis:
    name: str            # e.g. "correctness"
    lo: int              # inclusive low (worst)
    hi: int              # inclusive high (best)
    description: str     # what the judge should assess
    owner: str           # archetype core name that specializes in this axis
    tolerance: int = 1   # |pred-true| beyond this counts as a "mistake" (reflection)


@dataclass(frozen=True)
class Rubric:
    axes: Tuple[Axis, ...]

    def names(self) -> List[str]:
        return [a.name for a in self.axes]

    def owner_of(self, archetype: str) -> List[str]:
        """Axis names an archetype is responsible for.

        The generalist "Core Judge" owns every axis; each specialist owns the axes
        whose ``owner`` matches it (falling back to all axes if it owns none, so a
        node always has something to reflect on).
        """
        if archetype == "Core Judge":
            return self.names()
        owned = [a.name for a in self.axes if a.owner == archetype]
        return owned or self.names()

    def contract(self) -> str:
        """The JSON output contract appended to every node's system prompt."""
        lines = [
            "Respond with ONLY a compact JSON object, no prose, no markdown, of shape:",
            '{"per_axis": {' + ", ".join(
                f'"{a.name}": {{"value": {a.lo}-{a.hi}, "confidence": 0.0-1.0}}'
                for a in self.axes) + "}}",
            "Rate each axis on its integer scale where higher is better:",
        ]
        for a in self.axes:
            lines.append(f"  - {a.name} ({a.lo}-{a.hi}): {a.description}")
        lines.append(
            "Judge the axis your role owns with high confidence; for the others give "
            "your best estimate at lower confidence. If upstream judge findings are "
            "provided, weigh them by their stated priority.")
        return "\n".join(lines)

    def midpoint(self, axis_name: str) -> int:
        for a in self.axes:
            if a.name == axis_name:
                return (a.lo + a.hi) // 2
        return 0

    def span(self, axis_name: str) -> int:
        for a in self.axes:
            if a.name == axis_name:
                return max(1, a.hi - a.lo)
        return 1

    def tolerance(self, axis_name: str) -> int:
        for a in self.axes:
            if a.name == axis_name:
                return a.tolerance
        return 1


# Three separable HelpSteer2 axes, each owned by a distinct specialist. The Core
# Judge (generalist) must rate all three at once; specialists each focus on one.
HELPSTEER_RUBRIC = Rubric(axes=(
    Axis("correctness", 0, 4,
         "factual accuracy and freedom from errors in the response", "Fact-Checker"),
    Axis("coherence", 0, 4,
         "logical structure, consistency, and clarity of the response", "Coherence Judge"),
    Axis("helpfulness", 0, 4,
         "how well the response addresses the user's intent and is useful", "Relevance Judge"),
))
