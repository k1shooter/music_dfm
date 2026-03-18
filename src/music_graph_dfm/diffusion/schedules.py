"""Structure-first path schedule."""

from __future__ import annotations

import math
from dataclasses import dataclass

from music_graph_dfm.constants import COORD_GROUPS


@dataclass
class StructureFirstSchedule:
    span_shift: float = 0.18
    span_relation_shift: float = 0.36
    placement_shift: float = 0.56
    note_shift: float = 0.74
    temperature: float = 0.12

    def _shift(self, coord: str) -> float:
        group = COORD_GROUPS[coord]
        if group == "span":
            return self.span_shift
        if group == "span_relation":
            return self.span_relation_shift
        if group == "placement":
            return self.placement_shift
        return self.note_shift

    def kappa(self, coord: str, t: float) -> float:
        z = (float(t) - self._shift(coord)) / max(1e-6, self.temperature)
        k = 1.0 / (1.0 + math.exp(-z))
        return max(1e-6, min(1.0 - 1e-6, k))

    def dkappa_dt(self, coord: str, t: float) -> float:
        k = self.kappa(coord, t)
        return (k * (1.0 - k)) / max(1e-6, self.temperature)

    def eta(self, coord: str, t: float) -> float:
        k = self.kappa(coord, t)
        return self.dkappa_dt(coord, t) / max(1e-6, 1.0 - k)
