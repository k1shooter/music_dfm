"""Structure-first scheduler and kappa utilities for coordinate-wise discrete paths."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class StructureFirstSchedule:
    span_shift: float = 0.20
    e_ss_shift: float = 0.32
    e_ns_shift: float = 0.48
    note_shift: float = 0.64
    temperature: float = 0.12

    def _shift_for_coord(self, coord: str) -> float:
        if coord.startswith("span."):
            return self.span_shift
        if coord.startswith("e_ss."):
            return self.e_ss_shift
        if coord.startswith("e_ns."):
            return self.e_ns_shift
        return self.note_shift

    def kappa(self, coord: str, t: float) -> float:
        shift = self._shift_for_coord(coord)
        z = (t - shift) / max(1e-6, self.temperature)
        val = 1.0 / (1.0 + math.exp(-z))
        return max(1e-5, min(1.0 - 1e-5, val))

    def dkappa_dt(self, coord: str, t: float) -> float:
        k = self.kappa(coord, t)
        return (k * (1.0 - k)) / max(1e-6, self.temperature)

    def eta(self, coord: str, t: float) -> float:
        k = self.kappa(coord, t)
        dk = self.dkappa_dt(coord, t)
        return dk / max(1e-6, 1.0 - k)


def linear_kappa(t: float) -> float:
    return max(1e-5, min(1.0 - 1e-5, t))


def linear_eta(t: float) -> float:
    return 1.0 / max(1e-6, 1.0 - linear_kappa(t))
