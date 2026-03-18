"""Harmony-relative pitch token codec for FSNTG note attributes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


HARM_ROOT_OFFSETS = [0, 2, 4, 5, 7, 9, 11]
HARM_CHORD_DEGREES = {
    0: {0, 4, 7},
    1: {0, 3, 7},
    2: {0, 3, 6},
    3: {0, 4, 8},
    4: {0, 5, 7},
    5: {0, 4, 7, 10},
    6: {0, 3, 7, 10},
}


@dataclass(frozen=True)
class PitchToken:
    degree: int
    register_offset: int


class PitchTokenCodec:
    """Encodes/decodes pitch_token as a compact categorical over (degree, register_offset)."""

    def __init__(
        self,
        degrees: Iterable[int] | None = None,
        register_offsets: Iterable[int] | None = None,
        pad_token: int = 0,
    ) -> None:
        self.degrees = list(degrees if degrees is not None else range(12))
        self.register_offsets = list(register_offsets if register_offsets is not None else range(-2, 3))
        self.pad_token = pad_token
        self._token_to_pitch: Dict[int, PitchToken] = {self.pad_token: PitchToken(0, 0)}
        self._pitch_to_token: Dict[Tuple[int, int], int] = {}
        next_token = 1
        for d in self.degrees:
            for r in self.register_offsets:
                self._token_to_pitch[next_token] = PitchToken(d, r)
                self._pitch_to_token[(d, r)] = next_token
                next_token += 1

    @property
    def vocab_size(self) -> int:
        return len(self._token_to_pitch)

    def encode(self, degree: int, register_offset: int) -> int:
        if (degree, register_offset) in self._pitch_to_token:
            return self._pitch_to_token[(degree, register_offset)]
        # nearest fallback
        best_token = self.pad_token
        best_dist = float("inf")
        for (d, r), tok in self._pitch_to_token.items():
            dist = abs(d - degree) + abs(r - register_offset)
            if dist < best_dist:
                best_dist = dist
                best_token = tok
        return best_token

    def decode(self, token: int) -> PitchToken:
        return self._token_to_pitch.get(token, self._token_to_pitch[self.pad_token])

    def is_compatible(self, key: int, harm: int, token: int) -> bool:
        """Loose compatibility: token degree mapped into harmonic chord tones."""
        if token == self.pad_token:
            return True
        pt = self.decode(token)
        root_shift = HARM_ROOT_OFFSETS[harm % len(HARM_ROOT_OFFSETS)]
        chord = HARM_CHORD_DEGREES[harm % len(HARM_CHORD_DEGREES)]
        rel_to_harm_root = (pt.degree - root_shift) % 12
        return rel_to_harm_root in chord

    def absolute_pitch(self, key: int, harm: int, reg_center: int, token: int) -> int:
        """Decode MIDI pitch from harmony-relative token and span context."""
        pt = self.decode(token)
        if token == self.pad_token:
            return 60
        pitch_class = (key + pt.degree) % 12
        center = 36 + int(reg_center) * 6
        target = center + int(pt.register_offset) * 12
        base_oct = max(0, min(10, target // 12))
        candidate = base_oct * 12 + pitch_class
        while candidate < target - 6:
            candidate += 12
        while candidate > target + 6:
            candidate -= 12
        return int(max(0, min(127, candidate)))

    def to_dict(self) -> dict:
        return {
            "degrees": self.degrees,
            "register_offsets": self.register_offsets,
            "pad_token": self.pad_token,
        }

    def compatibility_table(self, num_keys: int = 12, num_harm: int = 7):
        """Returns [num_keys, num_harm, vocab_size] compatibility tensor/list."""
        table = [
            [
                [0.0 for _ in range(self.vocab_size)]
                for _ in range(num_harm)
            ]
            for _ in range(num_keys)
        ]
        for key in range(num_keys):
            for harm in range(num_harm):
                for token in range(self.vocab_size):
                    table[key][harm][token] = 1.0 if self.is_compatible(key, harm, token) else 0.0
        return table

    @classmethod
    def from_dict(cls, payload: dict) -> "PitchTokenCodec":
        return cls(
            degrees=payload.get("degrees"),
            register_offsets=payload.get("register_offsets"),
            pad_token=payload.get("pad_token", 0),
        )
