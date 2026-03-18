"""Harmony-relative pitch token codec for FSNTG-v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class PitchToken:
    degree_wrt_harmony: int
    register_offset: int


class PitchTokenCodec:
    """Flattened categorical over (degree_wrt_harmony, register_offset)."""

    def __init__(
        self,
        degrees: Iterable[int] | None = None,
        register_offsets: Iterable[int] | None = None,
        pad_token: int = 0,
        center_base_midi: int = 48,
    ) -> None:
        self.degrees = list(degrees if degrees is not None else range(12))
        self.register_offsets = list(register_offsets if register_offsets is not None else range(-3, 4))
        self.pad_token = int(pad_token)
        self.center_base_midi = int(center_base_midi)

        self._token_to_pitch: Dict[int, PitchToken] = {self.pad_token: PitchToken(0, 0)}
        self._pitch_to_token: Dict[Tuple[int, int], int] = {}
        next_token = 1
        for degree in self.degrees:
            for reg in self.register_offsets:
                self._token_to_pitch[next_token] = PitchToken(int(degree), int(reg))
                self._pitch_to_token[(int(degree), int(reg))] = next_token
                next_token += 1

    @property
    def vocab_size(self) -> int:
        return len(self._token_to_pitch)

    def encode(self, degree_wrt_harmony: int, register_offset: int) -> int:
        return encode_pitch_token(self, degree_wrt_harmony, register_offset)

    def decode(self, token: int) -> PitchToken:
        return decode_pitch_token(self, token)

    def encode_from_absolute_pitch(self, pitch: int, harmonic_root: int, reg_center: int) -> int:
        pitch = int(pitch)
        harmonic_root = int(harmonic_root) % 12
        reg_center = int(reg_center)
        degree = (pitch - harmonic_root) % 12
        center = self.center_base_midi + reg_center * 4
        register_offset = int(round((pitch - center) / 12.0))
        register_offset = max(min(self.register_offsets), min(max(self.register_offsets), register_offset))
        return self.encode(degree, register_offset)

    def absolute_pitch(self, key: int, harmonic_root: int, reg_center: int, token: int) -> int:
        """Reconstruct absolute MIDI pitch using harmony-root-relative degree."""
        del key  # kept in signature for compatibility with previous API.
        pt = self.decode(token)
        if token == self.pad_token:
            return 60
        pc = (int(harmonic_root) + pt.degree_wrt_harmony) % 12
        center = self.center_base_midi + int(reg_center) * 4 + pt.register_offset * 12
        octave = max(0, min(10, center // 12))
        candidate = octave * 12 + pc
        while candidate < center - 6:
            candidate += 12
        while candidate > center + 6:
            candidate -= 12
        return int(max(0, min(127, candidate)))

    def is_compatible(self, key: int, harmonic_root: int, token: int) -> bool:
        if token == self.pad_token:
            return True
        pt = self.decode(token)
        absolute_pc = (int(harmonic_root) + pt.degree_wrt_harmony) % 12
        rel_to_key = (absolute_pc - int(key)) % 12
        key_scale = {0, 2, 4, 5, 7, 9, 11}
        harmony_consonance = {0, 3, 4, 7, 10}
        return pt.degree_wrt_harmony in harmony_consonance or rel_to_key in key_scale

    def compatibility_table(self, num_keys: int = 12, num_harm: int = 12) -> List[List[List[float]]]:
        table = [
            [[0.0 for _ in range(self.vocab_size)] for _ in range(num_harm)]
            for _ in range(num_keys)
        ]
        for key in range(num_keys):
            for harm in range(num_harm):
                for token in range(self.vocab_size):
                    table[key][harm][token] = 1.0 if self.is_compatible(key, harm, token) else 0.0
        return table

    def to_dict(self) -> dict:
        return {
            "degrees": self.degrees,
            "register_offsets": self.register_offsets,
            "pad_token": self.pad_token,
            "center_base_midi": self.center_base_midi,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "PitchTokenCodec":
        return cls(
            degrees=payload.get("degrees"),
            register_offsets=payload.get("register_offsets"),
            pad_token=int(payload.get("pad_token", 0)),
            center_base_midi=int(payload.get("center_base_midi", 48)),
        )


def encode_pitch_token(codec: PitchTokenCodec, degree_wrt_harmony: int, register_offset: int) -> int:
    key = (int(degree_wrt_harmony) % 12, int(register_offset))
    if key in codec._pitch_to_token:
        return codec._pitch_to_token[key]

    best_token = codec.pad_token
    best_dist = float("inf")
    target_degree = int(degree_wrt_harmony) % 12
    target_reg = int(register_offset)
    for (degree, reg), token in codec._pitch_to_token.items():
        circular = min((degree - target_degree) % 12, (target_degree - degree) % 12)
        dist = circular + abs(reg - target_reg)
        if dist < best_dist:
            best_dist = dist
            best_token = token
    return best_token


def decode_pitch_token(codec: PitchTokenCodec, token: int) -> PitchToken:
    return codec._token_to_pitch.get(int(token), codec._token_to_pitch[codec.pad_token])
