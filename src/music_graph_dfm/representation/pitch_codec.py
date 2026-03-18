"""Harmony-relative pitch token codec for FSNTG-v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

MAJOR_SCALE_INTERVALS = {0, 2, 4, 5, 7, 9, 11}
CHORD_TONE_INTERVALS = {0, 3, 4, 7, 10}
TONAL_ROLE_CLASSES = ["chord_tone", "scale_tone", "chromatic"]


@dataclass(frozen=True)
class PitchToken:
    degree_wrt_harmony: int
    role_class: int
    register_offset: int


class PitchTokenCodec:
    """Flattened categorical over (degree_wrt_harmony, role_class, register_offset)."""

    def __init__(
        self,
        degrees: Iterable[int] | None = None,
        role_classes: Iterable[int] | None = None,
        register_offsets: Iterable[int] | None = None,
        pad_token: int = 0,
        center_base_midi: int = 48,
    ) -> None:
        self.degrees = list(degrees if degrees is not None else range(12))
        self.role_classes = list(role_classes if role_classes is not None else range(len(TONAL_ROLE_CLASSES)))
        self.register_offsets = list(register_offsets if register_offsets is not None else range(-3, 4))
        self.pad_token = int(pad_token)
        self.center_base_midi = int(center_base_midi)

        self._token_to_pitch: Dict[int, PitchToken] = {self.pad_token: PitchToken(0, 0, 0)}
        self._pitch_to_token: Dict[Tuple[int, int, int], int] = {}
        next_token = 1
        for degree in self.degrees:
            for role in self.role_classes:
                for reg in self.register_offsets:
                    token = PitchToken(int(degree), int(role), int(reg))
                    self._token_to_pitch[next_token] = token
                    self._pitch_to_token[(int(degree), int(role), int(reg))] = next_token
                    next_token += 1

    @property
    def vocab_size(self) -> int:
        return len(self._token_to_pitch)

    def encode(self, degree_wrt_harmony: int, role_class: int, register_offset: int | None = None) -> int:
        if register_offset is None:
            # Backward-compatible shorthand encode(degree, register_offset).
            register_offset = int(role_class)
            role_class = 0
        return encode_pitch_token(self, degree_wrt_harmony, role_class, register_offset)

    def decode(self, token: int) -> PitchToken:
        return decode_pitch_token(self, token)

    def encode_from_absolute_pitch(
        self,
        pitch: int,
        harmonic_root: int,
        key: int,
        reg_center: int,
    ) -> int:
        pitch = int(pitch)
        harmonic_root = int(harmonic_root) % 12
        key = int(key) % 12
        reg_center = int(reg_center)

        abs_pc = pitch % 12
        degree = (abs_pc - harmonic_root) % 12
        role = infer_role_class(abs_pc=abs_pc, key=key, harmonic_root=harmonic_root, degree=degree)

        center = self.center_base_midi + reg_center * 4
        register_offset = int(round((pitch - center) / 12.0))
        register_offset = max(min(self.register_offsets), min(max(self.register_offsets), register_offset))
        return self.encode(degree, role, register_offset)

    def absolute_pitch(self, key: int, harmonic_root: int, reg_center: int, token: int) -> int:
        """Reconstruct absolute MIDI pitch using harmony-root-relative pitch token."""
        if token == self.pad_token:
            return 60

        pt = self.decode(token)
        snapped_degree = snap_degree_to_role(
            degree=pt.degree_wrt_harmony,
            role_class=pt.role_class,
            key=int(key),
            harmonic_root=int(harmonic_root),
        )
        pc = (int(harmonic_root) + snapped_degree) % 12

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
        degree = int(pt.degree_wrt_harmony) % 12
        abs_pc = (int(harmonic_root) + degree) % 12
        key_rel = (abs_pc - int(key)) % 12

        if pt.role_class == 0:
            return degree in CHORD_TONE_INTERVALS
        if pt.role_class == 1:
            return key_rel in MAJOR_SCALE_INTERVALS
        return True

    def compatibility_for_state(self, host_span_state: dict, token: int) -> float:
        key = int(host_span_state.get("key", 0))
        harm = int(host_span_state.get("harm", 0))
        return 1.0 if self.is_compatible(key, harm, token) else 0.0

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
            "role_classes": self.role_classes,
            "register_offsets": self.register_offsets,
            "pad_token": self.pad_token,
            "center_base_midi": self.center_base_midi,
            "role_labels": TONAL_ROLE_CLASSES,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "PitchTokenCodec":
        return cls(
            degrees=payload.get("degrees"),
            role_classes=payload.get("role_classes"),
            register_offsets=payload.get("register_offsets"),
            pad_token=int(payload.get("pad_token", 0)),
            center_base_midi=int(payload.get("center_base_midi", 48)),
        )


def infer_role_class(abs_pc: int, key: int, harmonic_root: int, degree: int) -> int:
    del harmonic_root
    if int(degree) % 12 in CHORD_TONE_INTERVALS:
        return 0
    if (int(abs_pc) - int(key)) % 12 in MAJOR_SCALE_INTERVALS:
        return 1
    return 2


def snap_degree_to_role(degree: int, role_class: int, key: int, harmonic_root: int) -> int:
    degree = int(degree) % 12
    role_class = int(role_class)
    key = int(key) % 12
    harmonic_root = int(harmonic_root) % 12

    if role_class == 0 and degree in CHORD_TONE_INTERVALS:
        return degree
    if role_class == 1:
        abs_pc = (harmonic_root + degree) % 12
        if (abs_pc - key) % 12 in MAJOR_SCALE_INTERVALS:
            return degree
    if role_class == 2:
        return degree

    if role_class == 0:
        candidates = list(CHORD_TONE_INTERVALS)
    elif role_class == 1:
        candidates = [d for d in range(12) if ((harmonic_root + d) - key) % 12 in MAJOR_SCALE_INTERVALS]
    else:
        candidates = list(range(12))

    best = candidates[0]
    best_dist = float("inf")
    for cand in candidates:
        dist = min((cand - degree) % 12, (degree - cand) % 12)
        if dist < best_dist:
            best_dist = dist
            best = cand
    return int(best)


def encode_pitch_token(
    codec: PitchTokenCodec,
    degree_wrt_harmony: int,
    role_class: int,
    register_offset: int,
) -> int:
    key = (int(degree_wrt_harmony) % 12, int(role_class), int(register_offset))
    if key in codec._pitch_to_token:
        return codec._pitch_to_token[key]

    best_token = codec.pad_token
    best_dist = float("inf")
    target_degree = int(degree_wrt_harmony) % 12
    target_role = int(role_class)
    target_reg = int(register_offset)

    for (degree, role, reg), token in codec._pitch_to_token.items():
        circular = min((degree - target_degree) % 12, (target_degree - degree) % 12)
        role_penalty = 0 if role == target_role else 1
        reg_penalty = abs(reg - target_reg)
        dist = circular + role_penalty + reg_penalty
        if dist < best_dist:
            best_dist = dist
            best_token = token
    return best_token


def decode_pitch_token(codec: PitchTokenCodec, token: int) -> PitchToken:
    return codec._token_to_pitch.get(int(token), codec._token_to_pitch[codec.pad_token])
