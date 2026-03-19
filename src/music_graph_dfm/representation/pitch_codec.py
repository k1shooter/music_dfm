"""Harmony-relative pitch token codec for FSNTG-v2.

Primary API (context-aware):
- encode_pitch_token(abs_pitch, host_span_state)
- decode_pitch_token(token, host_span_state)
- compatibility_table(host_span_state, token)
- nearest_token_projection(abs_pitch, host_span_state)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Tuple

MAJOR_SCALE_INTERVALS = {0, 2, 4, 5, 7, 9, 11}
CHORD_TONE_INTERVALS = {0, 3, 4, 7, 10}
TONAL_ROLE_CLASSES = ["chord_tone", "scale_tone", "chromatic"]


@dataclass(frozen=True)
class PitchToken:
    degree_wrt_harmony: int
    role_class: int
    register_offset: int


def _coerce_host_state(host_span_state: Mapping[str, int]) -> tuple[int, int, int]:
    key = int(host_span_state.get("key", 0)) % 12
    harmonic_root = int(host_span_state.get("harm", 0)) % 12
    reg_center = int(host_span_state.get("reg_center", 4))
    return key, harmonic_root, reg_center


def _circular_distance(a: int, b: int, mod: int) -> int:
    d = (int(a) - int(b)) % mod
    return min(d, mod - d)


def _closest_pc_to_center(pc: int, center: int) -> int:
    octave = max(0, min(10, int(center) // 12))
    candidate = octave * 12 + int(pc)
    while candidate < center - 6:
        candidate += 12
    while candidate > center + 6:
        candidate -= 12
    return int(max(0, min(127, candidate)))


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

    # Backward-compatible component API.
    def encode(self, degree_wrt_harmony: int, role_class: int, register_offset: int | None = None) -> int:
        if register_offset is None:
            # Legacy shorthand encode(degree, register_offset).
            register_offset = int(role_class)
            role_class = 0
        return self.encode_components(
            degree_wrt_harmony=int(degree_wrt_harmony),
            role_class=int(role_class),
            register_offset=int(register_offset),
        )

    def decode(self, token: int) -> PitchToken:
        return self.decode_components(token)

    def encode_components(self, degree_wrt_harmony: int, role_class: int, register_offset: int) -> int:
        key = (int(degree_wrt_harmony) % 12, int(role_class), int(register_offset))
        if key in self._pitch_to_token:
            return self._pitch_to_token[key]

        best_token = self.pad_token
        best_dist = float("inf")
        target_degree = int(degree_wrt_harmony) % 12
        target_role = int(role_class)
        target_reg = int(register_offset)

        for (degree, role, reg), token in self._pitch_to_token.items():
            circular = _circular_distance(degree, target_degree, mod=12)
            role_penalty = 0 if role == target_role else 2
            reg_penalty = abs(reg - target_reg)
            dist = circular + role_penalty + reg_penalty
            if dist < best_dist:
                best_dist = dist
                best_token = token
        return best_token

    def decode_components(self, token: int) -> PitchToken:
        return self._token_to_pitch.get(int(token), self._token_to_pitch[self.pad_token])

    # Primary context-aware API.
    def nearest_token_projection(
        self,
        abs_pitch: int,
        host_span_state: Mapping[str, int],
        role_class: int | None = None,
    ) -> int:
        key, harmonic_root, reg_center = _coerce_host_state(host_span_state)
        abs_pitch = int(max(0, min(127, int(abs_pitch))))
        abs_pc = abs_pitch % 12

        degree = (abs_pc - harmonic_root) % 12
        if role_class is None:
            role_class = infer_role_class(
                abs_pc=abs_pc,
                key=key,
                harmonic_root=harmonic_root,
                degree=degree,
            )

        center = self.center_base_midi + int(reg_center) * 4
        register_offset = int(round((abs_pitch - center) / 12.0))
        register_offset = max(min(self.register_offsets), min(max(self.register_offsets), register_offset))

        exact = (int(degree), int(role_class), int(register_offset))
        if exact in self._pitch_to_token:
            return self._pitch_to_token[exact]

        best_token = self.pad_token
        best_dist = float("inf")
        for token, pitch_token in self._token_to_pitch.items():
            if token == self.pad_token:
                continue
            candidate_pitch = self.absolute_pitch(
                key=key,
                harmonic_root=harmonic_root,
                reg_center=reg_center,
                token=token,
            )
            abs_dist = abs(candidate_pitch - abs_pitch)
            degree_dist = _circular_distance(pitch_token.degree_wrt_harmony, degree, mod=12)
            role_penalty = 0 if int(pitch_token.role_class) == int(role_class) else 3
            reg_penalty = abs(int(pitch_token.register_offset) - int(register_offset))
            dist = abs_dist + degree_dist + role_penalty + reg_penalty
            if dist < best_dist:
                best_dist = dist
                best_token = token
        return best_token

    def encode_pitch_token(self, abs_pitch: int, host_span_state: Mapping[str, int]) -> int:
        return self.nearest_token_projection(abs_pitch=abs_pitch, host_span_state=host_span_state)

    def encode_from_absolute_pitch(
        self,
        pitch: int,
        harmonic_root: int,
        key: int,
        reg_center: int,
    ) -> int:
        return self.encode_pitch_token(
            abs_pitch=int(pitch),
            host_span_state={"harm": int(harmonic_root), "key": int(key), "reg_center": int(reg_center)},
        )

    def absolute_pitch(self, key: int, harmonic_root: int, reg_center: int, token: int) -> int:
        """Reconstruct absolute MIDI pitch using harmony-root-relative pitch token."""
        key = int(key) % 12
        harmonic_root = int(harmonic_root) % 12
        reg_center = int(reg_center)

        if int(token) == self.pad_token:
            center = self.center_base_midi + reg_center * 4
            return _closest_pc_to_center(harmonic_root, center)

        pt = self.decode_components(token)
        snapped_degree = snap_degree_to_role(
            degree=pt.degree_wrt_harmony,
            role_class=pt.role_class,
            key=key,
            harmonic_root=harmonic_root,
        )
        pc = (harmonic_root + snapped_degree) % 12

        center = self.center_base_midi + reg_center * 4 + int(pt.register_offset) * 12
        return _closest_pc_to_center(pc, center)

    def decode_pitch_token(self, token: int, host_span_state: Mapping[str, int]) -> int:
        key, harmonic_root, reg_center = _coerce_host_state(host_span_state)
        return self.absolute_pitch(
            key=key,
            harmonic_root=harmonic_root,
            reg_center=reg_center,
            token=int(token),
        )

    def is_compatible(self, key: int, harmonic_root: int, token: int) -> bool:
        if int(token) == self.pad_token:
            return True
        pt = self.decode_components(token)
        snapped_degree = snap_degree_to_role(
            degree=pt.degree_wrt_harmony,
            role_class=pt.role_class,
            key=int(key),
            harmonic_root=int(harmonic_root),
        )
        abs_pc = (int(harmonic_root) + snapped_degree) % 12
        key_rel = (abs_pc - int(key)) % 12

        if int(pt.role_class) == 0:
            return snapped_degree in CHORD_TONE_INTERVALS
        if int(pt.role_class) == 1:
            return key_rel in MAJOR_SCALE_INTERVALS
        return True

    def compatibility_for_state(self, host_span_state: Mapping[str, int], token: int) -> float:
        key, harmonic_root, _ = _coerce_host_state(host_span_state)
        return 1.0 if self.is_compatible(key=key, harmonic_root=harmonic_root, token=int(token)) else 0.0

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
        dist = _circular_distance(cand, degree, mod=12)
        if dist < best_dist:
            best_dist = dist
            best = cand
    return int(best)


_DEFAULT_CODEC = PitchTokenCodec()


def encode_pitch_token(
    abs_pitch: int,
    host_span_state: Mapping[str, int],
    codec: PitchTokenCodec | None = None,
) -> int:
    """Primary API: encode absolute pitch under host span context.

    A codec argument is optional; when omitted, a default codec is used.
    """
    active_codec = codec or _DEFAULT_CODEC
    return active_codec.encode_pitch_token(abs_pitch=int(abs_pitch), host_span_state=host_span_state)


def decode_pitch_token(
    token: int,
    host_span_state: Mapping[str, int],
    codec: PitchTokenCodec | None = None,
) -> int:
    """Primary API: decode token to absolute MIDI pitch under host span context."""
    active_codec = codec or _DEFAULT_CODEC
    return active_codec.decode_pitch_token(token=int(token), host_span_state=host_span_state)


def compatibility_table(
    host_span_state: Mapping[str, int],
    token: int,
    codec: PitchTokenCodec | None = None,
) -> float:
    """Primary API: compatibility score for one token under a host span context."""
    active_codec = codec or _DEFAULT_CODEC
    return active_codec.compatibility_for_state(host_span_state=host_span_state, token=int(token))


def nearest_token_projection(
    abs_pitch: int,
    host_span_state: Mapping[str, int],
    codec: PitchTokenCodec | None = None,
    role_class: int | None = None,
) -> int:
    """Project an absolute pitch to the nearest valid token under host span context."""
    active_codec = codec or _DEFAULT_CODEC
    return active_codec.nearest_token_projection(
        abs_pitch=int(abs_pitch),
        host_span_state=host_span_state,
        role_class=role_class,
    )


# Backward-compatibility aliases for older call sites.
def encode_pitch_components(
    codec: PitchTokenCodec,
    degree_wrt_harmony: int,
    role_class: int,
    register_offset: int,
) -> int:
    return codec.encode_components(
        degree_wrt_harmony=degree_wrt_harmony,
        role_class=role_class,
        register_offset=register_offset,
    )


def decode_pitch_components(codec: PitchTokenCodec, token: int) -> PitchToken:
    return codec.decode_components(token)


def encode_pitch_token_from_state(codec: PitchTokenCodec, abs_pitch: int, host_span_state: Mapping[str, int]) -> int:
    return codec.encode_pitch_token(abs_pitch=int(abs_pitch), host_span_state=host_span_state)


def decode_pitch_token_to_abs(codec: PitchTokenCodec, token: int, host_span_state: Mapping[str, int]) -> int:
    return codec.decode_pitch_token(token=int(token), host_span_state=host_span_state)


def compatibility_table_for_state(codec: PitchTokenCodec, host_span_state: Mapping[str, int], token: int) -> float:
    return codec.compatibility_for_state(host_span_state=host_span_state, token=int(token))
