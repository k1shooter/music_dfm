"""Constants and channel definitions for FSNTG."""

from __future__ import annotations

SPAN_CHANNELS = ["key", "harm", "meter", "section", "reg_center"]
NOTE_CHANNELS = ["active", "pitch_token", "velocity", "role"]
EDGE_NS_CHANNELS = ["template"]
EDGE_SS_CHANNELS = ["relation"]

# 0 is always reserved for no-edge.
E_NS_NONE = 0
E_SS_NONE = 0

SPAN_RELATIONS = [
    "none",
    "next",
    "repeat",
    "variation",
    "contrast",
    "modulation",
]

AUX_NOTE_RELATIONS = [
    "none",
    "same_onset",
    "overlap",
    "sequential_same_role",
]

DEFAULT_TICKS_PER_BEAT = 480
DEFAULT_BEATS_PER_BAR = 4
DEFAULT_TICKS_PER_BAR = DEFAULT_TICKS_PER_BEAT * DEFAULT_BEATS_PER_BAR
