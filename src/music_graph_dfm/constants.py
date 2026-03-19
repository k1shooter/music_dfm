"""Shared constants for FSNTG-v2 coordinates and labels."""

from __future__ import annotations

SPAN_CHANNELS = [
    "key",
    "harm_root",
    "harm_quality",
    "harm_function",
    "meter",
    "section",
    "reg_center",
]
NOTE_CHANNELS = ["active", "pitch_token", "velocity", "role"]
PLACEMENT_CHANNELS = ["host", "template"]

HARM_QUALITY_LABELS = [
    "unknown",
    "major",
    "minor",
    "dominant",
    "diminished",
    "augmented",
    "suspended",
]

HARM_FUNCTION_LABELS = [
    "unknown",
    "tonic",
    "predominant",
    "dominant",
]

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

GRAPH_KERNEL_APPROX_COORDS = [
    "span.harm_root",
    "note.pitch_token",
]

E_SS_NONE = 0

COORD_ORDER = [
    "span.key",
    "span.harm_root",
    "span.harm_quality",
    "span.harm_function",
    "span.meter",
    "span.section",
    "span.reg_center",
    "note.active",
    "note.pitch_token",
    "note.velocity",
    "note.role",
    "note.host",
    "note.template",
    "e_ss.relation",
]

COORD_GROUPS = {
    "span.key": "span",
    "span.harm_root": "span",
    "span.harm_quality": "span",
    "span.harm_function": "span",
    "span.meter": "span",
    "span.section": "span",
    "span.reg_center": "span",
    "e_ss.relation": "span_relation",
    "note.host": "placement",
    "note.template": "placement",
    "note.active": "note",
    "note.pitch_token": "note",
    "note.velocity": "note",
    "note.role": "note",
}

CACHE_SCHEMA_VERSION = "fsntg_v2_pop909_v3"

DEFAULT_TICKS_PER_BEAT = 480
