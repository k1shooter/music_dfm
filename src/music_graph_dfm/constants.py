"""Shared constants for FSNTG-v2 coordinates and labels."""

from __future__ import annotations

SPAN_CHANNELS = ["key", "harm", "meter", "section", "reg_center"]
NOTE_CHANNELS = ["active", "pitch_token", "velocity", "role"]
PLACEMENT_CHANNELS = ["host", "template"]

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

E_SS_NONE = 0

COORD_ORDER = [
    "span.key",
    "span.harm",
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
    "span.harm": "span",
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

CACHE_SCHEMA_VERSION = "fsntg_v2_pop909_v1"

DEFAULT_TICKS_PER_BEAT = 480
