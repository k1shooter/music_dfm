"""Rhythmic placement template vocabulary for note-span edges."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class RhythmTemplate:
    meter: int
    onset_bin: int
    duration_class: int
    tie_flag: int = 0
    extension_class: int = 0


class RhythmTemplateVocab:
    """Compact, data-driven template vocabulary with per-meter top-K selection."""

    def __init__(
        self,
        top_k_per_meter: int = 32,
        max_onset_bins: int = 32,
        duration_tick_values: List[int] | None = None,
    ) -> None:
        self.top_k_per_meter = top_k_per_meter
        self.max_onset_bins = max_onset_bins
        self.duration_tick_values = duration_tick_values or [120, 240, 360, 480, 720, 960, 1440, 1920]
        self.templates: List[RhythmTemplate] = [RhythmTemplate(-1, 0, 0, 0, 0)]  # idx 0 reserved for no-edge
        self.template_to_id: Dict[RhythmTemplate, int] = {self.templates[0]: 0}
        self.by_meter: Dict[int, List[int]] = defaultdict(list)

    @property
    def vocab_size(self) -> int:
        return len(self.templates)

    def fit(self, observed_templates: Iterable[Tuple[int, int, int, int, int]]) -> None:
        counts_by_meter: Dict[int, Counter[RhythmTemplate]] = defaultdict(Counter)
        for meter, onset, dur, tie, ext in observed_templates:
            tpl = RhythmTemplate(int(meter), int(onset), int(dur), int(tie), int(ext))
            counts_by_meter[tpl.meter][tpl] += 1

        for meter, counter in counts_by_meter.items():
            for tpl, _freq in counter.most_common(self.top_k_per_meter):
                self._register(tpl)
                self.by_meter[meter].append(self.template_to_id[tpl])

        if len(self.templates) == 1:
            # fallback if dataset is empty
            self._register(RhythmTemplate(0, 0, 3, 0, 0))
            self.by_meter[0].append(1)

    def _register(self, tpl: RhythmTemplate) -> int:
        if tpl in self.template_to_id:
            return self.template_to_id[tpl]
        idx = len(self.templates)
        self.templates.append(tpl)
        self.template_to_id[tpl] = idx
        return idx

    def _nearest_template_id(self, tpl: RhythmTemplate) -> int:
        candidates = self.by_meter.get(tpl.meter)
        if not candidates:
            candidates = list(range(1, len(self.templates)))
        if not candidates:
            return 0
        best = candidates[0]
        best_dist = float("inf")
        for idx in candidates:
            cand = self.templates[idx]
            dist = (
                abs(cand.onset_bin - tpl.onset_bin)
                + abs(cand.duration_class - tpl.duration_class)
                + (2 if cand.tie_flag != tpl.tie_flag else 0)
                + (1 if cand.extension_class != tpl.extension_class else 0)
            )
            if dist < best_dist:
                best_dist = dist
                best = idx
        return best

    def encode(self, meter: int, onset_bin: int, duration_class: int, tie_flag: int = 0, extension_class: int = 0) -> int:
        tpl = RhythmTemplate(int(meter), int(onset_bin), int(duration_class), int(tie_flag), int(extension_class))
        if tpl in self.template_to_id:
            return self.template_to_id[tpl]
        return self._nearest_template_id(tpl)

    def decode(self, template_id: int) -> RhythmTemplate:
        if template_id < 0 or template_id >= len(self.templates):
            return self.templates[0]
        return self.templates[template_id]

    def onset_to_ticks(self, template_id: int, ticks_per_span: int, onset_bins: int | None = None) -> int:
        tpl = self.decode(template_id)
        bins = onset_bins or self.max_onset_bins
        return int(round((tpl.onset_bin / max(1, bins - 1)) * ticks_per_span))

    def duration_to_ticks(self, template_id: int) -> int:
        tpl = self.decode(template_id)
        idx = max(0, min(len(self.duration_tick_values) - 1, tpl.duration_class))
        return self.duration_tick_values[idx]

    def stats(self) -> dict:
        return {
            "vocab_size": len(self.templates),
            "meters": {str(k): len(v) for k, v in self.by_meter.items()},
            "top_k_per_meter": self.top_k_per_meter,
        }

    def to_dict(self) -> dict:
        return {
            "top_k_per_meter": self.top_k_per_meter,
            "max_onset_bins": self.max_onset_bins,
            "duration_tick_values": self.duration_tick_values,
            "templates": [
                {
                    "meter": t.meter,
                    "onset_bin": t.onset_bin,
                    "duration_class": t.duration_class,
                    "tie_flag": t.tie_flag,
                    "extension_class": t.extension_class,
                }
                for t in self.templates
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "RhythmTemplateVocab":
        obj = cls(
            top_k_per_meter=int(payload.get("top_k_per_meter", 32)),
            max_onset_bins=int(payload.get("max_onset_bins", 32)),
            duration_tick_values=list(payload.get("duration_tick_values", [120, 240, 360, 480, 720, 960, 1440, 1920])),
        )
        obj.templates = []
        obj.template_to_id = {}
        obj.by_meter = defaultdict(list)
        for idx, t in enumerate(payload.get("templates", [])):
            tpl = RhythmTemplate(
                int(t.get("meter", -1)),
                int(t.get("onset_bin", 0)),
                int(t.get("duration_class", 0)),
                int(t.get("tie_flag", 0)),
                int(t.get("extension_class", 0)),
            )
            obj.templates.append(tpl)
            obj.template_to_id[tpl] = idx
            if idx > 0:
                obj.by_meter[tpl.meter].append(idx)
        if not obj.templates:
            obj.templates = [RhythmTemplate(-1, 0, 0, 0, 0)]
            obj.template_to_id = {obj.templates[0]: 0}
        return obj


def quantize_onset_to_bin(local_tick: int, ticks_per_span: int, onset_bins: int) -> int:
    if ticks_per_span <= 0:
        return 0
    frac = max(0.0, min(1.0, local_tick / ticks_per_span))
    return int(round(frac * (onset_bins - 1)))


def quantize_duration_to_class(duration_tick: int, duration_tick_values: List[int]) -> int:
    best_idx = 0
    best_dist = float("inf")
    for i, v in enumerate(duration_tick_values):
        dist = abs(int(duration_tick) - int(v))
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx
