"""Rhythmic template vocabulary with explicit tie/extension semantics."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple


@dataclass(frozen=True)
class RhythmTemplate:
    meter: int
    onset_bin: int
    duration_class: int
    tie_flag: int
    extension_class: int


class RhythmTemplateVocab:
    """Template vocabulary indexed over meter-conditioned rhythmic patterns."""

    def __init__(
        self,
        top_k_per_meter: int = 64,
        onset_bins: int = 8,
        duration_ticks: List[int] | None = None,
        max_extension_class: int = 4,
        tie_extension_fraction: float = 1.0,
    ) -> None:
        self.top_k_per_meter = int(top_k_per_meter)
        self.onset_bins = int(onset_bins)
        self.duration_ticks = list(duration_ticks if duration_ticks is not None else [60, 120, 180, 240, 360, 480, 720, 960])
        self.max_extension_class = int(max_extension_class)
        self.tie_extension_fraction = float(tie_extension_fraction)

        self.templates: List[RhythmTemplate] = [RhythmTemplate(-1, 0, 0, 0, 0)]
        self.template_to_id: Dict[RhythmTemplate, int] = {self.templates[0]: 0}
        self.by_meter: Dict[int, List[int]] = defaultdict(list)
        self.template_frequency: Dict[int, int] = defaultdict(int)

    @property
    def vocab_size(self) -> int:
        return len(self.templates)

    def fit(self, observed: Iterable[Tuple[int, int, int, int, int]]) -> None:
        counts: Dict[int, Counter[RhythmTemplate]] = defaultdict(Counter)
        for meter, onset, duration_class, tie, ext in observed:
            ext = max(0, min(self.max_extension_class, int(ext)))
            tpl = RhythmTemplate(int(meter), int(onset), int(duration_class), int(tie), ext)
            counts[int(meter)][tpl] += 1

        for meter, counter in counts.items():
            for tpl, freq in counter.most_common(self.top_k_per_meter):
                idx = self._register(tpl)
                self.by_meter[meter].append(idx)
                self.template_frequency[idx] += int(freq)

        if self.vocab_size == 1:
            idx = self._register(RhythmTemplate(4, 0, 3, 0, 0))
            self.by_meter[4].append(idx)
            self.template_frequency[idx] += 1

    def _register(self, tpl: RhythmTemplate) -> int:
        if tpl in self.template_to_id:
            return self.template_to_id[tpl]
        idx = len(self.templates)
        self.templates.append(tpl)
        self.template_to_id[tpl] = idx
        return idx

    def encode(
        self,
        meter: int,
        onset_bin: int,
        duration_class: int,
        tie_flag: int,
        extension_class: int,
    ) -> int:
        extension_class = max(0, min(self.max_extension_class, int(extension_class)))
        query = RhythmTemplate(
            meter=int(meter),
            onset_bin=int(onset_bin),
            duration_class=int(duration_class),
            tie_flag=int(tie_flag),
            extension_class=extension_class,
        )
        if query in self.template_to_id:
            return self.template_to_id[query]
        return self._nearest(query)

    def _nearest(self, query: RhythmTemplate) -> int:
        candidates = self.by_meter.get(query.meter)
        if not candidates:
            candidates = list(range(1, self.vocab_size))
        if not candidates:
            return 0
        best = candidates[0]
        best_dist = float("inf")
        for idx in candidates:
            tpl = self.templates[idx]
            dist = abs(tpl.onset_bin - query.onset_bin)
            dist += abs(tpl.duration_class - query.duration_class)
            dist += 2 * abs(tpl.tie_flag - query.tie_flag)
            dist += abs(tpl.extension_class - query.extension_class)
            if dist < best_dist:
                best_dist = dist
                best = idx
        return best

    def decode(self, template_id: int) -> RhythmTemplate:
        if 0 <= int(template_id) < len(self.templates):
            return self.templates[int(template_id)]
        return self.templates[0]

    def onset_ticks(self, template_id: int, ticks_per_span: int) -> int:
        tpl = self.decode(template_id)
        if self.onset_bins <= 1:
            return 0
        return int(round((tpl.onset_bin / (self.onset_bins - 1)) * ticks_per_span))

    def duration_ticks_with_semantics(self, template_id: int, ticks_per_span: int) -> int:
        tpl = self.decode(template_id)
        base = self.duration_ticks[max(0, min(len(self.duration_ticks) - 1, tpl.duration_class))]
        tie_bonus = int(round(ticks_per_span * self.tie_extension_fraction)) if tpl.tie_flag else 0
        extension = int(ticks_per_span) * int(tpl.extension_class)
        return int(max(1, base + tie_bonus + extension))

    def stats(self) -> dict:
        by_meter = {str(meter): len(ids) for meter, ids in self.by_meter.items()}
        return {
            "vocab_size": self.vocab_size,
            "top_k_per_meter": self.top_k_per_meter,
            "onset_bins": self.onset_bins,
            "max_extension_class": self.max_extension_class,
            "meters": by_meter,
        }

    def to_dict(self) -> dict:
        return {
            "top_k_per_meter": self.top_k_per_meter,
            "onset_bins": self.onset_bins,
            "duration_ticks": self.duration_ticks,
            "max_extension_class": self.max_extension_class,
            "tie_extension_fraction": self.tie_extension_fraction,
            "templates": [
                {
                    "meter": t.meter,
                    "onset_bin": t.onset_bin,
                    "duration_class": t.duration_class,
                    "tie_flag": t.tie_flag,
                    "extension_class": t.extension_class,
                    "count": int(self.template_frequency.get(i, 0)),
                }
                for i, t in enumerate(self.templates)
            ],
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "RhythmTemplateVocab":
        obj = cls(
            top_k_per_meter=int(payload.get("top_k_per_meter", 64)),
            onset_bins=int(payload.get("onset_bins", 8)),
            duration_ticks=list(payload.get("duration_ticks", [60, 120, 180, 240, 360, 480, 720, 960])),
            max_extension_class=int(payload.get("max_extension_class", 4)),
            tie_extension_fraction=float(payload.get("tie_extension_fraction", 1.0)),
        )
        obj.templates = []
        obj.template_to_id = {}
        obj.by_meter = defaultdict(list)
        obj.template_frequency = defaultdict(int)

        for idx, item in enumerate(payload.get("templates", [])):
            tpl = RhythmTemplate(
                meter=int(item.get("meter", -1)),
                onset_bin=int(item.get("onset_bin", 0)),
                duration_class=int(item.get("duration_class", 0)),
                tie_flag=int(item.get("tie_flag", 0)),
                extension_class=int(item.get("extension_class", 0)),
            )
            obj.templates.append(tpl)
            obj.template_to_id[tpl] = idx
            obj.template_frequency[idx] = int(item.get("count", 0))
            if idx > 0:
                obj.by_meter[tpl.meter].append(idx)

        if not obj.templates:
            obj.templates = [RhythmTemplate(-1, 0, 0, 0, 0)]
            obj.template_to_id = {obj.templates[0]: 0}
        return obj


def quantize_onset_bin(local_tick: int, ticks_per_span: int, onset_bins: int) -> int:
    if onset_bins <= 1 or ticks_per_span <= 0:
        return 0
    frac = max(0.0, min(1.0, float(local_tick) / float(ticks_per_span)))
    return int(round(frac * (onset_bins - 1)))


def quantize_duration_class(duration_tick: int, duration_ticks: List[int]) -> int:
    best_idx = 0
    best_dist = float("inf")
    for idx, value in enumerate(duration_ticks):
        dist = abs(int(duration_tick) - int(value))
        if dist < best_dist:
            best_dist = dist
            best_idx = idx
    return best_idx
