"""Evaluation pipelines for checkpoint-generated or pre-generated samples."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from music_graph_dfm.data import FSNTGV2JSONDataset
from music_graph_dfm.evaluation.metrics import aggregate_metrics, evaluate_generated_state
from music_graph_dfm.representation.pitch_codec import PitchTokenCodec
from music_graph_dfm.representation.rhythm_templates import RhythmTemplateVocab
from music_graph_dfm.representation.state import FSNTGV2State
from music_graph_dfm.training.runner import generate_samples_from_checkpoint
from music_graph_dfm.utils.io import load_json, save_json, write_jsonl
from music_graph_dfm.utils.midi import save_state_midi


def _load_codecs(data_root: Path) -> tuple[RhythmTemplateVocab, PitchTokenCodec]:
    rhythm = RhythmTemplateVocab.from_dict(load_json(data_root / "rhythm_templates.json"))
    pitch = PitchTokenCodec.from_dict(load_json(data_root / "pitch_codec.json"))
    return rhythm, pitch


def evaluate_sample_directory(
    sample_dir: Path,
    data_root: Path,
    reference_split: str = "test",
    out_path: Path | None = None,
) -> Dict[str, object]:
    sample_path = sample_dir / "samples.jsonl"
    generated = [FSNTGV2State.from_dict(row) for row in load_jsonl_like(sample_path)]
    refs = FSNTGV2JSONDataset(data_root / f"{reference_split}.jsonl")
    rhythm, pitch = _load_codecs(data_root)

    n = min(len(generated), len(refs))
    rows = [evaluate_generated_state(generated[i], rhythm, pitch, refs[i]) for i in range(n)]
    report = {
        "mode": "sample_directory",
        "num_examples": n,
        "metrics": aggregate_metrics(rows),
    }
    if out_path is not None:
        save_json(out_path, report)
    return report


def _load_checkpoint_extra(checkpoint: Path) -> dict:
    try:
        import torch
    except Exception:
        return {}
    payload = torch.load(checkpoint, map_location="cpu")
    return dict(payload.get("extra", {}))


def evaluate_reference_split(data_root: Path, split: str = "test", out_path: Path | None = None) -> Dict[str, object]:
    ds = FSNTGV2JSONDataset(data_root / f"{split}.jsonl")
    rhythm, pitch = _load_codecs(data_root)
    rows = [evaluate_generated_state(state, rhythm, pitch, reference=state) for state in ds.states]
    report = {
        "mode": "reference_sanity",
        "num_examples": len(rows),
        "metrics": aggregate_metrics(rows),
    }
    if out_path is not None:
        save_json(out_path, report)
    return report


def generate_from_checkpoint(
    checkpoint: Path,
    data_root: Path,
    out_dir: Path,
    split: str = "test",
    num_samples: int = 16,
    num_steps: int = 96,
    device: str = "cpu",
    sampler_mode: str = "dfm",
    whole_song_mode: str | None = None,
    whole_song_segments: int = 4,
    export_midi: bool = False,
) -> List[FSNTGV2State]:
    samples = generate_samples_from_checkpoint(
        checkpoint=checkpoint,
        data_root=data_root,
        split=split,
        num_samples=num_samples,
        num_steps=num_steps,
        device=device,
        sampler_mode=sampler_mode,
        whole_song_mode=whole_song_mode,
        whole_song_segments=whole_song_segments,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "samples.jsonl", (s.to_dict() for s in samples))
    if export_midi:
        rhythm, pitch = _load_codecs(data_root)
        midi_dir = out_dir / "midi"
        midi_dir.mkdir(parents=True, exist_ok=True)
        for i, state in enumerate(samples):
            save_state_midi(state, rhythm, pitch, midi_dir / f"sample_{i:04d}.mid")
    return samples


def evaluate_checkpoint(
    checkpoint: Path,
    data_root: Path,
    split: str = "test",
    num_samples: int = 16,
    num_steps: int = 96,
    device: str = "cpu",
    sampler_mode: str = "dfm",
    whole_song_mode: str | None = None,
    whole_song_segments: int = 4,
    export_midi: bool = False,
    out_dir: Path | None = None,
    out_path: Path | None = None,
) -> Dict[str, object]:
    out_dir = out_dir or Path("artifacts/eval_samples")
    ckpt_extra = _load_checkpoint_extra(checkpoint)
    samples = generate_from_checkpoint(
        checkpoint=checkpoint,
        data_root=data_root,
        out_dir=out_dir,
        split=split,
        num_samples=num_samples,
        num_steps=num_steps,
        device=device,
        sampler_mode=sampler_mode,
        whole_song_mode=whole_song_mode,
        whole_song_segments=whole_song_segments,
        export_midi=export_midi,
    )

    refs = FSNTGV2JSONDataset(data_root / f"{split}.jsonl")
    rhythm, pitch = _load_codecs(data_root)

    n = min(len(samples), len(refs))
    rows = [evaluate_generated_state(samples[i], rhythm, pitch, refs[i]) for i in range(n)]
    report = {
        "mode": "checkpoint_generation",
        "num_examples": n,
        "metrics": aggregate_metrics(rows),
        "sample_dir": str(out_dir),
        "whole_song_mode": whole_song_mode or "segment",
        "checkpoint_meta": {
            "mode": ckpt_extra.get("mode", ""),
            "graph_kernel": ckpt_extra.get("graph_kernel", {}),
            "model_cfg": ckpt_extra.get("model_cfg", {}),
            "vocab_sizes": ckpt_extra.get("vocab_sizes", {}),
        },
    }
    if out_path is not None:
        save_json(out_path, report)
    return report


def load_jsonl_like(path: Path) -> list[dict]:
    rows: list[dict] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            import json

            rows.append(json.loads(line))
    return rows
