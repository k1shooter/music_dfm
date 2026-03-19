#!/usr/bin/env python
"""Sample states from trained checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from music_graph_dfm.representation.pitch_codec import PitchTokenCodec
from music_graph_dfm.representation.rhythm_templates import RhythmTemplateVocab
from music_graph_dfm.training.runner import generate_samples_from_checkpoint
from music_graph_dfm.utils.io import load_json, write_jsonl
from music_graph_dfm.utils.midi import save_state_midi


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample from checkpoint")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data-root", type=str, default="data/cache/pop909_fsntg_v2")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=96)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-dir", type=str, default="artifacts/samples")
    parser.add_argument("--sampler-mode", type=str, default="dfm", choices=["dfm", "editflow"])
    parser.add_argument("--whole-song-mode", type=str, default=None, choices=["long_context", "stitching_baseline"])
    parser.add_argument("--whole-song-segments", type=int, default=4)
    parser.add_argument("--export-midi", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root).expanduser().resolve()

    states = generate_samples_from_checkpoint(
        checkpoint=Path(args.checkpoint).expanduser().resolve(),
        data_root=data_root,
        split=args.split,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        device=args.device,
        sampler_mode=args.sampler_mode,
        whole_song_mode=args.whole_song_mode,
        whole_song_segments=args.whole_song_segments,
    )
    write_jsonl(out_dir / "samples.jsonl", (s.to_dict() for s in states))

    if args.export_midi:
        rhythm = RhythmTemplateVocab.from_dict(load_json(data_root / "rhythm_templates.json"))
        pitch = PitchTokenCodec.from_dict(load_json(data_root / "pitch_codec.json"))
        midi_dir = out_dir / "midi"
        midi_dir.mkdir(parents=True, exist_ok=True)
        for i, state in enumerate(states):
            save_state_midi(state, rhythm, pitch, midi_dir / f"sample_{i:04d}.mid")

    print(json.dumps({"num_samples": len(states), "out_dir": str(out_dir)}, indent=2))


if __name__ == "__main__":
    main()
