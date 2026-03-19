#!/usr/bin/env python
"""End-to-end evaluation entrypoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from music_graph_dfm.evaluation.pipeline import (
    evaluate_checkpoint,
    evaluate_reference_split,
    evaluate_sample_directory,
)
from music_graph_dfm.utils.io import save_json


def _evaluate_checkpoint_mode(args: argparse.Namespace) -> dict:
    if args.whole_song_compare:
        long_context = evaluate_checkpoint(
            checkpoint=Path(args.checkpoint).expanduser().resolve(),
            data_root=Path(args.data_root).expanduser().resolve(),
            split=args.split,
            num_samples=args.num_samples,
            num_steps=args.num_steps,
            device=args.device,
            sampler_mode=args.sampler_mode,
            whole_song_mode="long_context",
            whole_song_segments=args.whole_song_segments,
            export_midi=args.export_midi,
            out_dir=Path(args.sample_out_dir).expanduser().resolve() / "long_context",
            out_path=None,
        )
        stitching = evaluate_checkpoint(
            checkpoint=Path(args.checkpoint).expanduser().resolve(),
            data_root=Path(args.data_root).expanduser().resolve(),
            split=args.split,
            num_samples=args.num_samples,
            num_steps=args.num_steps,
            device=args.device,
            sampler_mode=args.sampler_mode,
            whole_song_mode="stitching_baseline",
            whole_song_segments=args.whole_song_segments,
            export_midi=args.export_midi,
            out_dir=Path(args.sample_out_dir).expanduser().resolve() / "stitching_baseline",
            out_path=None,
        )
        return {
            "mode": "checkpoint_whole_song_compare",
            "long_context": long_context,
            "stitching_baseline": stitching,
        }

    return evaluate_checkpoint(
        checkpoint=Path(args.checkpoint).expanduser().resolve(),
        data_root=Path(args.data_root).expanduser().resolve(),
        split=args.split,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        device=args.device,
        sampler_mode=args.sampler_mode,
        whole_song_mode=args.whole_song_mode,
        whole_song_segments=args.whole_song_segments,
        export_midi=args.export_midi,
        out_dir=Path(args.sample_out_dir).expanduser().resolve(),
        out_path=None,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generated samples")
    parser.add_argument("--eval-mode", type=str, choices=["checkpoint", "sample-dir", "reference"], default="checkpoint")
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--sample-dir", type=str, default="artifacts/samples")
    parser.add_argument("--sample-out-dir", type=str, default="artifacts/eval_samples")
    parser.add_argument("--data-root", type=str, default="data/cache/pop909_fsntg_v2")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--num-steps", type=int, default=96)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--sampler-mode", type=str, default="dfm", choices=["dfm", "editflow"])
    parser.add_argument("--whole-song-mode", type=str, default=None, choices=["long_context", "stitching_baseline"])
    parser.add_argument("--whole-song-segments", type=int, default=4)
    parser.add_argument("--whole-song-compare", action="store_true")
    parser.add_argument("--export-midi", action="store_true")
    parser.add_argument("--out", type=str, default="artifacts/eval_report.json")
    args = parser.parse_args()

    if args.eval_mode == "checkpoint":
        report = _evaluate_checkpoint_mode(args)
    elif args.eval_mode == "sample-dir":
        report = evaluate_sample_directory(
            sample_dir=Path(args.sample_dir).expanduser().resolve(),
            data_root=Path(args.data_root).expanduser().resolve(),
            reference_split=args.split,
            out_path=None,
        )
    else:
        report = evaluate_reference_split(
            data_root=Path(args.data_root).expanduser().resolve(),
            split=args.split,
            out_path=None,
        )

    out_path = Path(args.out).expanduser().resolve()
    save_json(out_path, report)
    summary = {"saved": str(out_path), "mode": args.eval_mode}
    if isinstance(report, dict):
        summary["experimental"] = bool(report.get("experimental", False))
        ckpt_meta = report.get("checkpoint_meta", {})
        if isinstance(ckpt_meta, dict):
            summary["graph_kernel_is_approximate"] = bool(ckpt_meta.get("graph_kernel_is_approximate", False))
            summary["graph_kernel_target_rate_mode"] = ckpt_meta.get("graph_kernel_target_rate_mode", "")
            summary["editflow_mode"] = ckpt_meta.get("editflow_mode", "")
            summary["editflow_objective"] = ckpt_meta.get("editflow_objective", "")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
