"""Console entry point for music-graph-dfm."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from music_graph_dfm.evaluation.pipeline import (
    evaluate_checkpoint,
    evaluate_reference_split,
    evaluate_sample_directory,
)
from music_graph_dfm.preprocess import PreprocessConfig, preprocess_pop909
from music_graph_dfm.training.runner import generate_samples_from_checkpoint, run_training
from music_graph_dfm.utils.io import save_json, write_jsonl
from music_graph_dfm.utils.midi import save_state_midi


def _load_yaml(path: Path) -> dict:
    import yaml

    return yaml.safe_load(path.read_text(encoding="utf-8"))


def cmd_download_pop909(args: argparse.Namespace) -> None:
    target = Path(args.target_dir).expanduser().resolve()
    repo = "https://github.com/music-x-lab/POP909-Dataset"

    if target.exists() and any(target.iterdir()):
        if not args.force:
            print({"status": "exists", "path": str(target)})
            return
        shutil.rmtree(target)

    target.parent.mkdir(parents=True, exist_ok=True)
    subprocess.check_call(["git", "clone", "--depth", "1", repo, str(target)])
    print({"status": "downloaded", "path": str(target)})


def cmd_preprocess(args: argparse.Namespace) -> None:
    cfg = PreprocessConfig(
        raw_root=str(Path(args.raw_root).expanduser().resolve()),
        output_root=str(Path(args.output_root).expanduser().resolve()),
        span_resolution=args.span_resolution,
        top_k_per_meter=args.top_k_per_meter,
        onset_bins=args.onset_bins,
        max_extension_class=args.max_extension_class,
        split_seed=args.split_seed,
        train_ratio=args.train_ratio,
        valid_ratio=args.valid_ratio,
        min_notes_per_song=args.min_notes_per_song,
    )
    stats = preprocess_pop909(cfg)
    print(json.dumps(stats, indent=2))


def cmd_train(args: argparse.Namespace) -> None:
    cfg = _load_yaml(Path(args.config))
    if args.data_root:
        cfg["data_root"] = args.data_root
    if args.mode:
        cfg.setdefault("train", {})["mode"] = args.mode
    if args.device:
        cfg["device"] = args.device

    result = run_training(cfg)
    print(json.dumps(result["final"], indent=2))


def cmd_sample(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    states = generate_samples_from_checkpoint(
        checkpoint=Path(args.checkpoint).expanduser().resolve(),
        data_root=Path(args.data_root).expanduser().resolve(),
        split=args.split,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        device=args.device,
        whole_song_mode=args.whole_song_mode,
        whole_song_segments=args.whole_song_segments,
    )

    write_jsonl(out_dir / "samples.jsonl", (s.to_dict() for s in states))

    if args.export_midi:
        from music_graph_dfm.representation.pitch_codec import PitchTokenCodec
        from music_graph_dfm.representation.rhythm_templates import RhythmTemplateVocab
        from music_graph_dfm.utils.io import load_json

        data_root = Path(args.data_root).expanduser().resolve()
        rhythm = RhythmTemplateVocab.from_dict(load_json(data_root / "rhythm_templates.json"))
        pitch = PitchTokenCodec.from_dict(load_json(data_root / "pitch_codec.json"))
        midi_dir = out_dir / "midi"
        midi_dir.mkdir(parents=True, exist_ok=True)
        for i, state in enumerate(states):
            save_state_midi(state, rhythm, pitch, midi_dir / f"sample_{i:04d}.mid")

    print(json.dumps({"num_samples": len(states), "out_dir": str(out_dir)}, indent=2))


def cmd_eval(args: argparse.Namespace) -> None:
    data_root = Path(args.data_root).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve() if args.out else None

    if args.eval_mode == "checkpoint":
        report = evaluate_checkpoint(
            checkpoint=Path(args.checkpoint).expanduser().resolve(),
            data_root=data_root,
            split=args.split,
            num_samples=args.num_samples,
            num_steps=args.num_steps,
            device=args.device,
            out_dir=Path(args.sample_out_dir).expanduser().resolve(),
            out_path=out_path,
        )
    elif args.eval_mode == "sample-dir":
        report = evaluate_sample_directory(
            sample_dir=Path(args.sample_dir).expanduser().resolve(),
            data_root=data_root,
            reference_split=args.split,
            out_path=out_path,
        )
    else:
        report = evaluate_reference_split(
            data_root=data_root,
            split=args.split,
            out_path=out_path,
        )

    if out_path is None:
        print(json.dumps(report, indent=2))
    else:
        save_json(out_path, report)
        print(json.dumps({"saved": str(out_path), "mode": args.eval_mode}, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="music-graph-dfm")
    sub = parser.add_subparsers(dest="command", required=True)

    p_download = sub.add_parser("download-pop909", help="Clone POP909 dataset repository")
    p_download.add_argument("--target-dir", type=str, default="data/raw/POP909-Dataset")
    p_download.add_argument("--force", action="store_true")
    p_download.set_defaults(func=cmd_download_pop909)

    p_pre = sub.add_parser("preprocess", help="Preprocess POP909 to FSNTG-v2 cache")
    p_pre.add_argument("--raw-root", type=str, default="data/raw/POP909-Dataset/POP909")
    p_pre.add_argument("--output-root", type=str, default="data/cache/pop909_fsntg_v2")
    p_pre.add_argument("--span-resolution", type=str, choices=["beat", "half_bar", "bar"], default="beat")
    p_pre.add_argument("--top-k-per-meter", type=int, default=64)
    p_pre.add_argument("--onset-bins", type=int, default=8)
    p_pre.add_argument("--max-extension-class", type=int, default=4)
    p_pre.add_argument("--split-seed", type=int, default=7)
    p_pre.add_argument("--train-ratio", type=float, default=0.8)
    p_pre.add_argument("--valid-ratio", type=float, default=0.1)
    p_pre.add_argument("--min-notes-per-song", type=int, default=16)
    p_pre.set_defaults(func=cmd_preprocess)

    p_train = sub.add_parser("train", help="Train DFM or EditFlow")
    p_train.add_argument("--config", type=str, default="configs/train/default.yaml")
    p_train.add_argument("--data-root", type=str, default=None)
    p_train.add_argument("--mode", type=str, choices=["dfm", "editflow"], default=None)
    p_train.add_argument("--device", type=str, default=None)
    p_train.set_defaults(func=cmd_train)

    p_sample = sub.add_parser("sample", help="Sample from checkpoint")
    p_sample.add_argument("--checkpoint", type=str, required=True)
    p_sample.add_argument("--data-root", type=str, default="data/cache/pop909_fsntg_v2")
    p_sample.add_argument("--split", type=str, default="test")
    p_sample.add_argument("--num-samples", type=int, default=16)
    p_sample.add_argument("--num-steps", type=int, default=96)
    p_sample.add_argument("--device", type=str, default="cpu")
    p_sample.add_argument("--out-dir", type=str, default="artifacts/samples")
    p_sample.add_argument("--export-midi", action="store_true")
    p_sample.add_argument(
        "--whole-song-mode",
        type=str,
        default=None,
        choices=["long_context", "stitching_baseline"],
    )
    p_sample.add_argument("--whole-song-segments", type=int, default=4)
    p_sample.set_defaults(func=cmd_sample)

    p_eval = sub.add_parser("eval", help="Evaluate generated samples")
    p_eval.add_argument("--eval-mode", type=str, choices=["checkpoint", "sample-dir", "reference"], default="checkpoint")
    p_eval.add_argument("--checkpoint", type=str, default="")
    p_eval.add_argument("--sample-dir", type=str, default="artifacts/samples")
    p_eval.add_argument("--sample-out-dir", type=str, default="artifacts/eval_samples")
    p_eval.add_argument("--data-root", type=str, default="data/cache/pop909_fsntg_v2")
    p_eval.add_argument("--split", type=str, default="test")
    p_eval.add_argument("--num-samples", type=int, default=16)
    p_eval.add_argument("--num-steps", type=int, default=96)
    p_eval.add_argument("--device", type=str, default="cpu")
    p_eval.add_argument("--out", type=str, default="artifacts/eval_report.json")
    p_eval.set_defaults(func=cmd_eval)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
