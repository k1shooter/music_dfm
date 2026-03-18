#!/usr/bin/env python
"""Whole-song sampling by direct long context or segment stitching."""

from __future__ import annotations

import argparse
from pathlib import Path

from music_graph_dfm.data.dataset import FSNTGJSONDataset, collate_fsntg, infer_vocab_sizes
from music_graph_dfm.data.fsntg import (
    FSNTGState,
    cleanup_duplicate_notes,
    empty_state,
    project_one_host_per_active_note,
)
from music_graph_dfm.data.tensor_codec import coords_to_states
from music_graph_dfm.diffusion.state_ops import PriorConfig, sample_factorized_prior
from music_graph_dfm.samplers.ctmc_sampler import ctmc_sample
from music_graph_dfm.utils.training import build_model, load_checkpoint, load_codecs
from music_graph_dfm.viz.pianoroll_viz import save_midi_and_roll


def stitch_segments(segments: list[FSNTGState]) -> FSNTGState:
    if not segments:
        return empty_state(num_spans=1, num_notes=0)

    out = segments[0].copy()
    span_offset = out.num_spans
    note_offset = out.num_notes

    for seg in segments[1:]:
        for c, vals in seg.span_attrs.items():
            out.span_attrs[c].extend(vals)
        for c, vals in seg.note_attrs.items():
            out.note_attrs[c].extend(vals)

        for row in out.e_ss:
            row.extend([0 for _ in range(seg.num_spans)])
        for row in seg.e_ss:
            out.e_ss.append([0 for _ in range(span_offset)] + row)

        for row in seg.e_ns:
            out.e_ns.append([0 for _ in range(span_offset)] + row)

        shift = out.span_starts[-1] + out.ticks_per_span
        out.span_starts.extend([shift + j * out.ticks_per_span for j in range(seg.num_spans)])

        out.e_ss[span_offset - 1][span_offset] = 1  # deterministic next edge at stitch boundary
        span_offset += seg.num_spans
        note_offset += seg.num_notes

    out.validate_shapes()
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=Path("data/cache/pop909_fsntg"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mode", type=str, default="direct", choices=["direct", "stitch"])
    parser.add_argument("--segments", type=int, default=4)
    parser.add_argument("--num_steps", type=int, default=96)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out_dir", type=Path, default=Path("samples/whole_song"))
    args = parser.parse_args()

    import torch

    dataset = FSNTGJSONDataset(args.data_root / "test.jsonl")
    if len(dataset) == 0:
        raise RuntimeError("Empty test split.")

    vocab_sizes = infer_vocab_sizes(dataset.states)
    model = build_model(vocab_sizes=vocab_sizes, cfg={
        "hidden_dim": 256,
        "num_layers": 6,
        "num_heads": 8,
        "dropout": 0.1,
        "parameterization": "velocity",
        "use_long_context_block": True,
    })

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model.to(device)
    load_checkpoint(args.checkpoint, model)

    def sample_from_template_state(template_state: FSNTGState) -> FSNTGState:
        batch = collate_fsntg([template_state])
        batch = {k: ({kk: vv.to(device) for kk, vv in v.items()} if isinstance(v, dict) else v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
        x0 = sample_factorized_prior(batch, vocab_sizes=vocab_sizes, prior_cfg=PriorConfig())
        with torch.no_grad():
            sampled = ctmc_sample(model, x0, batch, num_steps=args.num_steps)
        return coords_to_states(sampled, batch)[0]

    if args.mode == "direct":
        base = dataset[0]
        sampled_state = sample_from_template_state(base)
    else:
        segments = []
        for i in range(args.segments):
            template = dataset[i % len(dataset)]
            segments.append(sample_from_template_state(template))
        sampled_state = stitch_segments(segments)

    rhythm_vocab, pitch_codec = load_codecs(args.data_root)
    sampled_state = project_one_host_per_active_note(sampled_state)
    sampled_state = cleanup_duplicate_notes(sampled_state, rhythm_vocab, pitch_codec)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    artifacts = save_midi_and_roll(sampled_state, rhythm_vocab, pitch_codec, args.out_dir, prefix="whole_song")
    print(artifacts)


if __name__ == "__main__":
    main()
