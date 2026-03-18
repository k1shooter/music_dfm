#!/usr/bin/env python
"""Sample a graph from trained FSNTG model and export MIDI/visualizations."""

from __future__ import annotations

import argparse
from pathlib import Path

from music_graph_dfm.data.dataset import FSNTGJSONDataset, collate_fsntg, infer_vocab_sizes
from music_graph_dfm.data.fsntg import cleanup_duplicate_notes, project_one_host_per_active_note
from music_graph_dfm.data.tensor_codec import coords_to_states
from music_graph_dfm.diffusion.state_ops import PriorConfig, batch_to_coords, sample_factorized_prior
from music_graph_dfm.samplers.ctmc_sampler import ctmc_sample
from music_graph_dfm.utils.training import build_model, load_checkpoint, load_codecs
from music_graph_dfm.viz.graph_viz import save_graph_visualization
from music_graph_dfm.viz.pianoroll_viz import save_midi_and_roll


def _move_to_device(obj, device):
    import torch

    if isinstance(obj, dict):
        return {k: _move_to_device(v, device) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_move_to_device(v, device) for v in obj]
    if torch.is_tensor(obj):
        return obj.to(device)
    return obj


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=Path("data/cache/pop909_fsntg"))
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--num_steps", type=int, default=64)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--out_dir", type=Path, default=Path("samples"))
    args = parser.parse_args()

    import torch

    dataset = FSNTGJSONDataset(args.data_root / f"{args.split}.jsonl")
    if len(dataset) == 0:
        raise RuntimeError("Empty dataset split.")

    ref_state = dataset[args.index % len(dataset)]
    batch = collate_fsntg([ref_state])

    vocab_sizes = infer_vocab_sizes(dataset.states)
    model = build_model(vocab_sizes=vocab_sizes, cfg={
        "hidden_dim": 256,
        "num_layers": 6,
        "num_heads": 8,
        "dropout": 0.1,
        "parameterization": "velocity",
        "use_long_context_block": False,
    })
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model.to(device)

    ckpt = load_checkpoint(args.checkpoint, model)
    if "extra" in ckpt and "vocab_sizes" in ckpt["extra"]:
        vocab_sizes = ckpt["extra"]["vocab_sizes"]

    batch = _move_to_device(batch, device)
    x0 = sample_factorized_prior(batch, vocab_sizes=vocab_sizes, prior_cfg=PriorConfig())

    with torch.no_grad():
        sampled_coords = ctmc_sample(
            model,
            init_coords=x0,
            base_batch=batch,
            num_steps=args.num_steps,
            t_start=1e-3,
            t_end=0.999,
        )

    states = coords_to_states(sampled_coords, batch)
    rhythm_vocab, pitch_codec = load_codecs(args.data_root)

    state = project_one_host_per_active_note(states[0])
    state = cleanup_duplicate_notes(state, rhythm_vocab, pitch_codec)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    save_graph_visualization(state, args.out_dir / "sample_graph.png")
    artifacts = save_midi_and_roll(state, rhythm_vocab, pitch_codec, args.out_dir, prefix="sample")
    print({"graph": str(args.out_dir / "sample_graph.png"), **artifacts})


if __name__ == "__main__":
    main()
