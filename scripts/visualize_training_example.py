#!/usr/bin/env python
"""Visualize a single cached training example as graph + decoded score artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from music_graph_dfm.data.dataset import FSNTGJSONDataset
from music_graph_dfm.utils.training import load_codecs
from music_graph_dfm.viz.graph_viz import save_graph_visualization
from music_graph_dfm.viz.pianoroll_viz import save_midi_and_roll


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=Path("data/cache/pop909_fsntg"))
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--out_dir", type=Path, default=Path("viz_examples"))
    args = parser.parse_args()

    ds = FSNTGJSONDataset(args.data_root / f"{args.split}.jsonl")
    if len(ds) == 0:
        raise RuntimeError("Dataset split is empty")

    st = ds[args.index % len(ds)]
    rhythm_vocab, pitch_codec = load_codecs(args.data_root)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    graph_path = save_graph_visualization(st, args.out_dir / "example_graph.png")
    other = save_midi_and_roll(st, rhythm_vocab, pitch_codec, args.out_dir, prefix="example")
    print({"graph": str(graph_path), **other})


if __name__ == "__main__":
    main()
