#!/usr/bin/env python
"""Print rhythmic placement template vocabulary statistics."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path

from music_graph_dfm.data.dataset import FSNTGJSONDataset
from music_graph_dfm.templates.rhythm_templates import RhythmTemplateVocab
from music_graph_dfm.utils.io import load_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=Path, default=Path("data/cache/pop909_fsntg"))
    parser.add_argument("--split", type=str, default="train")
    args = parser.parse_args()

    vocab = RhythmTemplateVocab.from_dict(load_json(args.data_root / "rhythm_templates.json"))
    ds = FSNTGJSONDataset(args.data_root / f"{args.split}.jsonl")

    cnt = Counter()
    by_meter = defaultdict(Counter)
    for st in ds.states:
        for i in range(st.num_notes):
            if int(st.note_attrs["active"][i]) == 0:
                continue
            for j, tpl in enumerate(st.e_ns[i]):
                if tpl == 0:
                    continue
                cnt[tpl] += 1
                meter = st.span_attrs["meter"][j]
                by_meter[meter][tpl] += 1

    print("=== Rhythm Template Vocabulary Stats ===")
    print(f"vocab_size: {vocab.vocab_size}")
    print(f"num_examples: {len(ds)}")
    print("global_top_templates:")
    for tpl, f in cnt.most_common(20):
        print(f"  id={tpl:3d} freq={f:6d} tpl={vocab.decode(tpl)}")
    print("per_meter_top_templates:")
    for meter, meter_cnt in sorted(by_meter.items()):
        print(f"  meter={meter}")
        for tpl, f in meter_cnt.most_common(10):
            print(f"    id={tpl:3d} freq={f:6d} tpl={vocab.decode(tpl)}")


if __name__ == "__main__":
    main()
