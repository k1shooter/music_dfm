import json
from pathlib import Path

import pytest


torch = pytest.importorskip("torch", exc_type=ImportError)

from music_graph_dfm.representation.pitch_codec import PitchTokenCodec
from music_graph_dfm.representation.rhythm_templates import RhythmTemplateVocab
from music_graph_dfm.representation.state import empty_state
from music_graph_dfm.evaluation.pipeline import evaluate_checkpoint
from music_graph_dfm.training.runner import generate_samples_from_checkpoint, run_training


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _make_state(seed: int):
    st = empty_state(num_spans=2, num_notes=2)
    st.span_attrs["key"] = [0, 0]
    st.span_attrs["harm_root"] = [0, 7]
    st.span_attrs["harm_quality"] = [1, 3]
    st.span_attrs["meter"] = [4, 4]
    st.span_attrs["section"] = [0, 0]
    st.span_attrs["reg_center"] = [4, 4]
    st.note_attrs["active"] = [1, 1]
    st.note_attrs["pitch_token"] = [1 + seed, 2 + seed]
    st.note_attrs["velocity"] = [8, 9]
    st.note_attrs["role"] = [0, 1]
    st.host = [1, 2]
    st.template = [1, 2]
    st.e_ss[0][1] = 1
    return st.to_dict()


def test_checkpoint_metadata_roundtrip(tmp_path: Path):
    data_root = tmp_path / "cache"
    train_rows = [_make_state(0), _make_state(1), _make_state(2)]
    valid_rows = [_make_state(0), _make_state(1)]
    test_rows = [_make_state(3), _make_state(4)]
    _write_jsonl(data_root / "train.jsonl", train_rows)
    _write_jsonl(data_root / "valid.jsonl", valid_rows)
    _write_jsonl(data_root / "test.jsonl", test_rows)

    rhythm = RhythmTemplateVocab(top_k_per_meter=8, onset_bins=8)
    rhythm.fit([(4, 0, 3, 0, 0), (4, 2, 3, 0, 0), (4, 0, 4, 1, 1)])
    pitch = PitchTokenCodec()
    (data_root / "rhythm_templates.json").write_text(json.dumps(rhythm.to_dict()), encoding="utf-8")
    (data_root / "pitch_codec.json").write_text(json.dumps(pitch.to_dict()), encoding="utf-8")
    (data_root / "stats.json").write_text(json.dumps({"schema_version": "fsntg_v2_pop909_v3"}), encoding="utf-8")
    (data_root / "preprocessing_config.json").write_text(json.dumps({"span_resolution": "beat"}), encoding="utf-8")

    ckpt_dir = tmp_path / "ckpt"
    cfg = {
        "seed": 7,
        "device": "cpu",
        "num_workers": 0,
        "data_root": str(data_root),
        "model": {
            "kind": "full",
            "hidden_dim": 32,
            "num_layers": 1,
            "num_heads": 2,
            "dropout": 0.0,
        },
        "diffusion": {
            "path_type": "mixture",
            "schedule": {
                "span_shift": 0.2,
                "span_relation_shift": 0.35,
                "placement_shift": 0.55,
                "note_shift": 0.7,
                "temperature": 0.2,
            },
            "prior": {
                "active_on_prob": 0.2,
                "template_on_prob": 0.25,
                "e_ss_non_none_prob": 0.05,
            },
            "graph_kernel": {"enabled": False},
        },
        "train": {
            "mode": "dfm",
            "epochs": 1,
            "batch_size": 2,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "beta_aux": 0.1,
            "beta_structure": 0.1,
            "save_every": 1,
            "checkpoint_dir": str(ckpt_dir),
        },
    }

    run_training(cfg)
    ckpt_path = ckpt_dir / "epoch_1.pt"
    payload = torch.load(ckpt_path, map_location="cpu")
    extra = payload["extra"]

    assert extra["mode"] == "dfm"
    assert "model_cfg" in extra
    assert "vocab_sizes" in extra
    assert "data_meta" in extra
    assert "graph_kernel" in extra
    assert "graph_kernel_target_rate_mode" in extra
    assert extra["graph_kernel"]["enabled"] is False
    assert extra["graph_kernel"]["approximate"] is False
    assert extra["data_meta"]["stats"]["schema_version"] == "fsntg_v2_pop909_v3"

    samples = generate_samples_from_checkpoint(
        checkpoint=ckpt_path,
        data_root=data_root,
        split="test",
        num_samples=1,
        num_steps=4,
        device="cpu",
        sampler_mode="dfm",
    )
    assert len(samples) == 1


def test_graph_kernel_warning_and_metadata_propagation(tmp_path: Path, caplog):
    data_root = tmp_path / "cache"
    _write_jsonl(data_root / "train.jsonl", [_make_state(0), _make_state(1)])
    _write_jsonl(data_root / "valid.jsonl", [_make_state(0)])
    _write_jsonl(data_root / "test.jsonl", [_make_state(2)])
    rhythm = RhythmTemplateVocab(top_k_per_meter=8, onset_bins=8)
    rhythm.fit([(4, 0, 3, 0, 0), (4, 2, 3, 0, 0)])
    pitch = PitchTokenCodec()
    (data_root / "rhythm_templates.json").write_text(json.dumps(rhythm.to_dict()), encoding="utf-8")
    (data_root / "pitch_codec.json").write_text(json.dumps(pitch.to_dict()), encoding="utf-8")
    (data_root / "stats.json").write_text(json.dumps({"schema_version": "fsntg_v2_pop909_v3"}), encoding="utf-8")
    (data_root / "preprocessing_config.json").write_text(json.dumps({"span_resolution": "beat"}), encoding="utf-8")

    ckpt_dir = tmp_path / "ckpt"
    cfg = {
        "seed": 7,
        "device": "cpu",
        "num_workers": 0,
        "data_root": str(data_root),
        "model": {"kind": "full", "hidden_dim": 32, "num_layers": 1, "num_heads": 2, "dropout": 0.0},
        "diffusion": {
            "path_type": "graph_kernel",
            "schedule": {"span_shift": 0.2, "span_relation_shift": 0.35, "placement_shift": 0.55, "note_shift": 0.7, "temperature": 0.2},
            "prior": {"active_on_prob": 0.2, "template_on_prob": 0.25, "e_ss_non_none_prob": 0.05},
            "graph_kernel": {"enabled": True},
        },
        "train": {
            "mode": "dfm",
            "epochs": 1,
            "batch_size": 1,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "beta_aux": 0.1,
            "beta_structure": 0.0,
            "save_every": 1,
            "checkpoint_dir": str(ckpt_dir),
        },
    }
    caplog.set_level("WARNING")
    run_training(cfg)

    assert any("approximate" in rec.message for rec in caplog.records)
    payload = torch.load(ckpt_dir / "epoch_1.pt", map_location="cpu")
    gk = payload["extra"]["graph_kernel"]
    assert gk["approximate"] is True
    assert gk["enabled"] is True
    assert "target_rate_approximation" in gk

    report = evaluate_checkpoint(
        checkpoint=ckpt_dir / "epoch_1.pt",
        data_root=data_root,
        split="test",
        num_samples=1,
        num_steps=2,
        device="cpu",
        sampler_mode="dfm",
        out_dir=tmp_path / "eval_samples",
        out_path=None,
    )
    assert report["experimental"] is True
    assert report["checkpoint_meta"]["graph_kernel_is_approximate"] is True
    sampling_meta = json.loads((tmp_path / "eval_samples" / "sampling_metadata.json").read_text(encoding="utf-8"))
    assert sampling_meta["graph_kernel_experimental"] is True
