import random
import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from music_graph_dfm.diffusion.edit_flow import (
    EditMoveType,
    derive_oracle_edit_move,
    sample_edit_ctmc_step,
    sample_forward_edit_ctmc_source,
)
from music_graph_dfm.representation.pitch_codec import PitchTokenCodec
from music_graph_dfm.representation.rhythm_templates import RhythmTemplateVocab
from music_graph_dfm.representation.state import empty_state
from music_graph_dfm.training.runner import run_training


def _target_state():
    st = empty_state(num_spans=2, num_notes=2)
    st.note_attrs["active"] = [1, 1]
    st.note_attrs["pitch_token"] = [3, 5]
    st.note_attrs["velocity"] = [8, 9]
    st.note_attrs["role"] = [0, 1]
    st.host = [1, 2]
    st.template = [2, 3]
    st.e_ss[0][1] = 1
    st.project_placement_consistency()
    return st


def test_forward_edit_ctmc_source_is_valid():
    target = _target_state()
    vocab_sizes = {
        "note.pitch_token": 16,
        "note.velocity": 16,
        "note.role": 8,
        "note.host": 4,
        "note.template": 6,
        "e_ss.relation": 6,
    }
    source = sample_forward_edit_ctmc_source(
        target_state=target,
        vocab_sizes=vocab_sizes,
        rng=random.Random(7),
        num_steps=1,
        h=0.5,
    )
    source.validate_shapes()
    source.project_placement_consistency()
    for i, active in enumerate(source.note_attrs["active"]):
        if int(active) == 0:
            assert int(source.host[i]) == 0
            assert int(source.template[i]) == 0
    assert derive_oracle_edit_move(source, target) is None or isinstance(
        derive_oracle_edit_move(source, target).move_type, EditMoveType
    )


def test_edit_sampler_substitute_host_is_offdiagonal():
    state = _target_state()
    # Force SUBSTITUTE_HOST with near-certain jump.
    lam = torch.full((1, 6), -20.0)
    lam[0, int(EditMoveType.SUBSTITUTE_HOST)] = 20.0
    edit_outputs_single = {
        "lambda_type": lam,
        "type_logits": torch.zeros((1, 6)),
        "note_logits": torch.zeros((1, state.num_notes)),
        "host_logits": torch.zeros((1, state.num_notes, 4)),
        "template_logits": torch.zeros((1, state.num_notes, 6)),
        "pitch_logits": torch.zeros((1, state.num_notes, 8)),
        "velocity_logits": torch.zeros((1, state.num_notes, 8)),
        "role_logits": torch.zeros((1, state.num_notes, 4)),
        "span_src_logits": torch.zeros((1, state.num_spans)),
        "span_dst_logits": torch.zeros((1, state.num_spans)),
        "insert_host_logits": torch.zeros((1, 4)),
        "insert_template_logits": torch.zeros((1, 6)),
        "insert_pitch_logits": torch.zeros((1, 8)),
        "insert_velocity_logits": torch.zeros((1, 8)),
        "insert_role_logits": torch.zeros((1, 4)),
        "span_rel_logits": torch.zeros((1, state.num_spans, state.num_spans, 6)),
    }
    edit_outputs_single["host_logits"][0, 0, 2] = 10.0
    next_state = sample_edit_ctmc_step(state=state, edit_outputs_single=edit_outputs_single, h=1.0)
    assert int(next_state.host[0]) != int(state.host[0])


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _make_train_state(seed: int) -> dict:
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


def test_editflow_rejects_multistep_source_by_default(tmp_path: Path):
    data_root = tmp_path / "cache"
    _write_jsonl(data_root / "train.jsonl", [_make_train_state(0), _make_train_state(1)])
    _write_jsonl(data_root / "valid.jsonl", [_make_train_state(2)])
    rhythm = RhythmTemplateVocab(top_k_per_meter=8, onset_bins=8)
    rhythm.fit([(4, 0, 3, 0, 0), (4, 2, 3, 0, 0)])
    pitch = PitchTokenCodec()
    (data_root / "rhythm_templates.json").write_text(json.dumps(rhythm.to_dict()), encoding="utf-8")
    (data_root / "pitch_codec.json").write_text(json.dumps(pitch.to_dict()), encoding="utf-8")
    (data_root / "stats.json").write_text(json.dumps({"schema_version": "fsntg_v2_pop909_v2"}), encoding="utf-8")
    (data_root / "preprocessing_config.json").write_text(json.dumps({"span_resolution": "beat"}), encoding="utf-8")

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
            "schedule": {"span_shift": 0.2, "span_relation_shift": 0.35, "placement_shift": 0.55, "note_shift": 0.7, "temperature": 0.2},
            "prior": {"active_on_prob": 0.2, "template_on_prob": 0.25, "e_ss_non_none_prob": 0.05},
            "graph_kernel": {"enabled": False},
        },
        "train": {
            "mode": "editflow",
            "epochs": 1,
            "batch_size": 1,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "save_every": 1,
            "checkpoint_dir": str(tmp_path / "ckpt"),
            "editflow_source_steps": 2,
        },
    }

    with pytest.raises(ValueError, match="one-step oracle"):
        run_training(cfg)


def test_editflow_multistep_segment_smoke(tmp_path: Path):
    data_root = tmp_path / "cache"
    _write_jsonl(data_root / "train.jsonl", [_make_train_state(0), _make_train_state(1), _make_train_state(2)])
    _write_jsonl(data_root / "valid.jsonl", [_make_train_state(3)])
    _write_jsonl(data_root / "test.jsonl", [_make_train_state(4)])
    rhythm = RhythmTemplateVocab(top_k_per_meter=8, onset_bins=8)
    rhythm.fit([(4, 0, 3, 0, 0), (4, 2, 3, 0, 0)])
    pitch = PitchTokenCodec()
    (data_root / "rhythm_templates.json").write_text(json.dumps(rhythm.to_dict()), encoding="utf-8")
    (data_root / "pitch_codec.json").write_text(json.dumps(pitch.to_dict()), encoding="utf-8")
    (data_root / "stats.json").write_text(json.dumps({"schema_version": "fsntg_v2_pop909_v2"}), encoding="utf-8")
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
            "schedule": {"span_shift": 0.2, "span_relation_shift": 0.35, "placement_shift": 0.55, "note_shift": 0.7, "temperature": 0.2},
            "prior": {"active_on_prob": 0.2, "template_on_prob": 0.25, "e_ss_non_none_prob": 0.05},
            "graph_kernel": {"enabled": False},
        },
        "train": {
            "mode": "editflow",
            "editflow_mode": "multistep_segment",
            "epochs": 1,
            "batch_size": 1,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
            "save_every": 1,
            "checkpoint_dir": str(ckpt_dir),
            "editflow_source_steps": 3,
        },
    }

    run_training(cfg)
    payload = torch.load(ckpt_dir / "epoch_1.pt", map_location="cpu")
    extra = payload["extra"]
    assert extra["editflow_mode"] == "multistep_segment"
    assert extra["editflow_is_experimental"] is True
