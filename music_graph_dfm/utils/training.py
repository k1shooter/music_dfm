"""Training-time utilities for model construction and checkpoints."""

from __future__ import annotations

from pathlib import Path

from music_graph_dfm.data.pitch_codec import PitchTokenCodec
from music_graph_dfm.models.hetero_fsntg_transformer import FSNTGHeteroTransformer, ModelConfig
from music_graph_dfm.templates.rhythm_templates import RhythmTemplateVocab
from music_graph_dfm.utils.io import load_json


def load_codecs(data_dir: str | Path) -> tuple[RhythmTemplateVocab, PitchTokenCodec]:
    data_dir = Path(data_dir)
    rhythm = RhythmTemplateVocab.from_dict(load_json(data_dir / "rhythm_templates.json"))
    pitch = PitchTokenCodec.from_dict(load_json(data_dir / "pitch_codec.json"))
    return rhythm, pitch


def build_model(vocab_sizes: dict, cfg: dict):
    mcfg = ModelConfig(
        hidden_dim=int(cfg.get("hidden_dim", 256)),
        num_layers=int(cfg.get("num_layers", 6)),
        num_heads=int(cfg.get("num_heads", 8)),
        dropout=float(cfg.get("dropout", 0.1)),
        parameterization=str(cfg.get("parameterization", "velocity")),
        use_long_context_block=bool(cfg.get("use_long_context_block", False)),
    )
    return FSNTGHeteroTransformer(vocab_sizes=vocab_sizes, cfg=mcfg)


def save_checkpoint(path: str | Path, model, optimizer=None, step: int = 0, extra: dict | None = None):
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("save_checkpoint requires torch") from exc

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "step": int(step),
        "extra": extra or {},
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)
    return path


def load_checkpoint(path: str | Path, model, optimizer=None):
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("load_checkpoint requires torch") from exc

    payload = torch.load(Path(path), map_location="cpu")
    model.load_state_dict(payload["model"])
    if optimizer is not None and "optimizer" in payload:
        optimizer.load_state_dict(payload["optimizer"])
    return payload
