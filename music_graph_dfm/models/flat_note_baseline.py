"""Flat note-tuple baseline model for ablation comparison."""

from __future__ import annotations


class FlatNoteTupleBaseline:
    def __init__(self, vocab_sizes: dict, hidden_dim: int = 128):
        try:
            import torch
            import torch.nn as nn
        except Exception as exc:
            raise RuntimeError("FlatNoteTupleBaseline requires torch") from exc

        self.vocab_sizes = vocab_sizes
        self.hidden_dim = hidden_dim

        class _Module(nn.Module):
            pass

        self.module = _Module()
        self.module.emb = nn.ModuleList(
            [
                nn.Embedding(max(2, int(v)), hidden_dim)
                for v in [
                    vocab_sizes.get("note.active", 2),
                    vocab_sizes.get("note.pitch_token", 32),
                    vocab_sizes.get("note.velocity", 16),
                    vocab_sizes.get("note.role", 8),
                    vocab_sizes.get("host", 32),
                    vocab_sizes.get("template", 64),
                ]
            ]
        )
        self.module.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.module.out_pitch = nn.Linear(hidden_dim, max(2, int(vocab_sizes.get("note.pitch_token", 32))))

    def parameters(self):
        return self.module.parameters()

    def to(self, device):
        self.module.to(device)
        return self

    def __call__(self, tuples):
        import torch

        x = 0
        for i, emb in enumerate(self.module.emb):
            x = x + emb(tuples[..., i])
        h = self.module.mlp(x)
        return self.module.out_pitch(h)
