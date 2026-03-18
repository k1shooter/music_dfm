"""Simple baseline model with minimal graph interactions."""

from __future__ import annotations

from typing import Dict

try:
    import torch
    import torch.nn as nn
except Exception:  # pragma: no cover
    torch = None
    nn = None

from music_graph_dfm.constants import NOTE_CHANNELS, SPAN_CHANNELS
from music_graph_dfm.models.hetero_transformer import ModelConfig, NodeHead, PairHead, TimeEmbedding


if nn is not None:

    class SimpleFactorizedBaseline(nn.Module):
        def __init__(self, vocab_sizes: Dict[str, int], cfg: ModelConfig):
            super().__init__()
            d = cfg.hidden_dim
            self.span_emb = nn.ModuleDict(
                {
                    channel: nn.Embedding(max(2, int(vocab_sizes[f"span.{channel}"])), d)
                    for channel in SPAN_CHANNELS
                }
            )
            self.note_emb = nn.ModuleDict(
                {
                    channel: nn.Embedding(max(2, int(vocab_sizes[f"note.{channel}"])), d)
                    for channel in NOTE_CHANNELS
                }
            )
            self.host_emb = nn.Embedding(max(2, int(vocab_sizes["note.host"])), d)
            self.template_emb = nn.Embedding(max(2, int(vocab_sizes["note.template"])), d)
            self.e_ss_emb = nn.Embedding(max(2, int(vocab_sizes["e_ss.relation"])), d)
            self.time_emb = TimeEmbedding(d)

            self.span_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
            self.note_mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))

            self.span_heads = nn.ModuleDict(
                {
                    channel: NodeHead(d, max(2, int(vocab_sizes[f"span.{channel}"])))
                    for channel in SPAN_CHANNELS
                }
            )
            self.note_heads = nn.ModuleDict(
                {
                    channel: NodeHead(d, max(2, int(vocab_sizes[f"note.{channel}"])))
                    for channel in NOTE_CHANNELS
                }
            )
            self.host_head = NodeHead(d, max(2, int(vocab_sizes["note.host"])))
            self.template_head = NodeHead(d, max(2, int(vocab_sizes["note.template"])))
            self.e_ss_head = PairHead(d, max(2, int(vocab_sizes["e_ss.relation"])))

            # Reuse edit API expected by edit-flow code.
            self.edit_lambda = nn.Linear(d * 2, 6)
            self.edit_type = nn.Linear(d * 2, 6)
            self.note_idx = nn.Linear(d, 1)
            self.edit_host = nn.Linear(d, max(2, int(vocab_sizes["note.host"])))
            self.edit_template = nn.Linear(d, max(2, int(vocab_sizes["note.template"])))
            self.edit_pitch = nn.Linear(d, max(2, int(vocab_sizes["note.pitch_token"])))
            self.edit_velocity = nn.Linear(d, max(2, int(vocab_sizes["note.velocity"])))
            self.edit_role = nn.Linear(d, max(2, int(vocab_sizes["note.role"])))
            self.span_src = nn.Linear(d, 1)
            self.span_dst = nn.Linear(d, 1)
            self.insert_host = nn.Linear(d * 2, max(2, int(vocab_sizes["note.host"])))
            self.insert_template = nn.Linear(d * 2, max(2, int(vocab_sizes["note.template"])))
            self.span_rel = nn.Linear(d, max(2, int(vocab_sizes["e_ss.relation"])))

        def _encode(self, batch: dict, t):
            span_mask = batch["span_mask"]
            note_mask = batch["note_mask"]
            bsz, s = span_mask.shape
            n = note_mask.shape[1]

            span_h = torch.zeros((bsz, s, self.span_mlp[0].in_features), device=span_mask.device)
            note_h = torch.zeros((bsz, n, self.note_mlp[0].in_features), device=span_mask.device)

            for channel in SPAN_CHANNELS:
                span_h = span_h + self.span_emb[channel](batch["span"][channel])
            for channel in NOTE_CHANNELS:
                note_h = note_h + self.note_emb[channel](batch["note"][channel])
            note_h = note_h + self.host_emb(batch["host"]) + self.template_emb(batch["template"])

            if not torch.is_tensor(t):
                t = torch.tensor(float(t), device=span_h.device)
            if t.dim() == 0:
                t = t.repeat(bsz)
            temb = self.time_emb(t)
            span_h = self.span_mlp(span_h + temb[:, None, :]) * span_mask.unsqueeze(-1)
            note_h = self.note_mlp(note_h + temb[:, None, :]) * note_mask.unsqueeze(-1)
            return span_h, note_h

        def forward(self, batch: dict, t):
            span_h, note_h = self._encode(batch, t)
            outputs = {}
            for channel in SPAN_CHANNELS:
                outputs[f"span.{channel}"] = self.span_heads[channel](span_h)

            outputs["note.active"] = self.note_heads["active"](note_h)
            outputs["note.pitch_token"] = self.note_heads["pitch_token"](note_h)
            outputs["note.velocity"] = self.note_heads["velocity"](note_h)
            outputs["note.role"] = self.note_heads["role"](note_h)
            outputs["note.host"] = self.host_head(note_h)
            outputs["note.template"] = self.template_head(note_h)

            pair = span_h[:, :, None, :] + span_h[:, None, :, :] + self.e_ss_emb(batch["e_ss"])
            outputs["e_ss.relation"] = self.e_ss_head(pair)
            return outputs

        def forward_edit(self, batch: dict, t):
            span_h, note_h = self._encode(batch, t)
            span_mask = batch["span_mask"].to(torch.float32)
            note_mask = batch["note_mask"].to(torch.float32)
            span_pool = (span_h * span_mask.unsqueeze(-1)).sum(dim=1) / span_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            note_pool = (note_h * note_mask.unsqueeze(-1)).sum(dim=1) / note_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            g = torch.cat([span_pool, note_pool], dim=-1)
            pair = span_h[:, :, None, :] + span_h[:, None, :, :]
            return {
                "lambda_type": self.edit_lambda(g),
                "type_logits": self.edit_type(g),
                "note_logits": self.note_idx(note_h).squeeze(-1),
                "host_logits": self.edit_host(note_h),
                "template_logits": self.edit_template(note_h),
                "pitch_logits": self.edit_pitch(note_h),
                "velocity_logits": self.edit_velocity(note_h),
                "role_logits": self.edit_role(note_h),
                "span_src_logits": self.span_src(span_h).squeeze(-1),
                "span_dst_logits": self.span_dst(span_h).squeeze(-1),
                "insert_host_logits": self.insert_host(g),
                "insert_template_logits": self.insert_template(g),
                "span_rel_logits": self.span_rel(pair),
            }


else:

    class SimpleFactorizedBaseline:  # type: ignore
        def __init__(self, *args, **kwargs):
            del args, kwargs
            raise RuntimeError("SimpleFactorizedBaseline requires torch")
