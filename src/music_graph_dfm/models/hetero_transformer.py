"""Heterogeneous graph transformer for FSNTG-v2."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

from music_graph_dfm.constants import AUX_NOTE_RELATIONS, NOTE_CHANNELS, SPAN_CHANNELS

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except Exception:  # pragma: no cover
    torch = None
    nn = None
    F = None


@dataclass
class ModelConfig:
    hidden_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1


if nn is not None:

    class TimeEmbedding(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.dim = dim
            self.proj = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

        def forward(self, t: torch.Tensor) -> torch.Tensor:
            if t.dim() == 0:
                t = t[None]
            half = self.dim // 2
            freqs = torch.exp(
                torch.arange(half, device=t.device, dtype=torch.float32)
                * (-math.log(10000) / max(1, half - 1))
            )
            x = t[:, None] * freqs[None, :]
            emb = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
            if emb.shape[-1] < self.dim:
                emb = torch.cat([emb, torch.zeros((emb.shape[0], 1), device=t.device)], dim=-1)
            return self.proj(emb)


    class NodeHead(nn.Module):
        def __init__(self, hidden_dim: int, vocab: int):
            super().__init__()
            self.lambda_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 1))
            self.logits_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, vocab))

        def forward(self, h: torch.Tensor) -> dict:
            return {"lambda": self.lambda_head(h), "logits": self.logits_head(h)}


    class PairHead(nn.Module):
        def __init__(self, hidden_dim: int, vocab: int):
            super().__init__()
            self.pre = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
            self.lambda_head = nn.Linear(hidden_dim, 1)
            self.logits_head = nn.Linear(hidden_dim, vocab)

        def forward(self, h: torch.Tensor) -> dict:
            p = self.pre(h)
            return {"lambda": self.lambda_head(p), "logits": self.logits_head(p)}


    class HeteroBlock(nn.Module):
        def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
            super().__init__()
            self.span_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            self.note_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
            self.span_norm = nn.LayerNorm(hidden_dim)
            self.note_norm = nn.LayerNorm(hidden_dim)
            self.span_msg = nn.Linear(hidden_dim, hidden_dim)
            self.note_msg = nn.Linear(hidden_dim, hidden_dim)
            self.span_ffn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
            )
            self.note_ffn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
            )

        def _agg(self, adj: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
            deg = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
            return adj @ src / deg

        def forward(
            self,
            span_h: torch.Tensor,
            note_h: torch.Tensor,
            adj_ns: torch.Tensor,
            adj_ss: torch.Tensor,
            adj_nn: torch.Tensor,
            span_mask: torch.Tensor,
            note_mask: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            span_self, _ = self.span_attn(span_h, span_h, span_h, key_padding_mask=~span_mask, need_weights=False)
            note_self, _ = self.note_attn(note_h, note_h, note_h, key_padding_mask=~note_mask, need_weights=False)

            span_from_note = self._agg(adj_ns.transpose(1, 2), note_h)
            note_from_span = self._agg(adj_ns, span_h)
            span_from_span = self._agg(adj_ss, span_h)
            note_from_note = self._agg(adj_nn, note_h)

            span_h = self.span_norm(span_h + span_self + self.span_msg(span_from_note + span_from_span))
            note_h = self.note_norm(note_h + note_self + self.note_msg(note_from_span + note_from_note))

            span_h = span_h + self.span_ffn(span_h)
            note_h = note_h + self.note_ffn(note_h)

            span_h = span_h * span_mask.unsqueeze(-1)
            note_h = note_h * note_mask.unsqueeze(-1)
            return span_h, note_h


    class FSNTGV2HeteroTransformer(nn.Module):
        def __init__(
            self,
            vocab_sizes: Dict[str, int],
            cfg: ModelConfig,
            template_onset_ticks: list[int] | None = None,
            template_duration_ticks: list[int] | None = None,
        ):
            super().__init__()
            self.vocab_sizes = vocab_sizes
            self.hidden_dim = cfg.hidden_dim

            self.span_emb = nn.ModuleDict(
                {
                    channel: nn.Embedding(max(2, int(vocab_sizes[f"span.{channel}"])), cfg.hidden_dim)
                    for channel in SPAN_CHANNELS
                }
            )
            self.note_emb = nn.ModuleDict(
                {
                    channel: nn.Embedding(max(2, int(vocab_sizes[f"note.{channel}"])), cfg.hidden_dim)
                    for channel in NOTE_CHANNELS
                }
            )
            self.host_emb = nn.Embedding(max(2, int(vocab_sizes["note.host"])), cfg.hidden_dim)
            self.template_emb = nn.Embedding(max(2, int(vocab_sizes["note.template"])), cfg.hidden_dim)
            self.e_ss_emb = nn.Embedding(max(2, int(vocab_sizes["e_ss.relation"])), cfg.hidden_dim)
            self.aux_rel_emb = nn.Embedding(len(AUX_NOTE_RELATIONS), cfg.hidden_dim)

            self.time_emb = TimeEmbedding(cfg.hidden_dim)
            self.layers = nn.ModuleList(
                [HeteroBlock(cfg.hidden_dim, cfg.num_heads, cfg.dropout) for _ in range(cfg.num_layers)]
            )

            self.span_heads = nn.ModuleDict(
                {
                    channel: NodeHead(cfg.hidden_dim, max(2, int(vocab_sizes[f"span.{channel}"])))
                    for channel in SPAN_CHANNELS
                }
            )
            self.note_heads = nn.ModuleDict(
                {
                    channel: NodeHead(cfg.hidden_dim, max(2, int(vocab_sizes[f"note.{channel}"])))
                    for channel in NOTE_CHANNELS
                }
            )
            self.host_head = NodeHead(cfg.hidden_dim, max(2, int(vocab_sizes["note.host"])))
            self.template_head = NodeHead(cfg.hidden_dim, max(2, int(vocab_sizes["note.template"])))
            self.e_ss_head = PairHead(cfg.hidden_dim, max(2, int(vocab_sizes["e_ss.relation"])))

            # Edit-flow heads.
            num_types = 6
            self.edit_global = nn.Sequential(nn.Linear(cfg.hidden_dim * 2, cfg.hidden_dim), nn.SiLU())
            self.edit_lambda_type = nn.Linear(cfg.hidden_dim, num_types)
            self.edit_type_logits = nn.Linear(cfg.hidden_dim, num_types)
            self.edit_note_idx = nn.Linear(cfg.hidden_dim, 1)
            self.edit_host_logits = nn.Linear(cfg.hidden_dim, max(2, int(vocab_sizes["note.host"])))
            self.edit_template_logits = nn.Linear(cfg.hidden_dim, max(2, int(vocab_sizes["note.template"])))
            self.edit_pitch_logits = nn.Linear(cfg.hidden_dim, max(2, int(vocab_sizes["note.pitch_token"])))
            self.edit_velocity_logits = nn.Linear(cfg.hidden_dim, max(2, int(vocab_sizes["note.velocity"])))
            self.edit_role_logits = nn.Linear(cfg.hidden_dim, max(2, int(vocab_sizes["note.role"])))
            self.edit_span_src = nn.Linear(cfg.hidden_dim, 1)
            self.edit_span_dst = nn.Linear(cfg.hidden_dim, 1)
            self.edit_insert_host = nn.Linear(cfg.hidden_dim, max(2, int(vocab_sizes["note.host"])))
            self.edit_insert_template = nn.Linear(cfg.hidden_dim, max(2, int(vocab_sizes["note.template"])))
            self.edit_span_rel = nn.Linear(cfg.hidden_dim, max(2, int(vocab_sizes["e_ss.relation"])))

            onset = template_onset_ticks or [0]
            dur = template_duration_ticks or [1]
            self.register_buffer("template_onset_ticks", torch.tensor(onset, dtype=torch.float32), persistent=False)
            self.register_buffer("template_duration_ticks", torch.tensor(dur, dtype=torch.float32), persistent=False)

        def _template_lookup(self, template: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            vmax = self.template_onset_ticks.shape[0] - 1
            template = template.clamp(min=0, max=max(0, vmax))
            onset = self.template_onset_ticks[template]
            duration = self.template_duration_ticks[template]
            return onset, duration

        def _reconstruct_aux_relations(self, batch: dict) -> torch.Tensor:
            host = batch["host"]
            template = batch["template"]
            role = batch["note"]["role"]
            active = batch["note"]["active"]
            note_mask = batch["note_mask"]
            ticks_per_span = batch["ticks_per_span"].to(host.device)

            bsz, n = host.shape
            rel = torch.zeros((bsz, n, n), dtype=torch.long, device=host.device)
            onset_tbl, dur_tbl = self._template_lookup(template)

            for b in range(bsz):
                tps = ticks_per_span[b].to(torch.float32)
                span_idx = host[b].to(torch.float32) - 1.0
                onset = span_idx * tps + onset_tbl[b]
                end = onset + dur_tbl[b]
                valid = note_mask[b] & (active[b] == 1) & (host[b] > 0) & (template[b] > 0)

                for i in range(n):
                    if not bool(valid[i]):
                        continue
                    for j in range(i + 1, n):
                        if not bool(valid[j]):
                            continue
                        if int(onset[i].item()) == int(onset[j].item()):
                            rel[b, i, j] = max(rel[b, i, j], 1)
                            rel[b, j, i] = max(rel[b, j, i], 1)
                        if float(onset[i].item()) < float(end[j].item()) and float(onset[j].item()) < float(end[i].item()):
                            rel[b, i, j] = max(rel[b, i, j], 2)
                            rel[b, j, i] = max(rel[b, j, i], 2)

                for r in torch.unique(role[b][valid]):
                    idx = torch.where(valid & (role[b] == r))[0]
                    if idx.numel() < 2:
                        continue
                    onset_sel = onset[idx]
                    order = idx[torch.argsort(onset_sel)]
                    for left, right in zip(order[:-1], order[1:]):
                        rel[b, left, right] = max(rel[b, left, right], 3)

            return rel

        def _build_adj(self, batch: dict, aux_rel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            host = batch["host"]
            template = batch["template"]
            active = batch["note"]["active"]
            note_mask = batch["note_mask"]
            span_mask = batch["span_mask"]
            e_ss = batch["e_ss"]

            bsz, n = host.shape
            s = span_mask.shape[1]

            host_idx = (host - 1).clamp(min=0, max=max(0, s - 1))
            adj_ns = F.one_hot(host_idx, num_classes=s).to(torch.float32)
            valid_note = note_mask & (active == 1) & (host > 0) & (template > 0)
            adj_ns = adj_ns * valid_note.unsqueeze(-1).to(torch.float32)

            adj_ss = (e_ss != 0).to(torch.float32)
            eye = torch.eye(s, device=e_ss.device, dtype=torch.float32).unsqueeze(0)
            adj_ss = (adj_ss + eye) * (span_mask.unsqueeze(-1) & span_mask.unsqueeze(-2)).to(torch.float32)

            adj_nn = (aux_rel != 0).to(torch.float32)
            eye_n = torch.eye(n, device=host.device, dtype=torch.float32).unsqueeze(0)
            adj_nn = (adj_nn + eye_n) * (note_mask.unsqueeze(-1) & note_mask.unsqueeze(-2)).to(torch.float32)
            return adj_ns, adj_ss, adj_nn

        def _encode(self, batch: dict, t: torch.Tensor):
            span_mask = batch["span_mask"]
            note_mask = batch["note_mask"]
            bsz, s = span_mask.shape
            n = note_mask.shape[1]

            span_h = torch.zeros((bsz, s, self.hidden_dim), device=span_mask.device)
            note_h = torch.zeros((bsz, n, self.hidden_dim), device=span_mask.device)

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
            span_h = span_h + temb[:, None, :]
            note_h = note_h + temb[:, None, :]

            aux_rel = self._reconstruct_aux_relations(batch)
            adj_ns, adj_ss, adj_nn = self._build_adj(batch, aux_rel)

            for layer in self.layers:
                span_h, note_h = layer(span_h, note_h, adj_ns, adj_ss, adj_nn, span_mask, note_mask)

            return span_h, note_h, aux_rel

        def forward(self, batch: dict, t):
            span_h, note_h, aux_rel = self._encode(batch, t)
            e_ss = batch["e_ss"]
            aux_emb = self.aux_rel_emb(aux_rel)

            outputs = {}
            for channel in SPAN_CHANNELS:
                outputs[f"span.{channel}"] = self.span_heads[channel](span_h)

            outputs["note.active"] = self.note_heads["active"](note_h)
            outputs["note.pitch_token"] = self.note_heads["pitch_token"](note_h)
            outputs["note.velocity"] = self.note_heads["velocity"](note_h)
            outputs["note.role"] = self.note_heads["role"](note_h)
            outputs["note.host"] = self.host_head(note_h)
            outputs["note.template"] = self.template_head(note_h)

            pair_ss = span_h[:, :, None, :] + span_h[:, None, :, :] + self.e_ss_emb(e_ss)
            outputs["e_ss.relation"] = self.e_ss_head(pair_ss)

            # Use aux relation embedding to affect placement/content heads.
            note_h = note_h + (aux_emb.sum(dim=-2) / max(1, aux_emb.shape[-2]))
            outputs["note.pitch_token"] = self.note_heads["pitch_token"](note_h)
            outputs["note.velocity"] = self.note_heads["velocity"](note_h)
            outputs["note.role"] = self.note_heads["role"](note_h)
            outputs["note.host"] = self.host_head(note_h)
            outputs["note.template"] = self.template_head(note_h)
            return outputs

        def forward_edit(self, batch: dict, t):
            span_h, note_h, _ = self._encode(batch, t)
            span_mask = batch["span_mask"].to(torch.float32)
            note_mask = batch["note_mask"].to(torch.float32)

            span_pool = (span_h * span_mask.unsqueeze(-1)).sum(dim=1) / span_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            note_pool = (note_h * note_mask.unsqueeze(-1)).sum(dim=1) / note_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            g = self.edit_global(torch.cat([span_pool, note_pool], dim=-1))

            pair_ss = span_h[:, :, None, :] + span_h[:, None, :, :]

            return {
                "lambda_type": self.edit_lambda_type(g),
                "type_logits": self.edit_type_logits(g),
                "note_logits": self.edit_note_idx(note_h).squeeze(-1),
                "host_logits": self.edit_host_logits(note_h),
                "template_logits": self.edit_template_logits(note_h),
                "pitch_logits": self.edit_pitch_logits(note_h),
                "velocity_logits": self.edit_velocity_logits(note_h),
                "role_logits": self.edit_role_logits(note_h),
                "span_src_logits": self.edit_span_src(span_h).squeeze(-1),
                "span_dst_logits": self.edit_span_dst(span_h).squeeze(-1),
                "insert_host_logits": self.edit_insert_host(g),
                "insert_template_logits": self.edit_insert_template(g),
                "span_rel_logits": self.edit_span_rel(pair_ss),
            }


else:

    class FSNTGV2HeteroTransformer:  # type: ignore
        def __init__(self, *args, **kwargs):
            del args, kwargs
            raise RuntimeError("FSNTGV2HeteroTransformer requires torch")


    @dataclass
    class ModelConfig:  # type: ignore
        hidden_dim: int = 256
        num_layers: int = 6
        num_heads: int = 8
        dropout: float = 0.1
