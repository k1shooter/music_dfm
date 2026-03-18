"""Heterogeneous FSNTG transformer with factorized channel heads."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict

from music_graph_dfm.utils.constants import AUX_NOTE_RELATIONS, NOTE_CHANNELS, SPAN_CHANNELS

try:
    import torch
    import torch.nn as nn
except Exception:
    torch = None
    nn = None


@dataclass
class ModelConfig:
    hidden_dim: int = 256
    num_layers: int = 6
    num_heads: int = 8
    dropout: float = 0.1
    parameterization: str = "velocity"  # velocity | posterior
    use_long_context_block: bool = False


if nn is not None:

    class TimeEmbedding(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.dim = dim
            self.proj = nn.Sequential(
                nn.Linear(dim, dim),
                nn.SiLU(),
                nn.Linear(dim, dim),
            )

        def _sinusoidal(self, t: torch.Tensor) -> torch.Tensor:
            if t.dim() == 0:
                t = t[None]
            half = self.dim // 2
            freq = torch.exp(
                torch.arange(half, device=t.device, dtype=torch.float32)
                * (-math.log(10000) / max(1, half - 1))
            )
            ang = t[:, None] * freq[None, :]
            emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=-1)
            if emb.shape[-1] < self.dim:
                emb = torch.cat([emb, torch.zeros((emb.shape[0], 1), device=t.device)], dim=-1)
            return emb

        def forward(self, t: torch.Tensor) -> torch.Tensor:
            return self.proj(self._sinusoidal(t))


    class NodeCoordinateHead(nn.Module):
        def __init__(self, hidden_dim: int, vocab_size: int):
            super().__init__()
            self.lambda_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )
            self.logits_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, vocab_size),
            )

        def forward(self, h: torch.Tensor) -> dict:
            return {
                "lambda": self.lambda_head(h),
                "logits": self.logits_head(h),
            }


    class PairCoordinateHead(nn.Module):
        def __init__(self, hidden_dim: int, vocab_size: int):
            super().__init__()
            self.pre = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU())
            self.lambda_head = nn.Linear(hidden_dim, 1)
            self.logits_head = nn.Linear(hidden_dim, vocab_size)

        def forward(self, pair_h: torch.Tensor) -> dict:
            h = self.pre(pair_h)
            return {
                "lambda": self.lambda_head(h),
                "logits": self.logits_head(h),
            }


    class OptionalLongContextBlock(nn.Module):
        """Optional long-context block with Mamba if available, GRU fallback."""

        def __init__(self, hidden_dim: int):
            super().__init__()
            try:
                from mamba_ssm import Mamba  # type: ignore

                self.impl = Mamba(d_model=hidden_dim)
                self.is_mamba = True
            except Exception:
                self.impl = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
                self.is_mamba = False

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if self.is_mamba:
                return self.impl(x)
            out, _ = self.impl(x)
            return out


    class HeteroBlock(nn.Module):
        def __init__(self, hidden_dim: int, num_heads: int, dropout: float):
            super().__init__()
            self.span_attn = nn.MultiheadAttention(
                hidden_dim,
                num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.note_attn = nn.MultiheadAttention(
                hidden_dim,
                num_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.span_ln = nn.LayerNorm(hidden_dim)
            self.note_ln = nn.LayerNorm(hidden_dim)

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

            self.span_msg_proj = nn.Linear(hidden_dim, hidden_dim)
            self.note_msg_proj = nn.Linear(hidden_dim, hidden_dim)

        def _aggregate(self, adj: torch.Tensor, src_h: torch.Tensor) -> torch.Tensor:
            deg = adj.sum(dim=-1, keepdim=True).clamp(min=1.0)
            return adj @ src_h / deg

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
            span_self, _ = self.span_attn(
                query=span_h,
                key=span_h,
                value=span_h,
                key_padding_mask=~span_mask,
                need_weights=False,
            )
            note_self, _ = self.note_attn(
                query=note_h,
                key=note_h,
                value=note_h,
                key_padding_mask=~note_mask,
                need_weights=False,
            )

            span_from_note = self._aggregate(adj_ns.transpose(1, 2), note_h)
            note_from_span = self._aggregate(adj_ns, span_h)
            span_from_span = self._aggregate(adj_ss, span_h)
            note_from_note = self._aggregate(adj_nn, note_h)

            span_h = self.span_ln(span_h + span_self + self.span_msg_proj(span_from_note + span_from_span))
            note_h = self.note_ln(note_h + note_self + self.note_msg_proj(note_from_span + note_from_note))

            span_h = span_h + self.span_ffn(span_h)
            note_h = note_h + self.note_ffn(note_h)

            span_h = span_h * span_mask.unsqueeze(-1)
            note_h = note_h * note_mask.unsqueeze(-1)
            return span_h, note_h


    class FSNTGHeteroTransformer(nn.Module):
        def __init__(self, vocab_sizes: Dict[str, int], cfg: ModelConfig):
            super().__init__()
            self.cfg = cfg
            self.hidden_dim = cfg.hidden_dim
            self.parameterization = cfg.parameterization

            self.span_emb = nn.ModuleDict(
                {c: nn.Embedding(max(2, vocab_sizes[f"span.{c}"]), cfg.hidden_dim) for c in SPAN_CHANNELS}
            )
            self.note_emb = nn.ModuleDict(
                {c: nn.Embedding(max(2, vocab_sizes[f"note.{c}"]), cfg.hidden_dim) for c in NOTE_CHANNELS}
            )
            self.e_ns_emb = nn.Embedding(max(2, vocab_sizes["e_ns.template"]), cfg.hidden_dim)
            self.e_ss_emb = nn.Embedding(max(2, vocab_sizes["e_ss.relation"]), cfg.hidden_dim)
            self.aux_nn_emb = nn.Embedding(len(AUX_NOTE_RELATIONS), cfg.hidden_dim)

            self.time_emb = TimeEmbedding(cfg.hidden_dim)
            self.layers = nn.ModuleList(
                [HeteroBlock(cfg.hidden_dim, cfg.num_heads, cfg.dropout) for _ in range(cfg.num_layers)]
            )
            self.long_context = OptionalLongContextBlock(cfg.hidden_dim) if cfg.use_long_context_block else None

            self.span_heads = nn.ModuleDict(
                {
                    c: NodeCoordinateHead(cfg.hidden_dim, max(2, vocab_sizes[f"span.{c}"]))
                    for c in SPAN_CHANNELS
                }
            )
            self.note_heads = nn.ModuleDict(
                {
                    c: NodeCoordinateHead(cfg.hidden_dim, max(2, vocab_sizes[f"note.{c}"]))
                    for c in NOTE_CHANNELS
                }
            )
            self.e_ns_head = PairCoordinateHead(cfg.hidden_dim, max(2, vocab_sizes["e_ns.template"]))
            self.e_ss_head = PairCoordinateHead(cfg.hidden_dim, max(2, vocab_sizes["e_ss.relation"]))

        def forward(self, batch: dict, t, aux_note_rel=None):
            span_mask = batch["span_mask"]
            note_mask = batch["note_mask"]
            bsz, s = span_mask.shape
            _, n = note_mask.shape

            span_h = torch.zeros((bsz, s, self.hidden_dim), device=span_mask.device)
            note_h = torch.zeros((bsz, n, self.hidden_dim), device=span_mask.device)

            for c in SPAN_CHANNELS:
                span_h = span_h + self.span_emb[c](batch["span"][c])
            for c in NOTE_CHANNELS:
                note_h = note_h + self.note_emb[c](batch["note"][c])

            if not torch.is_tensor(t):
                t = torch.tensor(float(t), device=span_h.device)
            if t.dim() == 0:
                t = t.repeat(bsz)
            temb = self.time_emb(t)
            span_h = span_h + temb[:, None, :]
            note_h = note_h + temb[:, None, :]

            e_ns = batch["e_ns"]
            e_ss = batch["e_ss"]
            adj_ns = (e_ns != 0).to(torch.float32)
            adj_ss = (e_ss != 0).to(torch.float32)

            if aux_note_rel is None:
                aux_note_rel = reconstruct_aux_relations(batch)
            adj_nn = (aux_note_rel != 0).to(torch.float32)

            for layer in self.layers:
                span_h, note_h = layer(span_h, note_h, adj_ns, adj_ss, adj_nn, span_mask, note_mask)
                if self.long_context is not None:
                    span_h = span_h + self.long_context(span_h)

            outputs = {}
            for c in SPAN_CHANNELS:
                outputs[f"span.{c}"] = self.span_heads[c](span_h)
            for c in NOTE_CHANNELS:
                outputs[f"note.{c}"] = self.note_heads[c](note_h)

            pair_ns = note_h[:, :, None, :] + span_h[:, None, :, :] + self.e_ns_emb(e_ns)
            outputs["e_ns.template"] = self.e_ns_head(pair_ns)

            pair_ss = span_h[:, :, None, :] + span_h[:, None, :, :] + self.e_ss_emb(e_ss)
            outputs["e_ss.relation"] = self.e_ss_head(pair_ss)
            return outputs


    def reconstruct_aux_relations(batch: dict) -> torch.Tensor:
        """Deterministically reconstruct local note-note relation graph."""
        e_ns = batch["e_ns"]
        roles = batch["note"]["role"]
        active = batch["note"]["active"]
        bsz, n, _s = e_ns.shape

        host = (e_ns != 0).float().argmax(dim=-1)
        tpl = e_ns.max(dim=-1).values

        rel = torch.zeros((bsz, n, n), dtype=torch.long, device=e_ns.device)
        for b in range(bsz):
            for i in range(n):
                if int(active[b, i].item()) == 0:
                    continue
                for j in range(n):
                    if i == j or int(active[b, j].item()) == 0:
                        continue
                    if int(host[b, i]) == int(host[b, j]) and int(tpl[b, i]) == int(tpl[b, j]):
                        rel[b, i, j] = 1
                    if int(host[b, i]) == int(host[b, j]) and abs(int(tpl[b, i]) - int(tpl[b, j])) <= 1:
                        rel[b, i, j] = max(int(rel[b, i, j]), 2)
                    if int(roles[b, i]) == int(roles[b, j]) and int(tpl[b, i]) < int(tpl[b, j]):
                        rel[b, i, j] = max(int(rel[b, i, j]), 3)
        return rel


else:

    class FSNTGHeteroTransformer:  # type: ignore
        def __init__(self, *args, **kwargs):
            del args, kwargs
            raise RuntimeError("FSNTGHeteroTransformer requires torch")


    def reconstruct_aux_relations(batch: dict):  # type: ignore
        del batch
        raise RuntimeError("reconstruct_aux_relations requires torch")
