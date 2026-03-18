"""Rate matching, auxiliary denoising, and music-structure regularizers."""

from __future__ import annotations

from typing import Dict

from music_graph_dfm.diffusion.state_ops import COORD_ORDER


def _cross_entropy_logits(logits, target):
    import torch

    vocab = logits.shape[-1]
    flat_logits = logits.reshape(-1, vocab)
    flat_target = target.reshape(-1)
    return torch.nn.functional.cross_entropy(flat_logits, flat_target, reduction="mean")


def rate_matching_loss(
    outputs: Dict[str, dict],
    x_t: Dict[str, "torch.Tensor"],
    x_1: Dict[str, "torch.Tensor"],
    xt_is_x0: Dict[str, "torch.Tensor"],
    eta: Dict[str, float],
    eps: float = 1e-8,
):
    import torch

    total = torch.tensor(0.0, device=next(iter(x_t.values())).device)
    for coord in COORD_ORDER:
        out = outputs[coord]
        lam = torch.nn.functional.softplus(out["lambda"]).squeeze(-1)
        logits = out["logits"]
        pi = torch.softmax(logits, dim=-1)

        xt = x_t[coord]
        x1 = x_1[coord]
        indicator = xt_is_x0[coord].to(torch.float32)

        # Predicted rates r(v)=lambda*pi(v)
        r = lam.unsqueeze(-1) * pi

        target = torch.zeros_like(r)
        idx = x1.unsqueeze(-1)
        target.scatter_(-1, idx, indicator.unsqueeze(-1) * float(eta[coord]))

        xt_idx = xt.unsqueeze(-1)
        mask_not_current = torch.ones_like(target, dtype=torch.bool)
        mask_not_current.scatter_(-1, xt_idx, False)

        r_masked = r[mask_not_current]
        target_masked = target[mask_not_current]
        channel_loss = r_masked.sum() - (target_masked * torch.log(r_masked + eps)).sum()
        total = total + channel_loss / max(1, xt.numel())
    return total


def auxiliary_denoising_loss(
    outputs: Dict[str, dict],
    x_1: Dict[str, "torch.Tensor"],
):
    import torch

    losses = []
    for coord in COORD_ORDER:
        losses.append(_cross_entropy_logits(outputs[coord]["logits"], x_1[coord]))
    return torch.stack(losses).mean()


def host_uniqueness_regularizer(outputs: Dict[str, dict], x_t: Dict[str, "torch.Tensor"]) -> "torch.Tensor":
    import torch

    logits = outputs["e_ns.template"]["logits"]
    probs = torch.softmax(logits, dim=-1)
    p_non_none = 1.0 - probs[..., 0]
    lhs = p_non_none.sum(dim=-1)
    active = x_t["note.active"].to(torch.float32)
    return ((lhs - active) ** 2).mean()


def harmonic_compatibility_regularizer(
    outputs: Dict[str, dict],
    x_t: Dict[str, "torch.Tensor"],
    compat_table: "torch.Tensor" | None = None,
) -> "torch.Tensor":
    import torch

    if compat_table is None:
        return torch.tensor(0.0, device=next(iter(x_t.values())).device)

    pitch_probs = torch.softmax(outputs["note.pitch_token"]["logits"], dim=-1)

    # host via current argmax over non-none probability
    e_ns_probs = torch.softmax(outputs["e_ns.template"]["logits"], dim=-1)
    p_non_none = 1.0 - e_ns_probs[..., 0]
    host = p_non_none.argmax(dim=-1)

    key = x_t["span.key"].gather(1, host)
    harm = x_t["span.harm"].gather(1, host)

    key = key.clamp(min=0, max=compat_table.shape[0] - 1)
    harm = harm.clamp(min=0, max=compat_table.shape[1] - 1)

    b, n, z = pitch_probs.shape
    compat = torch.zeros((b, n, z), dtype=torch.float32, device=pitch_probs.device)
    for bi in range(b):
        compat[bi] = compat_table[key[bi], harm[bi]]

    incompat = (1.0 - compat) * pitch_probs
    return incompat.mean()


def duplicate_note_penalty(x_t: Dict[str, "torch.Tensor"]) -> "torch.Tensor":
    import torch

    active = x_t["note.active"]
    pitch = x_t["note.pitch_token"]
    role = x_t["note.role"]
    host = (x_t["e_ns.template"] != 0).float().argmax(dim=-1)
    tpl = x_t["e_ns.template"].max(dim=-1).values

    penalties = []
    for b in range(active.shape[0]):
        seen = {}
        dup = 0
        total = 0
        for i in range(active.shape[1]):
            if int(active[b, i].item()) == 0:
                continue
            key = (int(host[b, i].item()), int(tpl[b, i].item()), int(pitch[b, i].item()), int(role[b, i].item()))
            total += 1
            if key in seen:
                dup += 1
            seen[key] = 1
        penalties.append(float(dup) / max(1, total))
    return torch.tensor(penalties, device=active.device).mean()


def voice_leading_penalty(x_t: Dict[str, "torch.Tensor"]) -> "torch.Tensor":
    import torch

    active = x_t["note.active"]
    pitch = x_t["note.pitch_token"]
    role = x_t["note.role"]
    tpl = x_t["e_ns.template"].max(dim=-1).values
    penalties = []
    for b in range(active.shape[0]):
        total = 0
        bad = 0
        for r in torch.unique(role[b]).tolist():
            idx = [i for i in range(active.shape[1]) if int(active[b, i]) == 1 and int(role[b, i]) == int(r)]
            idx.sort(key=lambda i: int(tpl[b, i]))
            for i0, i1 in zip(idx[:-1], idx[1:]):
                total += 1
                if abs(int(pitch[b, i1]) - int(pitch[b, i0])) > 12:
                    bad += 1
        penalties.append(float(bad) / max(1, total))
    return torch.tensor(penalties, device=active.device).mean()


def repetition_consistency_penalty(x_t: Dict[str, "torch.Tensor"]) -> "torch.Tensor":
    import torch

    e_ss = x_t["e_ss.relation"]
    repeat_mask = e_ss == 2
    var_mask = e_ss == 3
    # Sparse relation prior regularizer.
    sparse = (e_ss != 0).float().mean()
    consistency = (repeat_mask.float().mean() + 0.5 * var_mask.float().mean())
    return sparse - consistency


def music_structure_loss(
    outputs: Dict[str, dict],
    x_t: Dict[str, "torch.Tensor"],
    compat_table: "torch.Tensor" | None = None,
):
    import torch

    host = host_uniqueness_regularizer(outputs, x_t)
    harm = harmonic_compatibility_regularizer(outputs, x_t, compat_table)
    dup = duplicate_note_penalty(x_t)
    vl = voice_leading_penalty(x_t)
    rep = repetition_consistency_penalty(x_t)
    total = host + harm + dup + vl + rep
    return {
        "total": total,
        "host": host,
        "harm": harm,
        "dup": dup,
        "voice_leading": vl,
        "repetition": rep,
    }
