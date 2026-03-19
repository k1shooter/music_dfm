"""CTMC reverse sampling with strict off-diagonal jump semantics."""

from __future__ import annotations

from typing import Callable, Dict

from music_graph_dfm.constants import COORD_ORDER
from music_graph_dfm.diffusion.masking import coordinate_masks, enforce_state_constraints
from music_graph_dfm.diffusion.state_ops import coords_to_batch


def _normalize_offdiag(pi, current):
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("_normalize_offdiag requires torch") from exc

    vocab = pi.shape[-1]
    current_oh = torch.nn.functional.one_hot(current.clamp(min=0, max=vocab - 1), num_classes=vocab).to(torch.float32)
    offdiag = pi * (1.0 - current_oh)
    denom = offdiag.sum(dim=-1, keepdim=True)

    probs = torch.where(denom > 1e-12, offdiag / denom.clamp(min=1e-12), torch.zeros_like(offdiag))
    has_offdiag_mass = denom.squeeze(-1) > 1e-12
    return probs, has_offdiag_mass


def ctmc_jump_step(
    x_t: Dict[str, "torch.Tensor"],
    outputs: Dict[str, dict],
    h: float,
    batch: dict,
    guidance_fn: Callable[[str, Dict[str, "torch.Tensor"], "torch.Tensor", "torch.Tensor"], tuple["torch.Tensor", "torch.Tensor"]] | None = None,
    debug_assertions: bool = False,
) -> Dict[str, "torch.Tensor"]:
    """One CTMC step: jump destinations exclude current state."""
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("ctmc_jump_step requires torch") from exc

    masks = coordinate_masks(batch)
    x_next: Dict[str, torch.Tensor] = {}

    for coord in COORD_ORDER:
        xt = x_t[coord]
        out = outputs[coord]
        lam = torch.nn.functional.softplus(out["lambda"]).squeeze(-1)
        pi = torch.softmax(out["logits"], dim=-1)
        pi_off, has_mass = _normalize_offdiag(pi, xt)

        if guidance_fn is not None:
            lam, pi_off = guidance_fn(coord, x_t, lam, pi_off)
            pi_guided = torch.clamp(pi_off, min=0.0)
            pi_off, guided_has_mass = _normalize_offdiag(pi_guided, xt)
            has_mass = has_mass & guided_has_mass

        if pi_off.shape[-1] <= 1:
            x_next[coord] = xt
            continue

        p_jump = 1.0 - torch.exp(-float(h) * lam)
        p_jump = torch.clamp(p_jump, min=0.0, max=1.0)
        mask = masks[coord]
        jump = torch.bernoulli(p_jump).to(torch.bool) & has_mass & mask

        vocab = pi_off.shape[-1]
        current_oh = torch.nn.functional.one_hot(xt.clamp(min=0, max=vocab - 1), num_classes=vocab).to(torch.float32)
        safe_probs = torch.where(has_mass.unsqueeze(-1), pi_off, current_oh)
        sample = torch.distributions.Categorical(probs=safe_probs).sample()
        updated = torch.where(jump, sample, xt)
        if debug_assertions and torch.any(jump):
            if bool(torch.any((updated == xt) & jump).item()):
                raise AssertionError(f"Off-diagonal CTMC violation on coord={coord}: jump landed on current state.")

        x_next[coord] = torch.where(mask, updated, xt)
        if debug_assertions:
            if bool(torch.any(x_next[coord][~mask] != xt[~mask]).item()):
                raise AssertionError(f"Masking violation on coord={coord}: masked coordinates changed.")

    x_next = enforce_state_constraints(x_next, batch)
    if debug_assertions:
        inactive = x_next["note.active"] == 0
        if bool(torch.any(x_next["note.host"][inactive] != 0).item()):
            raise AssertionError("Invariant violation: inactive notes must have host=0.")
        if bool(torch.any(x_next["note.template"][inactive] != 0).item()):
            raise AssertionError("Invariant violation: inactive notes must have template=0.")
    return x_next


def ctmc_sample(
    model,
    init_coords: Dict[str, "torch.Tensor"],
    base_batch: dict,
    num_steps: int,
    t_start: float = 1e-3,
    t_end: float = 0.999,
    guidance_fn=None,
    debug_assertions: bool = False,
):
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("ctmc_sample requires torch") from exc

    x_t = {k: v.clone() for k, v in init_coords.items()}
    ts = torch.linspace(t_start, t_end, steps=num_steps, device=next(iter(x_t.values())).device)

    for i in range(num_steps - 1):
        t = ts[i]
        h = float(ts[i + 1] - ts[i])
        batch_xt = coords_to_batch(base_batch, x_t)
        outputs = model(batch_xt, t)
        x_t = ctmc_jump_step(
            x_t,
            outputs,
            h,
            batch_xt,
            guidance_fn=guidance_fn,
            debug_assertions=debug_assertions,
        )

    return x_t
