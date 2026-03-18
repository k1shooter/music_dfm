"""Energy-style guidance as rate reweighting for CTMC transitions."""

from __future__ import annotations

from typing import Callable


def make_energy_guidance(
    energy_fn: Callable[[str, dict, "torch.Tensor"], "torch.Tensor"],
    strength: float = 1.0,
):
    """Returns a guidance callback compatible with `ctmc_jump_step`.

    `energy_fn(coord, x_t, candidate_values)` should return energies with same shape as
    `candidate_values` (without batch mask). Lower energy increases probability.
    """
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("make_energy_guidance requires torch") from exc

    def _guidance(coord: str, x_t: dict, lam: torch.Tensor, pi_off: torch.Tensor):
        vocab = pi_off.shape[-1]
        candidates = torch.arange(vocab, device=pi_off.device)
        view_shape = [1 for _ in range(pi_off.dim() - 1)] + [vocab]
        candidates = candidates.view(*view_shape).expand_as(pi_off)
        energies = energy_fn(coord, x_t, candidates)
        weights = torch.exp(-float(strength) * energies)
        guided = pi_off * weights
        guided = guided / guided.sum(dim=-1, keepdim=True).clamp(min=1e-8)
        return lam, guided

    return _guidance
