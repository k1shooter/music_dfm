"""Discrete paths: mixture default and optional graph-kernel interpolation."""

from __future__ import annotations

from typing import Dict


def mixture_sample_tensor(x0, x1, kappa: float):
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("mixture_sample_tensor requires torch") from exc

    probs = torch.full_like(x1, fill_value=float(kappa), dtype=torch.float32)
    take_x1 = torch.bernoulli(probs).to(torch.bool)
    xt = torch.where(take_x1, x1, x0)
    xt_is_x0 = ~take_x1
    return xt, xt_is_x0


def graph_kernel_sample_tensor(x0, x1, kappa: float, kernel):
    """Samples xt from q_t ~= (1-kappa) delta_x0 + kappa K[x1,:]."""
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("graph_kernel_sample_tensor requires torch") from exc

    vocab = kernel.shape[0]
    one_hot_x0 = torch.nn.functional.one_hot(x0.clamp(min=0, max=vocab - 1), num_classes=vocab).to(torch.float32)
    k_rows = kernel[x1.clamp(min=0, max=vocab - 1)]
    probs = (1.0 - float(kappa)) * one_hot_x0 + float(kappa) * k_rows
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    xt = torch.distributions.Categorical(probs=probs).sample()
    xt_is_x0 = xt == x0
    return xt, xt_is_x0


def build_graph_kernel_from_edges(num_states: int, edges: Dict[int, list[int]]) -> list[list[float]]:
    """Builds a row-stochastic transition graph kernel prototype."""
    mat = [[0.0 for _ in range(num_states)] for _ in range(num_states)]
    for u in range(num_states):
        neigh = edges.get(u, [])
        if not neigh:
            mat[u][u] = 1.0
            continue
        p = 1.0 / len(neigh)
        for v in neigh:
            if 0 <= v < num_states:
                mat[u][v] += p
    return mat


def graph_kernel_path_row(
    x0: int,
    x1: int,
    t: float,
    kernel: list[list[float]],
    kappa: float,
) -> list[float]:
    """Approximate generalized discrete path row distribution.

    q_t(.|x0,x1) ~= (1-kappa) delta_x0 + kappa K[x1,:]
    """
    del t
    n = len(kernel)
    out = [0.0 for _ in range(n)]
    if 0 <= x0 < n:
        out[x0] += 1.0 - kappa
    if 0 <= x1 < n:
        row = kernel[x1]
        for i in range(n):
            out[i] += kappa * row[i]
    s = sum(out)
    if s <= 0:
        out[0] = 1.0
        return out
    return [v / s for v in out]


def sample_from_probs(probs: list[float], rng_state=None) -> int:
    import random

    rng = rng_state or random
    r = rng.random()
    c = 0.0
    for i, p in enumerate(probs):
        c += p
        if r <= c:
            return i
    return len(probs) - 1
