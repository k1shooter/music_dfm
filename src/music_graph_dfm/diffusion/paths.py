"""Coordinate-wise discrete forward paths."""

from __future__ import annotations


def mixture_sample_tensor(x0, x1, kappa: float):
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("mixture_sample_tensor requires torch") from exc

    probs = torch.full_like(x1, fill_value=float(kappa), dtype=torch.float32)
    take_x1 = torch.bernoulli(probs).to(torch.bool)
    xt = torch.where(take_x1, x1, x0)
    return xt, ~take_x1


def graph_kernel_sample_tensor(x0, x1, kappa: float, kernel):
    """Approximate generalized path q_t = (1-kappa)delta_x0 + kappa K[x1,:]."""
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("graph_kernel_sample_tensor requires torch") from exc

    vocab = kernel.shape[0]
    x0 = x0.clamp(min=0, max=vocab - 1)
    x1 = x1.clamp(min=0, max=vocab - 1)
    onehot = torch.nn.functional.one_hot(x0, num_classes=vocab).to(torch.float32)
    probs = (1.0 - float(kappa)) * onehot + float(kappa) * kernel[x1]
    probs = probs / probs.sum(dim=-1, keepdim=True).clamp(min=1e-8)
    xt = torch.distributions.Categorical(probs=probs).sample()
    return xt, (xt == x0)


def graph_kernel_target_distribution(x1, kernel):
    try:
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("graph_kernel_target_distribution requires torch") from exc

    vocab = kernel.shape[0]
    x1 = x1.clamp(min=0, max=vocab - 1)
    return kernel[x1]
