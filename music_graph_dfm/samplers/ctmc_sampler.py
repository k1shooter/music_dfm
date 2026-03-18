"""Always-valid CTMC sampler for factorized FSNTG coordinates."""

from __future__ import annotations

from typing import Dict

from music_graph_dfm.diffusion.state_ops import COORD_ORDER, coords_to_batch


def ctmc_jump_step(x_t: Dict[str, "torch.Tensor"], outputs: Dict[str, dict], h: float) -> Dict[str, "torch.Tensor"]:
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("ctmc_jump_step requires torch") from exc

    x_next: Dict[str, torch.Tensor] = {}
    for coord in COORD_ORDER:
        lam = torch.nn.functional.softplus(outputs[coord]["lambda"]).squeeze(-1)
        pi = torch.softmax(outputs[coord]["logits"], dim=-1)
        p_jump = 1.0 - torch.exp(-h * lam)
        jump = torch.bernoulli(p_jump).to(torch.bool)
        sampled = torch.distributions.Categorical(probs=pi).sample()
        x_next[coord] = torch.where(jump, sampled, x_t[coord])
    return x_next


def ctmc_sample(
    model,
    init_coords: Dict[str, "torch.Tensor"],
    base_batch: dict,
    num_steps: int,
    t_start: float = 1e-4,
    t_end: float = 0.999,
    guidance_fn=None,
):
    try:
        import torch
    except Exception as exc:
        raise RuntimeError("ctmc_sample requires torch") from exc

    x_t = {k: v.clone() for k, v in init_coords.items()}
    ts = torch.linspace(t_start, t_end, steps=num_steps, device=next(iter(x_t.values())).device)
    for i in range(num_steps - 1):
        t = ts[i]
        h = float(ts[i + 1] - ts[i])
        batch_xt = coords_to_batch(base_batch, x_t)
        outputs = model(batch_xt, t)
        if guidance_fn is not None:
            outputs = guidance_fn(outputs, x_t, t)
        x_t = ctmc_jump_step(x_t, outputs, h)
    return x_t
