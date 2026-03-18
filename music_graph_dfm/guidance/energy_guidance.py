"""Optional guidance hooks for CTMC sampling."""

from __future__ import annotations


def scale_logits(outputs: dict, scale: float = 1.0) -> dict:
    out = {}
    for coord, val in outputs.items():
        out[coord] = {
            "lambda": val["lambda"],
            "logits": val["logits"] * scale,
        }
    return out


def host_constraint_guidance(outputs: dict, weight: float = 0.2) -> dict:
    """Boosts non-none note-span edge probabilities to encourage host assignment."""
    out = {k: {"lambda": v["lambda"], "logits": v["logits"].clone()} for k, v in outputs.items()}
    if "e_ns.template" in out:
        logits = out["e_ns.template"]["logits"]
        if logits.shape[-1] > 1:
            logits[..., 1:] += weight
    return out


def compose_guidance(*guiders):
    def _fn(outputs, x_t, t):
        del x_t, t
        out = outputs
        for g in guiders:
            out = g(out)
        return out

    return _fn
