"""Compatibility wrapper for CTMC sampler APIs."""

from music_graph_dfm.diffusion.ctmc import ctmc_jump_step, ctmc_sample

__all__ = ["ctmc_jump_step", "ctmc_sample"]
