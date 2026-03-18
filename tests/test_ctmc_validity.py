import pytest


torch = pytest.importorskip("torch", exc_type=ImportError)

from music_graph_dfm.samplers.ctmc_sampler import ctmc_jump_step


def test_ctmc_probabilities_validity():
    x_t = {
        "span.key": torch.zeros((1, 2), dtype=torch.long),
        "span.harm": torch.zeros((1, 2), dtype=torch.long),
        "span.meter": torch.zeros((1, 2), dtype=torch.long),
        "span.section": torch.zeros((1, 2), dtype=torch.long),
        "span.reg_center": torch.zeros((1, 2), dtype=torch.long),
        "e_ss.relation": torch.zeros((1, 2, 2), dtype=torch.long),
        "e_ns.template": torch.zeros((1, 3, 2), dtype=torch.long),
        "note.active": torch.zeros((1, 3), dtype=torch.long),
        "note.pitch_token": torch.zeros((1, 3), dtype=torch.long),
        "note.velocity": torch.zeros((1, 3), dtype=torch.long),
        "note.role": torch.zeros((1, 3), dtype=torch.long),
    }

    outputs = {}
    for k, xt in x_t.items():
        vocab = 4
        outputs[k] = {
            "lambda": torch.full((*xt.shape, 1), 0.1),
            "logits": torch.zeros((*xt.shape, vocab)),
        }

    x_next = ctmc_jump_step(x_t, outputs, h=0.05)
    for k in x_t:
        assert x_next[k].shape == x_t[k].shape
        assert (x_next[k] >= 0).all().item() is True
