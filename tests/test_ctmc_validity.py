import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from music_graph_dfm.constants import COORD_ORDER
from music_graph_dfm.diffusion.ctmc import ctmc_jump_step


def test_ctmc_jump_excludes_current_state():
    batch = {
        "span_mask": torch.tensor([[True, True]]),
        "note_mask": torch.tensor([[True, True]]),
    }

    x_t = {
        "span.key": torch.ones((1, 2), dtype=torch.long),
        "span.harm_root": torch.ones((1, 2), dtype=torch.long),
        "span.harm_quality": torch.ones((1, 2), dtype=torch.long),
        "span.meter": torch.ones((1, 2), dtype=torch.long),
        "span.section": torch.ones((1, 2), dtype=torch.long),
        "span.reg_center": torch.ones((1, 2), dtype=torch.long),
        "e_ss.relation": torch.ones((1, 2, 2), dtype=torch.long),
        "note.host": torch.ones((1, 2), dtype=torch.long),
        "note.template": torch.ones((1, 2), dtype=torch.long),
        "note.active": torch.ones((1, 2), dtype=torch.long),
        "note.pitch_token": torch.ones((1, 2), dtype=torch.long),
        "note.velocity": torch.ones((1, 2), dtype=torch.long),
        "note.role": torch.ones((1, 2), dtype=torch.long),
    }

    outputs = {}
    for coord in COORD_ORDER:
        shape = x_t[coord].shape
        vocab = 4
        logits = torch.zeros((*shape, vocab), dtype=torch.float32)
        logits[..., 1] = 100.0  # strongly prefer current category if bug exists
        outputs[coord] = {
            "lambda": torch.full((*shape, 1), 80.0),  # jump probability ~1
            "logits": logits,
        }

    batch_full = {
        "span_mask": batch["span_mask"],
        "note_mask": batch["note_mask"],
    }

    x_next = ctmc_jump_step(x_t=x_t, outputs=outputs, h=1.0, batch=batch_full)
    for coord in COORD_ORDER:
        assert (x_next[coord] != 1).all().item() is True
