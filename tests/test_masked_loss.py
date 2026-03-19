import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from music_graph_dfm.constants import COORD_ORDER
from music_graph_dfm.diffusion.losses import auxiliary_denoising_loss


def _dummy_outputs_and_targets():
    batch = {
        "span_mask": torch.tensor([[True, False]]),
        "note_mask": torch.tensor([[True, False]]),
    }
    x1 = {
        "span.key": torch.tensor([[1, 3]]),
        "span.harm_root": torch.tensor([[2, 5]]),
        "span.harm_quality": torch.tensor([[1, 3]]),
        "span.harm_function": torch.tensor([[1, 3]]),
        "span.meter": torch.tensor([[4, 4]]),
        "span.section": torch.tensor([[0, 1]]),
        "span.reg_center": torch.tensor([[3, 7]]),
        "e_ss.relation": torch.tensor([[[0, 2], [1, 3]]]),
        "note.host": torch.tensor([[1, 0]]),
        "note.template": torch.tensor([[3, 0]]),
        "note.active": torch.tensor([[1, 0]]),
        "note.pitch_token": torch.tensor([[4, 9]]),
        "note.velocity": torch.tensor([[6, 12]]),
        "note.role": torch.tensor([[1, 2]]),
    }
    outputs = {}
    for coord in COORD_ORDER:
        shape = x1[coord].shape
        vocab = 16
        outputs[coord] = {
            "lambda": torch.zeros((*shape, 1), dtype=torch.float32),
            "logits": torch.zeros((*shape, vocab), dtype=torch.float32),
        }
    masks = {
        "span.key": batch["span_mask"],
        "span.harm_root": batch["span_mask"],
        "span.harm_quality": batch["span_mask"],
        "span.harm_function": batch["span_mask"],
        "span.meter": batch["span_mask"],
        "span.section": batch["span_mask"],
        "span.reg_center": batch["span_mask"],
        "e_ss.relation": batch["span_mask"].unsqueeze(-1) & batch["span_mask"].unsqueeze(-2),
        "note.host": batch["note_mask"],
        "note.template": batch["note_mask"],
        "note.active": batch["note_mask"],
        "note.pitch_token": batch["note_mask"],
        "note.velocity": batch["note_mask"],
        "note.role": batch["note_mask"],
    }
    return outputs, x1, masks


def test_padded_coordinates_do_not_affect_aux_loss():
    outputs, x1, masks = _dummy_outputs_and_targets()
    loss_a = auxiliary_denoising_loss(outputs, x1, masks)

    # Change only padded targets/logits.
    for coord in COORD_ORDER:
        x1[coord][..., -1] = 15 if x1[coord].dim() == 2 else x1[coord][..., -1]
        outputs[coord]["logits"][..., -1, :] = 100.0

    loss_b = auxiliary_denoising_loss(outputs, x1, masks)
    assert torch.allclose(loss_a, loss_b)
