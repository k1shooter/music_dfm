import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from music_graph_dfm.constants import COORD_ORDER
from music_graph_dfm.diffusion.ctmc import _normalize_offdiag, ctmc_jump_step


def _base_state():
    x_t = {
        "span.key": torch.ones((1, 2), dtype=torch.long),
        "span.harm_root": torch.ones((1, 2), dtype=torch.long),
        "span.harm_quality": torch.ones((1, 2), dtype=torch.long),
        "span.meter": torch.ones((1, 2), dtype=torch.long),
        "span.section": torch.ones((1, 2), dtype=torch.long),
        "span.reg_center": torch.ones((1, 2), dtype=torch.long),
        "e_ss.relation": torch.ones((1, 2, 2), dtype=torch.long),
        "note.host": torch.ones((1, 3), dtype=torch.long),
        "note.template": torch.ones((1, 3), dtype=torch.long),
        "note.active": torch.ones((1, 3), dtype=torch.long),
        "note.pitch_token": torch.ones((1, 3), dtype=torch.long),
        "note.velocity": torch.ones((1, 3), dtype=torch.long),
        "note.role": torch.ones((1, 3), dtype=torch.long),
    }
    outputs = {}
    for coord in COORD_ORDER:
        shape = x_t[coord].shape
        logits = torch.zeros((*shape, 5), dtype=torch.float32)
        logits[..., 1] = 100.0
        outputs[coord] = {
            "lambda": torch.full((*shape, 1), 80.0),
            "logits": logits,
        }
    batch = {
        "span_mask": torch.tensor([[True, True]]),
        "note_mask": torch.tensor([[True, True, False]]),
    }
    return x_t, outputs, batch


def test_jump_destination_is_offdiagonal():
    x_t, outputs, batch = _base_state()
    x_next = ctmc_jump_step(x_t=x_t, outputs=outputs, h=1.0, batch=batch, debug_assertions=True)

    for coord in COORD_ORDER:
        mask = batch["span_mask"] if coord.startswith("span.") else batch["note_mask"]
        if coord == "e_ss.relation":
            mask = batch["span_mask"].unsqueeze(-1) & batch["span_mask"].unsqueeze(-2)
        assert (x_next[coord][mask] != x_t[coord][mask]).all().item() is True


def test_offdiag_distribution_normalization():
    pi = torch.tensor([[[0.0, 0.7, 0.3, 0.0]]], dtype=torch.float32)
    current = torch.tensor([[1]], dtype=torch.long)
    offdiag, has_mass = _normalize_offdiag(pi, current)

    assert has_mass.item() is True
    assert torch.allclose(offdiag.sum(dim=-1), torch.ones_like(offdiag.sum(dim=-1)))
    assert offdiag[..., 1].item() == 0.0


def test_masked_coordinates_always_stay():
    x_t, outputs, batch = _base_state()
    x_next = ctmc_jump_step(x_t=x_t, outputs=outputs, h=1.0, batch=batch)

    for coord in COORD_ORDER:
        if coord == "e_ss.relation":
            mask = batch["span_mask"].unsqueeze(-1) & batch["span_mask"].unsqueeze(-2)
            assert torch.equal(x_next[coord][~mask], x_t[coord][~mask])
            continue
        if coord.startswith("span."):
            mask = batch["span_mask"]
        else:
            mask = batch["note_mask"]
        assert torch.equal(x_next[coord][~mask], x_t[coord][~mask])


def test_inactive_notes_force_zero_host_template_after_step():
    x_t, outputs, batch = _base_state()
    x_t["note.active"][0, 1] = 0
    x_t["note.host"][0, 1] = 2
    x_t["note.template"][0, 1] = 4
    x_next = ctmc_jump_step(x_t=x_t, outputs=outputs, h=1.0, batch=batch, debug_assertions=True)
    assert int(x_next["note.host"][0, 1].item()) == 0
    assert int(x_next["note.template"][0, 1].item()) == 0


def test_degenerate_offdiag_mass_forces_stay():
    x_t, outputs, batch = _base_state()
    for coord in COORD_ORDER:
        logits = torch.full_like(outputs[coord]["logits"], fill_value=-100.0)
        logits[..., 1] = 100.0  # all mass on current category only
        outputs[coord]["logits"] = logits
        outputs[coord]["lambda"] = torch.full_like(outputs[coord]["lambda"], fill_value=80.0)

    x_next = ctmc_jump_step(x_t=x_t, outputs=outputs, h=1.0, batch=batch)
    for coord in COORD_ORDER:
        assert torch.equal(x_next[coord], x_t[coord])
