import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from music_graph_dfm.constants import COORD_ORDER
from music_graph_dfm.diffusion.losses import rate_matching_loss


def _dummy_inputs():
    x_t = {
        "span.key": torch.tensor([[0]]),
        "span.harm_root": torch.tensor([[0]]),
        "span.harm_quality": torch.tensor([[1]]),
        "span.meter": torch.tensor([[0]]),
        "span.section": torch.tensor([[0]]),
        "span.reg_center": torch.tensor([[0]]),
        "e_ss.relation": torch.tensor([[[0]]]),
        "note.host": torch.tensor([[1]]),
        "note.template": torch.tensor([[1]]),
        "note.active": torch.tensor([[1]]),
        "note.pitch_token": torch.tensor([[1]]),
        "note.velocity": torch.tensor([[1]]),
        "note.role": torch.tensor([[1]]),
    }
    x_1 = {k: v.clone() for k, v in x_t.items()}
    x_1["note.pitch_token"] = torch.tensor([[3]])

    outputs = {}
    for coord in COORD_ORDER:
        shape = x_t[coord].shape
        vocab = 6
        logits = torch.zeros((*shape, vocab), dtype=torch.float32)
        if coord == "note.pitch_token":
            logits[..., 0] = -2.0
            logits[..., 1] = -1.0
            logits[..., 2] = 2.5
            logits[..., 3] = 0.5
            logits[..., 4] = -0.5
            logits[..., 5] = -1.5
        outputs[coord] = {
            "lambda": torch.zeros((*shape, 1), dtype=torch.float32),
            "logits": logits,
        }

    xt_is_x0 = {k: torch.ones_like(v, dtype=torch.bool) for k, v in x_t.items()}
    eta = {k: 1.0 for k in COORD_ORDER}
    masks = {
        "span.key": torch.tensor([[True]]),
        "span.harm_root": torch.tensor([[True]]),
        "span.harm_quality": torch.tensor([[True]]),
        "span.meter": torch.tensor([[True]]),
        "span.section": torch.tensor([[True]]),
        "span.reg_center": torch.tensor([[True]]),
        "e_ss.relation": torch.tensor([[[True]]]),
        "note.host": torch.tensor([[True]]),
        "note.template": torch.tensor([[True]]),
        "note.active": torch.tensor([[True]]),
        "note.pitch_token": torch.tensor([[True]]),
        "note.velocity": torch.tensor([[True]]),
        "note.role": torch.tensor([[True]]),
    }

    kernel = torch.eye(6) * 0.2
    kernel[:, 2] += 0.8  # non-identity diffusion target mass
    kernel = kernel / kernel.sum(dim=-1, keepdim=True)
    return outputs, x_t, x_1, xt_is_x0, eta, masks, kernel


def test_graph_kernel_path_changes_rate_target():
    outputs, x_t, x_1, xt_is_x0, eta, masks, kernel = _dummy_inputs()

    mix_loss = rate_matching_loss(
        outputs,
        x_t,
        x_1,
        xt_is_x0,
        eta,
        masks,
        path_meta={"path_type": "mixture", "graph_kernels": {}},
    )
    gk_loss = rate_matching_loss(
        outputs,
        x_t,
        x_1,
        xt_is_x0,
        eta,
        masks,
        path_meta={"path_type": "graph_kernel", "graph_kernels": {"note.pitch_token": kernel}},
    )

    assert float(mix_loss.item()) != float(gk_loss.item())
