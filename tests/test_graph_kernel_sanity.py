import pytest

torch = pytest.importorskip("torch", exc_type=ImportError)

from music_graph_dfm.diffusion.paths import (
    graph_kernel_sample_tensor,
    graph_kernel_target_distribution,
    graph_kernel_target_rate_approximation,
)


def test_graph_kernel_distribution_is_normalized():
    kernel = torch.tensor(
        [
            [0.0, 2.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
        ],
        dtype=torch.float32,
    )
    x1 = torch.tensor([[0, 1, 2]], dtype=torch.long)
    target = graph_kernel_target_distribution(x1, kernel)
    assert target.shape == (1, 3, 3)
    assert torch.allclose(target.sum(dim=-1), torch.ones((1, 3)))


def test_graph_kernel_sample_shapes_and_rate_mask():
    kernel = torch.eye(5, dtype=torch.float32)
    x0 = torch.tensor([[1, 2, 3]], dtype=torch.long)
    x1 = torch.tensor([[2, 3, 4]], dtype=torch.long)

    xt, xt_is_x0 = graph_kernel_sample_tensor(x0=x0, x1=x1, kappa=0.5, kernel=kernel)
    assert xt.shape == x0.shape
    assert xt_is_x0.shape == x0.shape

    approx = graph_kernel_target_rate_approximation(x_t=xt, x1=x1, eta=1.0, kernel=kernel)
    for i in range(xt.shape[1]):
        assert float(approx[0, i, int(xt[0, i].item())].item()) == 0.0
