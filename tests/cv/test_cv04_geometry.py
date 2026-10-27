"""
Tests de CV04 · 3D y vídeo: convolución 3D, pérdidas de profundidad/normales y nubes de puntos.
"""

import pytest
import torch
import torch.nn.functional as F
from helpers import assert_close, load

g = load("cv/04_3d_video/geometry.py")


@pytest.mark.parametrize(
    ("size", "kernel", "stride", "padding", "dilation"),
    [(16, 3, 1, 0, 1), (16, 3, 2, 1, 1), (9, 5, 2, 2, 1), (10, 3, 1, 2, 2)],
)
def test_conv3d_output_size(size: int, kernel: int, stride: int, padding: int, dilation: int) -> None:
    conv = torch.nn.Conv3d(1, 1, kernel, stride, padding, dilation)
    expected = conv(torch.zeros(1, 1, size, size, size)).shape[-1]
    assert g.conv3d_output_size(size, kernel, stride, padding, dilation) == expected


def test_conv3d_single_channel_hand_value() -> None:
    """Kernel de unos 2×2×2 sobre un cubo 2×2×2 → suma de todo."""

    volume = torch.arange(8.0).view(2, 2, 2)
    out = g.conv3d_single_channel(volume, torch.ones(2, 2, 2))
    assert out.shape == (1, 1, 1), "Forma incorrecta."
    assert out.item() == 28.0


def test_conv3d_single_channel_matches_torch() -> None:
    volume, kernel = torch.randn(5, 6, 7), torch.randn(2, 3, 2)
    expected = F.conv3d(volume[None, None], kernel[None, None])[0, 0]
    assert_close(g.conv3d_single_channel(volume, kernel), expected, atol=1e-4)


@pytest.mark.parametrize(("stride", "padding"), [(1, 0), (2, 1), (1, 2)])
@pytest.mark.parametrize("use_bias", [True, False])
def test_conv3d_manual_matches_torch(stride: int, padding: int, use_bias: bool) -> None:
    x = torch.randn(2, 3, 5, 6, 6)
    weight = torch.randn(4, 3, 3, 3, 3)
    bias = torch.randn(4) if use_bias else None
    out = g.conv3d_manual(x, weight, bias, stride, padding)
    expected = F.conv3d(x, weight, bias, stride, padding)
    assert out.shape == expected.shape, "Forma incorrecta."
    assert_close(out, expected, atol=1e-4)


def test_scale_invariant_loss_is_scale_invariant() -> None:
    """Con λ = 1, L(c · ŷ, y) = L(ŷ, y)."""

    pred = torch.rand(2, 1, 4, 4) + 0.5
    target = torch.rand(2, 1, 4, 4) + 0.5
    base = g.scale_invariant_loss(pred, target)
    assert base.dim() == 0, "Debe ser un escalar."
    assert_close(g.scale_invariant_loss(3.0 * pred, target), base, atol=1e-5)
    assert g.scale_invariant_loss(3.0 * target, target).item() == pytest.approx(0.0, abs=1e-5)


def test_scale_invariant_loss_hand_values() -> None:
    """d = (log 2, 0): λ=0 → (log 2)² / 2; λ=1 → (log 2)²/2 − (log 2)²/4."""

    target = torch.ones(1, 1, 1, 2)
    pred = torch.tensor([[[[2.0, 1.0]]]])
    l2 = torch.log(torch.tensor(2.0)) ** 2
    assert_close(g.scale_invariant_loss(pred, target, lam=0.0), l2 / 2)
    assert_close(g.scale_invariant_loss(pred, target, lam=1.0), l2 / 4)
    assert_close(g.scale_invariant_loss(pred, target, lam=0.5), l2 / 2 - 0.5 * l2 / 4)


def test_scale_invariant_loss_per_image() -> None:
    """Se calcula por imagen: cada imagen tiene su propia escala."""

    target = torch.rand(2, 1, 3, 3) + 0.5
    pred = torch.stack([2.0 * target[0], 5.0 * target[1]])
    assert g.scale_invariant_loss(pred, target).item() == pytest.approx(0.0, abs=1e-5)


def test_surface_normal_loss() -> None:
    normals = torch.randn(2, 3, 4, 4)
    assert g.surface_normal_loss(normals, normals).item() == pytest.approx(0.0, abs=1e-6)
    assert g.surface_normal_loss(3 * normals, normals).item() == pytest.approx(0.0, abs=1e-6)
    assert g.surface_normal_loss(-normals, normals).item() == pytest.approx(2.0, abs=1e-5)
    up = torch.zeros(1, 3, 1, 2)
    up[:, 2] = 1.0
    side = torch.zeros(1, 3, 1, 2)
    side[:, 0] = 1.0
    assert g.surface_normal_loss(side, up).item() == pytest.approx(1.0)


def test_chamfer_distance_hand_value() -> None:
    """P = {0, (1,0,0)}, Q = {0}: (0 + 1)/2 + 0 = 0.5."""

    p = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    q = torch.zeros(1, 1, 3)
    out = g.chamfer_distance(p, q)
    assert out.shape == (1,), "Forma incorrecta: una distancia por pareja."
    assert_close(out, torch.tensor([0.5]))


def test_chamfer_distance_properties() -> None:
    p, q = torch.randn(3, 10, 3), torch.randn(3, 7, 3)
    assert_close(g.chamfer_distance(p, p), torch.zeros(3), atol=1e-5)
    assert_close(g.chamfer_distance(p, q), g.chamfer_distance(q, p), atol=1e-5)
    perm = torch.randperm(10)
    assert_close(g.chamfer_distance(p[:, perm], q), g.chamfer_distance(p, q), atol=1e-5)


def test_f1_at_delta() -> None:
    target = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]])
    pred = torch.tensor([[[0.05, 0.0, 0.0], [5.0, 0.0, 0.0], [6.0, 0.0, 0.0]]])
    # precisión 1/3, recall 1/2 → F1 = 2·(1/6)/(5/6) = 0.4
    assert_close(g.f1_at_delta(pred, target, 0.1), torch.tensor([0.4]))
    assert_close(g.f1_at_delta(target, target, 0.0), torch.tensor([1.0]))
    far = torch.full((1, 2, 3), 100.0)
    assert_close(g.f1_at_delta(far, target, 0.1), torch.tensor([0.0]))


def test_point_cloud_classifier_permutation_invariant() -> None:
    model = g.PointCloudClassifier(num_classes=4, hidden=16)
    points = torch.randn(2, 50, 3)
    out = model(points)
    assert out.shape == (2, 4), "Forma incorrecta."
    assert_close(model(points[:, torch.randperm(50)]), out, atol=1e-5)
    assert model(torch.randn(2, 13, 3)).shape == (2, 4), "Admite cualquier número de puntos."


def test_voxel_classifier() -> None:
    model = g.VoxelClassifier(num_classes=3, channels=4)
    out = model((torch.rand(2, 1, 8, 8, 8) > 0.5).float())
    assert out.shape == (2, 3), "Forma incorrecta."
    assert sum(isinstance(m, torch.nn.Conv3d) for m in model.modules()) == 2
