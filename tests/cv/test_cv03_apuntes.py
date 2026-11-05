"""
Tests de los ejercicios del tema 3 de los apuntes.
"""

import pytest
import torch
import torch.nn.functional as F
from helpers import assert_close, load
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

a = load("cv/03_segmentation_style_transfer/segmentation/apuntes.py")


class CenterClassifier(nn.Module):
    """Clase 1 si el píxel central del parche es positivo; cuenta los *forward*."""

    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        center = x[:, 0, x.shape[2] // 2, x.shape[3] // 2]
        return torch.stack([-center, center], dim=1)


def test_sliding_window_labels_center_pixel() -> None:
    """Etiqueta cada píxel con el centro de su parche y hace H·W *forwards*."""

    image = torch.tensor([[[1.0, -1.0, 2.0], [-3.0, 4.0, -5.0]]])
    model = CenterClassifier()
    labels = a.sliding_window_segmentation(model, image, 3)
    assert torch.equal(labels, torch.tensor([[1, 0, 1], [0, 1, 0]]))
    assert model.calls == 6


@pytest.mark.parametrize("stride,padding", [(1, 0), (2, 0), (2, 1), (3, 2)])
def test_conv_transpose_general_matches_torch(stride: int, padding: int) -> None:
    """Coincide con ``F.conv_transpose2d`` en varios canales."""

    x, kernel = torch.randn(2, 4, 5), torch.randn(2, 3, 4, 4)
    expected = F.conv_transpose2d(
        x.unsqueeze(0), kernel, stride=stride, padding=padding
    )[0]
    assert_close(a.conv_transpose2d_general(x, kernel, stride, padding), expected)


def test_bed_of_nails_hand_values() -> None:
    """2×2 → 4×4 con ceros en las posiciones nuevas."""

    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])
    expected = torch.tensor(
        [[[1.0, 0, 2, 0], [0.0, 0, 0, 0], [3.0, 0, 4, 0], [0.0, 0, 0, 0]]]
    )
    assert_close(a.bed_of_nails_upsample(x, 2), expected)


def test_nearest_matches_interpolate() -> None:
    """Coincide con ``F.interpolate(mode="nearest")``."""

    x = torch.randn(3, 4, 5)
    expected = F.interpolate(x.unsqueeze(0), scale_factor=3, mode="nearest")[0]
    assert_close(a.nearest_neighbor_upsample(x, 3), expected)


@pytest.mark.parametrize("size", [(8, 10), (7, 13), (4, 5)])
def test_bilinear_matches_interpolate(size: tuple[int, int]) -> None:
    """Coincide con ``F.interpolate(mode="bilinear")``."""

    x = torch.randn(3, 4, 5)
    expected = F.interpolate(x.unsqueeze(0), size=size, mode="bilinear")[0]
    assert_close(a.bilinear_upsample(x, size), expected)


def test_max_unpool_matches_torch() -> None:
    """Coincide con ``F.max_unpool2d``."""

    x = torch.randn(1, 3, 6, 6)
    pooled, indices = F.max_pool2d(x, 2, return_indices=True)
    expected = F.max_unpool2d(pooled, indices, 2)[0]
    assert_close(a.max_unpool(pooled[0], indices[0], (6, 6)), expected)


def test_fcn_resnet18_recovers_input_size() -> None:
    """La salida tiene ``N`` canales y el tamaño de la entrada."""

    model = a.FCNResNet18(5).eval()
    assert model(torch.zeros(1, 3, 64, 96)).shape == (1, 5, 64, 96)


def test_train_segmentation_epoch_updates_model() -> None:
    """Devuelve la pérdida media, modifica los pesos y deja el modelo en ``train``."""

    model = nn.Conv2d(3, 2, 1)
    before = model.weight.data.clone()
    data = TensorDataset(torch.randn(4, 3, 4, 4), torch.randint(0, 2, (4, 4, 4)))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = a.train_segmentation_epoch(model, DataLoader(data, batch_size=2), optimizer)
    assert loss > 0 and model.training
    assert torch.norm(model.weight.data - before) > 0
