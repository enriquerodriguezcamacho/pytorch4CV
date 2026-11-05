"""
Tests de los ejercicios del tema 2 de los apuntes.
"""

import pytest
import torch
import torch.nn.functional as F
from helpers import assert_close, load
from torch import nn

a = load("cv/02_object_detection/apuntes.py")
r = load("cv/02_object_detection/rcnn.py")


class CountingBackbone(nn.Module):
    """Cuenta las llamadas para comprobar el cuello de botella de R-CNN."""

    def __init__(self, net: nn.Module) -> None:
        super().__init__()
        self.net = net
        self.calls = 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        return self.net(x)


def _vector_backbone() -> CountingBackbone:
    return CountingBackbone(
        nn.Sequential(
            nn.Conv2d(3, 4, 3, padding=1), nn.AdaptiveAvgPool2d(1), nn.Flatten()
        )
    )


def test_single_object_detector_shapes() -> None:
    """Dos cabezas: clase y coordenadas."""

    model = a.SingleObjectDetector(_vector_backbone(), 4, 3)
    logits, boxes = model(torch.zeros(2, 3, 16, 16))
    assert logits.shape == (2, 3) and boxes.shape == (2, 4)


def test_single_object_loss_ignores_background_boxes() -> None:
    """Las cajas de fondo no afectan a la pérdida y ``λ`` escala solo la regresión."""

    logits = torch.tensor([[2.0, 0.5], [0.1, 1.0]])
    labels = torch.tensor([0, 1])
    pred = torch.tensor([[9.0, 9, 9, 9], [1.0, 1, 1, 1]])
    true = torch.zeros(2, 4)
    ce = F.cross_entropy(logits, labels)
    assert_close(a.single_object_loss(logits, pred, labels, true, lam=2.0), ce + 2.0)


def test_single_object_loss_all_background() -> None:
    """Si todo es fondo, solo queda la CE."""

    logits, labels = torch.randn(2, 3), torch.zeros(2, dtype=torch.long)
    loss = a.single_object_loss(logits, torch.ones(2, 4), labels, torch.zeros(2, 4))
    assert_close(loss, F.cross_entropy(logits, labels))


def test_random_proposals() -> None:
    """Salen ``c·H·W`` cajas válidas dentro de la imagen."""

    boxes = a.random_proposals(6, 8, 3)
    assert boxes.shape == (3 * 6 * 8, 4)
    assert (boxes[:, 2] >= boxes[:, 0]).all() and (boxes[:, 3] >= boxes[:, 1]).all()
    assert boxes[:, [0, 2]].max() <= 8 and boxes[:, [1, 3]].max() <= 6


def test_iou_both_formats_agree() -> None:
    """Las dos representaciones de las mismas cajas dan el mismo IoU: 1/7."""

    corner = a.iou(torch.tensor([0.0, 0, 2, 2]), torch.tensor([1.0, 1, 3, 3]))
    center = a.iou(
        torch.tensor([1.0, 1, 2, 2]), torch.tensor([2.0, 2, 2, 2]), fmt="cxcywh"
    )
    assert_close(corner, torch.tensor(1 / 7))
    assert_close(center, corner)


def test_count_anchors_matches_anchor_shapes() -> None:
    """Coincide con el número de anchors por píxel de ``anchor_shapes``."""

    per_pixel = r.anchor_shapes([0.75, 0.5, 0.25], [1.0, 2.0, 0.5]).shape[0]
    assert a.count_anchors(4, 5, 3, 3) == per_pixel * 20 == 5 * 20


def test_roi_pool_region_notes_example() -> None:
    """Región 3×3 a 2×2: celdas de 2×2 que se solapan en el centro."""

    region = torch.arange(9.0).view(1, 3, 3)
    expected = torch.tensor([[[4.0, 5.0], [7.0, 8.0]]])
    assert_close(a.roi_pool_region(region, (2, 2)), expected)


def test_roi_pool_region_mean_and_errors() -> None:
    """Modo media en el caso divisible y error si el modo no existe."""

    region = torch.arange(16.0).view(1, 4, 4)
    assert_close(a.roi_pool_region(region, (2, 2), "mean"), F.avg_pool2d(region, 2))
    with pytest.raises(ValueError):
        a.roi_pool_region(region, (2, 2), "median")


def test_rcnn_runs_backbone_per_proposal() -> None:
    """R-CNN ejecuta la CNN una vez por propuesta."""

    backbone = _vector_backbone()
    model = a.RCNN(backbone, 4, 3, crop_size=8)
    proposals = torch.tensor([[0.0, 0, 8, 8], [2.0, 2, 10, 12], [1.0, 3, 9, 9]])
    logits, boxes = model(torch.zeros(3, 16, 16), proposals)
    assert backbone.calls == 3
    assert logits.shape == (3, 3) and boxes.shape == (3, 4)


def test_fast_rcnn_runs_backbone_once() -> None:
    """Fast R-CNN ejecuta la CNN una sola vez."""

    backbone = CountingBackbone(nn.Conv2d(3, 4, 3, padding=1))
    model = a.FastRCNN(backbone, 4, 3)
    proposals = torch.tensor([[0.0, 0, 8, 8], [2.0, 2, 10, 12], [1.0, 3, 9, 9]])
    logits, boxes = model(torch.zeros(3, 16, 16), proposals)
    assert backbone.calls == 1
    assert logits.shape == (3, 3) and boxes.shape == (3, 4)
