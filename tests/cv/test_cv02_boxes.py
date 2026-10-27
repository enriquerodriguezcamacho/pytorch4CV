"""
Tests de CV02 · Detección de objetos: formatos de cajas, IoU y NMS.
"""

import pytest
import torch
from helpers import assert_close, load
from torchvision import ops

b = load("cv/02_object_detection/boxes.py")


def _random_boxes(n: int, size: float = 100.0) -> torch.Tensor:
    """Cajas ``xyxy`` aleatorias válidas (x2 > x1, y2 > y1)."""

    xy = torch.rand(n, 2) * size
    wh = torch.rand(n, 2) * size / 2 + 1
    return torch.cat([xy, xy + wh], dim=1)


@pytest.mark.parametrize("out_fmt", ["xyxy", "xywh", "cxcywh"])
def test_box_convert_matches_torchvision(out_fmt: str) -> None:
    """Coincide con ``torchvision.ops.box_convert``."""

    boxes = _random_boxes(10)
    assert_close(b.box_convert(boxes, "xyxy", out_fmt), ops.box_convert(boxes, "xyxy", out_fmt))


def test_box_convert_hand_values() -> None:
    """(0, 0, 4, 2) xyxy → (2, 1, 4, 2) cxcywh → (0, 0, 4, 2) xywh."""

    box = torch.tensor([[0.0, 0.0, 4.0, 2.0]])
    cxcywh = b.box_convert(box, "xyxy", "cxcywh")
    assert_close(cxcywh, torch.tensor([[2.0, 1.0, 4.0, 2.0]]))
    assert_close(b.box_convert(cxcywh, "cxcywh", "xywh"), torch.tensor([[0.0, 0.0, 4.0, 2.0]]))


def test_box_convert_roundtrip_and_batch_dims() -> None:
    """Ida y vuelta devuelve lo mismo y funciona con dimensiones extra [..., 4]."""

    boxes = _random_boxes(12).view(3, 4, 4)
    for fmt in ("xywh", "cxcywh"):
        back = b.box_convert(b.box_convert(boxes, "xyxy", fmt), fmt, "xyxy")
        assert back.shape == (3, 4, 4), "Forma incorrecta."
        assert_close(back, boxes, atol=1e-4)


def test_box_convert_invalid_format() -> None:
    with pytest.raises(ValueError):
        b.box_convert(torch.zeros(1, 4), "xyxy", "yolo")


def test_box_area() -> None:
    boxes = _random_boxes(8)
    assert_close(b.box_area(boxes), ops.box_area(boxes))
    assert b.box_area(torch.tensor([0.0, 0.0, 3.0, 2.0])).item() == 6.0


def test_box_iou_matches_torchvision() -> None:
    """Matriz N×M igual que ``torchvision.ops.box_iou``."""

    boxes1, boxes2 = _random_boxes(5), _random_boxes(7)
    out = b.box_iou(boxes1, boxes2)
    assert out.shape == (5, 7), "Forma incorrecta: debe ser [N, M]."
    assert_close(out, ops.box_iou(boxes1, boxes2))


def test_box_iou_hand_values() -> None:
    """Dos cuadrados 2×2 desplazados 1: inter 2, unión 6 → 1/3. Disjuntas → 0."""

    a = torch.tensor([[0.0, 0.0, 2.0, 2.0]])
    other = torch.tensor([[1.0, 0.0, 3.0, 2.0], [5.0, 5.0, 6.0, 6.0], [0.0, 0.0, 2.0, 2.0]])
    assert_close(b.box_iou(a, other), torch.tensor([[1 / 3, 0.0, 1.0]]))


def test_iou_midpoint_elementwise() -> None:
    """Compara elemento a elemento (no todas contra todas) en formato ``cxcywh``."""

    boxes1, boxes2 = _random_boxes(6), _random_boxes(6)
    expected = ops.box_iou(boxes1, boxes2).diagonal()
    out = b.iou_midpoint(
        ops.box_convert(boxes1, "xyxy", "cxcywh").view(2, 3, 4),
        ops.box_convert(boxes2, "xyxy", "cxcywh").view(2, 3, 4),
    )
    assert out.shape == (2, 3), "Forma incorrecta: debe ser [...]."
    assert_close(out.flatten(), expected, atol=1e-4)


@pytest.mark.parametrize("iou_threshold", [0.1, 0.3, 0.5, 0.7])
def test_nms_matches_torchvision(iou_threshold: float) -> None:
    boxes = _random_boxes(40)
    scores = torch.rand(40)
    out = b.nms(boxes, scores, iou_threshold)
    assert out.dtype == torch.int64, "Devuelve índices int64."
    assert torch.equal(out, ops.nms(boxes, scores, iou_threshold))


def test_nms_hand_example() -> None:
    """La caja 1 solapa mucho con la 0 (mejor score) y se elimina; la 2 se queda."""

    boxes = torch.tensor(
        [[0.0, 0.0, 10.0, 10.0], [1.0, 1.0, 10.0, 10.0], [20.0, 20.0, 30.0, 30.0]]
    )
    scores = torch.tensor([0.9, 0.8, 0.7])
    assert b.nms(boxes, scores, 0.5).tolist() == [0, 2]
    assert b.nms(boxes, scores, 0.9).tolist() == [0, 1, 2], "IoU = 0.81 < 0.9: no se suprime."


def test_nms_strict_threshold() -> None:
    """Solo se suprime si IoU es estrictamente mayor que el umbral."""

    boxes = torch.tensor([[0.0, 0.0, 2.0, 2.0], [1.0, 0.0, 3.0, 2.0]])  # IoU = 1/3
    scores = torch.tensor([0.9, 0.8])
    assert b.nms(boxes, scores, 1 / 3).tolist() == [0, 1]


def test_batched_nms_matches_torchvision() -> None:
    boxes = _random_boxes(40)
    scores = torch.rand(40)
    labels = torch.randint(0, 3, (40,))
    out = b.batched_nms(boxes, scores, labels, 0.3)
    assert torch.equal(out, ops.batched_nms(boxes, scores, labels, 0.3))


def test_batched_nms_different_classes_not_suppressed() -> None:
    boxes = torch.tensor([[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]])
    scores = torch.tensor([0.9, 0.8])
    assert b.batched_nms(boxes, scores, torch.tensor([0, 1]), 0.5).tolist() == [0, 1]
    assert b.batched_nms(boxes, scores, torch.tensor([0, 0]), 0.5).tolist() == [0]


def test_batched_nms_empty() -> None:
    out = b.batched_nms(torch.zeros(0, 4), torch.zeros(0), torch.zeros(0, dtype=torch.int64), 0.5)
    assert out.numel() == 0 and out.dtype == torch.int64
