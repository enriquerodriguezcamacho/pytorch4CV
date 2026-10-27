"""
Tests de CV02 · Detección de objetos: anchors, codificación de cajas, RoI pooling y RPN.
"""

import math

import pytest
import torch
from helpers import assert_close, load
from torchvision import ops

r = load("cv/02_object_detection/rcnn.py")


def test_anchor_shapes_n_plus_m_minus_1() -> None:
    """n + m − 1 anchors, en el orden de los apuntes."""

    out = r.anchor_shapes([0.75, 0.5, 0.25], [1.0, 2.0, 0.5])
    assert out.shape == (5, 2), "Deben salir n + m − 1 = 5 anchors."
    sqrt2 = math.sqrt(2)
    expected = torch.tensor(
        [
            [0.75, 0.75],
            [0.5, 0.5],
            [0.25, 0.25],
            [0.75 * sqrt2, 0.75 / sqrt2],
            [0.75 / sqrt2, 0.75 * sqrt2],
        ]
    )
    assert_close(out, expected)


def test_anchor_shapes_all_combinations() -> None:
    out = r.anchor_shapes([0.5, 0.25], [1.0, 4.0], all_combinations=True)
    assert out.shape == (4, 2), "Producto cartesiano: n · m anchors."
    assert_close(out, torch.tensor([[0.5, 0.5], [0.25, 0.25], [1.0, 0.25], [0.5, 0.125]]))
    assert_close(out[:, 0] / out[:, 1], torch.tensor([1.0, 1.0, 4.0, 4.0]), atol=1e-5)


def test_generate_anchors_shape_and_centers() -> None:
    feat_h, feat_w, stride = 2, 3, 8
    out = r.generate_anchors(feat_h, feat_w, stride, [0.5, 0.25], [1.0, 2.0])
    assert out.shape == (feat_h * feat_w * 3, 4), "Forma incorrecta."
    centers = ((out[:, :2] + out[:, 2:]) / 2).view(feat_h, feat_w, 3, 2)
    assert_close(centers[0, 0, 0], torch.tensor([4.0, 4.0]))
    assert_close(centers[0, 2, 1], torch.tensor([20.0, 4.0]), atol=1e-4)  # fila 0, columna 2
    assert_close(centers[1, 0, 2], torch.tensor([4.0, 12.0]), atol=1e-4)  # fila 1, columna 0


def test_generate_anchors_sizes() -> None:
    """Anchor (s=0.5, r=1) en una imagen 16×24: 12 × 8 píxeles."""

    out = r.generate_anchors(2, 3, 8, [0.5], [1.0])
    wh = out[:, 2:] - out[:, :2]
    assert_close(wh, torch.tensor([[12.0, 8.0]]).expand(6, 2))


def test_encode_hand_values() -> None:
    reference = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    target = torch.tensor([[5.0, 0.0, 15.0, 20.0]])  # centro (10, 10), 10 × 20
    out = r.encode_boxes(reference, target)
    assert_close(out, torch.tensor([[0.5, 0.5, 0.0, math.log(2)]]))
    weighted = r.encode_boxes(reference, target, (10.0, 10.0, 5.0, 5.0))
    assert_close(weighted, torch.tensor([[5.0, 5.0, 0.0, 5 * math.log(2)]]))


def test_encode_decode_roundtrip() -> None:
    xy = torch.rand(10, 2) * 50
    reference = torch.cat([xy, xy + torch.rand(10, 2) * 30 + 2], dim=1)
    xy = torch.rand(10, 2) * 50
    targets = torch.cat([xy, xy + torch.rand(10, 2) * 30 + 2], dim=1)
    weights = (10.0, 10.0, 5.0, 5.0)
    deltas = r.encode_boxes(reference, targets, weights)
    assert_close(r.decode_boxes(reference, deltas, weights), targets, atol=1e-3)


def test_decode_clips_large_deltas() -> None:
    reference = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    out = r.decode_boxes(reference, torch.tensor([[0.0, 0.0, 100.0, 100.0]]))
    assert torch.isfinite(out).all(), "Recorta t_w y t_h antes del exp."
    assert_close(out[0, 2] - out[0, 0], torch.tensor(10.0 * 1000 / 16), atol=1e-2)


@pytest.mark.parametrize("spatial_scale", [1.0, 0.5, 0.25])
@pytest.mark.parametrize("output_size", [(2, 2), (3, 2)])
def test_roi_pool_matches_torchvision(spatial_scale: float, output_size: tuple[int, int]) -> None:
    features = torch.randn(2, 3, 12, 12)
    rois = torch.tensor(
        [
            [0, 0.0, 0.0, 20.0, 20.0],
            [1, 4.0, 2.0, 30.0, 18.0],
            [0, 10.0, 10.0, 11.0, 11.0],
            [1, -4.0, -2.0, 50.0, 60.0],
        ]
    )
    out = r.roi_pool(features, rois, output_size, spatial_scale)
    assert out.shape == (4, 3, *output_size), "Forma incorrecta."
    assert_close(out, ops.roi_pool(features, rois, output_size, spatial_scale))


def test_roi_pool_mean_hand_value() -> None:
    """Mapa 4×4 con valores 0..15, RoI completa y salida 2×2 con media."""

    features = torch.arange(16.0).view(1, 1, 4, 4)
    rois = torch.tensor([[0, 0.0, 0.0, 3.0, 3.0]])
    out = r.roi_pool(features, rois, (2, 2), mode="mean")
    assert_close(out, torch.tensor([[[[2.5, 4.5], [10.5, 12.5]]]]))
    with pytest.raises(ValueError):
        r.roi_pool(features, rois, (2, 2), mode="sum")


def test_rpn_head_shapes_and_order() -> None:
    """Salidas aplanadas por celda y luego por anchor (mismo orden que generate_anchors)."""

    head = r.RPNHead(in_channels=8, num_anchors=3)
    x = torch.randn(2, 8, 4, 5)
    logits, deltas = head(x)
    assert logits.shape == (2, 4 * 5 * 3), "Forma de los logits incorrecta."
    assert deltas.shape == (2, 4 * 5 * 3, 4), "Forma de los desplazamientos incorrecta."

    t = torch.relu(head.conv(x))
    raw_logits, raw_deltas = head.cls_logits(t), head.bbox_pred(t)
    # celda (fila 1, columna 2), anchor 2 → índice (1 * 5 + 2) * 3 + 2
    idx = (1 * 5 + 2) * 3 + 2
    assert_close(logits[:, idx], raw_logits[:, 2, 1, 2])
    assert_close(deltas[:, idx], raw_deltas[:, 2 * 4 : 3 * 4, 1, 2])
