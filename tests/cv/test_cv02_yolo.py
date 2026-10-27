"""
Tests de CV02 · Detección de objetos: YOLOv1 (codificación, pérdida, decodificación y cabeza).
"""

import pytest
import torch
from helpers import assert_close, load

cfg = load("cv/02_object_detection/config.py")
y = load("cv/02_object_detection/yolo.py")

S, B, C = 3, 2, 4


def _config() -> "cfg.YOLOConfig":
    return cfg.YOLOConfig(S=S, B=B, C=C)


def test_config_cell_size() -> None:
    assert cfg.YOLOConfig().cell_size == 30, "5·2 + 20 = 30 (7×7×30 en el paper)."


def test_encode_targets_hand_values() -> None:
    """Centro (0.5, 0.8) con S=3 → fila 2, columna 1; x = 1.5 − 1, y = 2.4 − 2."""

    boxes = torch.tensor([[0.5, 0.8, 0.2, 0.4]])
    out = y.encode_targets(boxes, torch.tensor([3]), S, B, C)
    assert out.shape == (S, S, 5 * B + C), "Forma incorrecta."
    cell = out[2, 1]
    assert_close(cell[:C], torch.tensor([0.0, 0.0, 0.0, 1.0]))
    assert_close(cell[C : C + 5], torch.tensor([1.0, 0.5, 0.4, 0.2, 0.4]), atol=1e-5)
    assert (cell[C + 5 :] == 0).all(), "El segundo hueco queda a cero."
    assert out.sum() == pytest.approx(cell.sum().item()), "Las demás celdas quedan a cero."


def test_encode_targets_border_and_collision() -> None:
    """cx = 1 cae en la última columna; el segundo objeto de la misma celda se ignora."""

    boxes = torch.tensor([[1.0, 0.1, 0.1, 0.1], [0.95, 0.15, 0.3, 0.3]])
    out = y.encode_targets(boxes, torch.tensor([0, 1]), S, B, C)
    assert out[0, 2, 0] == 1 and out[0, 2, 1] == 0, "Se queda el primer objeto."
    assert out[0, 2, C] == 1


def _perfect_predictions(targets: torch.Tensor) -> torch.Tensor:
    """Predicciones idénticas al objetivo en el primer hueco (segundo hueco a cero)."""

    return targets.clone()


def _targets() -> torch.Tensor:
    t1 = y.encode_targets(torch.tensor([[0.5, 0.5, 0.3, 0.4]]), torch.tensor([1]), S, B, C)
    t2 = y.encode_targets(
        torch.tensor([[0.1, 0.1, 0.2, 0.2], [0.9, 0.9, 0.5, 0.1]]), torch.tensor([0, 2]), S, B, C
    )
    return torch.stack([t1, t2])


def test_yolo_loss_zero_for_perfect_predictions() -> None:
    targets = _targets()
    coord, obj, noobj, cls, total = y.YOLOLoss(_config())(_perfect_predictions(targets), targets)
    for name, value in zip(("coord", "obj", "noobj", "class", "total"), (coord, obj, noobj, cls, total)):
        assert value.dim() == 0, f"{name} debe ser un escalar."
        assert value.item() == pytest.approx(0.0, abs=1e-4), f"{name} debería ser 0."


def test_yolo_loss_accepts_flat_predictions() -> None:
    targets = _targets()
    preds = torch.rand_like(targets)
    loss = y.YOLOLoss(_config())
    assert_close(loss(preds.flatten(1), targets)[-1], loss(preds, targets)[-1])


def test_yolo_loss_noobj_hand_value() -> None:
    """Todas las confianzas a 1 sin objetos: λ_noobj · S·S·B / batch."""

    preds = torch.zeros(1, S, S, 5 * B + C)
    preds[..., C] = 1.0
    preds[..., C + 5] = 1.0
    targets = torch.zeros(1, S, S, 5 * B + C)
    coord, obj, noobj, cls, total = y.YOLOLoss(_config())(preds, targets)
    assert noobj.item() == pytest.approx(0.5 * S * S * B)
    assert coord.item() == obj.item() == cls.item() == 0.0, "Sin objetos solo hay noobj."
    assert total.item() == pytest.approx(noobj.item())


def test_yolo_loss_class_and_obj_hand_values() -> None:
    """Una celda con objeto: clases a cero (error 1) y confianza 0.5 (error 0.25)."""

    targets = y.encode_targets(torch.tensor([[0.5, 0.5, 0.3, 0.3]]), torch.tensor([2]), S, B, C)[None]
    preds = targets.clone()
    preds[..., :C] = 0.0
    preds[0, 1, 1, C] = 0.5
    _, obj, _, cls, _ = y.YOLOLoss(_config())(preds, targets)
    assert cls.item() == pytest.approx(1.0)
    assert obj.item() == pytest.approx(0.25)


def test_yolo_loss_responsible_box_is_highest_iou() -> None:
    """Si la segunda caja es la buena, la primera (mala) no penaliza en coord."""

    targets = y.encode_targets(torch.tensor([[0.5, 0.5, 0.3, 0.3]]), torch.tensor([0]), S, B, C)[None]
    preds = targets.clone()
    box = targets[0, 1, 1, C : C + 5].clone()
    preds[0, 1, 1, C + 5 : C + 10] = box  # segundo hueco perfecto
    preds[0, 1, 1, C : C + 5] = torch.tensor([1.0, 0.0, 0.0, 0.9, 0.9])  # primero malo
    loss = y.YOLOLoss(_config())
    best = loss._responsible_box(preds, targets)
    assert best[0, 1, 1].item() == 1, "El responsable es el de mayor IoU."
    coord, obj, *_ = loss(preds, targets)
    assert coord.item() == pytest.approx(0.0, abs=1e-4)
    assert obj.item() == pytest.approx(0.0, abs=1e-6)


def test_yolo_loss_coord_scaling_and_backward() -> None:
    targets = _targets()
    preds = torch.rand_like(targets, requires_grad=True)
    loss_5 = y.YOLOLoss(_config())(preds, targets)[0]
    loss_10 = y.YOLOLoss(cfg.YOLOConfig(S=S, B=B, C=C, lambda_coord=10.0))(preds, targets)[0]
    assert_close(loss_10, 2 * loss_5)
    loss_5.backward()
    assert preds.grad is not None and torch.isfinite(preds.grad).all()


def test_cells_to_boxes_inverts_encoding() -> None:
    """Codificar y decodificar recupera la caja original en píxeles."""

    boxes = torch.tensor([[0.5, 0.8, 0.2, 0.4]])
    target = y.encode_targets(boxes, torch.tensor([3]), S, B, C)
    out = y.cells_to_boxes(target[None], S, B, C, img_w=200, img_h=100)
    assert out.shape == (1, S * S * B, 6), "Forma incorrecta."
    row = out[0, (2 * S + 1) * B]  # celda (2, 1), caja 0
    assert_close(row, torch.tensor([3.0, 1.0, 80.0, 60.0, 120.0, 100.0]), atol=1e-4)


def test_cells_to_boxes_score_is_class_specific() -> None:
    preds = torch.zeros(1, S, S, 5 * B + C)
    preds[0, 0, 0, :C] = torch.tensor([0.1, 0.6, 0.2, 0.1])
    preds[0, 0, 0, C] = 0.5
    preds[0, 0, 0, C + 5] = 0.25
    out = y.cells_to_boxes(preds.flatten(1), S, B, C, 1.0, 1.0)
    assert_close(out[0, 0, :2], torch.tensor([1.0, 0.3]))
    assert_close(out[0, 1, :2], torch.tensor([1.0, 0.15]))


def test_yolo_head_shape() -> None:
    config = _config()
    head = y.YOLOv1Head(in_features=8 * 4 * 4, hidden=32, config=config)
    out = head(torch.randn(2, 8, 4, 4))
    assert out.shape == (2, S, S, config.cell_size), "Forma incorrecta."
    assert any(isinstance(mod, torch.nn.LeakyReLU) for mod in head.modules()), "Usa LeakyReLU."
