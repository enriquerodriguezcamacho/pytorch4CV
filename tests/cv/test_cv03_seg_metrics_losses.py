"""
Tests de CV03 · Segmentación: métricas (accuracy, IoU, Dice) y pérdidas por píxel.
"""

import math

import pytest
import torch
import torch.nn.functional as F
from helpers import assert_close, load

metrics = load("cv/03_segmentation_style_transfer/segmentation/metrics.py")
losses = load("cv/03_segmentation_style_transfer/segmentation/losses.py")

IGNORE = 255
TARGET = torch.tensor([[[0, 0, 1], [1, 2, IGNORE]]])
PRED = torch.tensor([[[0, 1, 1], [1, 2, 2]]])
EXPECTED_CM = torch.tensor([[1, 1, 0], [0, 2, 0], [0, 0, 1]])
EXPECTED_ACC = 0.8
EXPECTED_IOU = torch.tensor([0.5, 2.0 / 3.0, 1.0])
EXPECTED_MIOU = (0.5 + 2.0 / 3.0 + 1.0) / 3.0


def test_confusion_matrix() -> None:
    """
    Matriz de confusión calculada a mano, ignorando el 255.
    """

    cm = metrics.confusion_matrix(PRED, TARGET, 3, ignore_index=IGNORE)
    assert cm.shape == (3, 3), "Forma incorrecta."
    assert torch.equal(cm.long(), EXPECTED_CM), "Valores incorrectos."
    assert cm.sum() == 5, "Solo cuentan los 5 píxeles válidos."


def test_pixel_accuracy() -> None:
    """
    4 de 5 píxeles válidos están bien; sin ignore_index se comparan los 6.
    """

    acc = metrics.pixel_accuracy(PRED, TARGET, ignore_index=IGNORE)
    assert torch.isclose(acc, torch.tensor(EXPECTED_ACC)), "Accuracy incorrecta."
    acc_all = metrics.pixel_accuracy(PRED, TARGET)
    assert torch.isclose(acc_all, torch.tensor(4 / 6)), "Sin ignore_index cuentan todos."


def test_iou_per_class_and_miou() -> None:
    """
    IoU por clase y mIoU calculados a mano.
    """

    iou = metrics.iou_per_class(PRED, TARGET, 3, ignore_index=IGNORE)
    assert iou.shape == (3,), "Debe haber una IoU por clase."
    assert_close(iou, EXPECTED_IOU)
    miou = metrics.mean_iou(PRED, TARGET, 3, ignore_index=IGNORE)
    assert math.isclose(miou.item(), EXPECTED_MIOU, rel_tol=1e-5), "mIoU incorrecta."


def test_iou_absent_class_is_nan() -> None:
    """
    Una clase que no aparece en pred ni en target da nan y no cuenta en la mIoU.
    """

    iou = metrics.iou_per_class(PRED, TARGET, 4, ignore_index=IGNORE)
    assert torch.isnan(iou[3]), "La clase ausente debe dar nan."
    miou = metrics.mean_iou(PRED, TARGET, 4, ignore_index=IGNORE)
    assert math.isclose(miou.item(), EXPECTED_MIOU, rel_tol=1e-5), "nan no debe contar."


def test_perfect_prediction() -> None:
    """
    Predicción perfecta: accuracy 1, mIoU 1, Dice 1.
    """

    y = torch.randint(0, 4, (2, 8, 8))
    assert torch.isclose(metrics.pixel_accuracy(y, y), torch.tensor(1.0)), "Acc != 1."
    assert torch.isclose(metrics.mean_iou(y, y, 4), torch.tensor(1.0)), "mIoU != 1."
    assert torch.isclose(metrics.dice_coefficient(y == 1, y == 1), torch.tensor(1.0)), (
        "Dice != 1."
    )


def test_dice_coefficient() -> None:
    """
    Dice = 2·1 / (2 + 2) = 0.5; máscaras vacías = 1; relación Dice = 2 IoU / (1 + IoU).
    """

    a = torch.tensor([1, 1, 0, 0])
    b = torch.tensor([1, 0, 1, 0])
    assert torch.isclose(metrics.dice_coefficient(a, b), torch.tensor(0.5)), "Dice."
    empty = torch.zeros(5, dtype=torch.bool)
    assert torch.isclose(metrics.dice_coefficient(empty, empty), torch.tensor(1.0)), (
        "Dos máscaras vacías coinciden perfectamente."
    )
    iou = 1 / 3
    assert math.isclose(0.5, 2 * iou / (1 + iou)), "Relación Dice-IoU."


@pytest.mark.parametrize("use_weights", [False, True])
def test_pixel_cross_entropy_matches_torch(use_weights: bool) -> None:
    """
    Coincide con ``F.cross_entropy`` con ignore_index y pesos de clase.
    """

    logits = torch.randn(2, 4, 5, 6)
    target = torch.randint(0, 4, (2, 5, 6))
    target[0, 0, :3] = IGNORE
    weights = torch.tensor([0.5, 1.0, 2.0, 3.0]) if use_weights else None
    out = losses.pixel_cross_entropy(logits, target, IGNORE, weights)
    expected = F.cross_entropy(logits, target, weight=weights, ignore_index=IGNORE)
    assert out.dim() == 0, "Debe devolver un escalar."
    assert_close(out, expected)


def test_pixel_cross_entropy_hand_value() -> None:
    """
    Logits nulos con 2 clases: -log(0.5) = log 2 en cada píxel.
    """

    logits = torch.zeros(1, 2, 2, 2)
    target = torch.tensor([[[0, 1], [IGNORE, 1]]])
    out = losses.pixel_cross_entropy(logits, target)
    assert torch.isclose(out, torch.tensor(math.log(2.0))), "Valor incorrecto."


def test_soft_dice_loss_hand_values() -> None:
    """
    Probabilidades 0.5: Dice por clase (1 + s) / (2 + s).
    """

    logits = torch.zeros(1, 2, 1, 2)
    target = torch.tensor([[[0, 1]]])
    out0 = losses.soft_dice_loss(logits, target, smooth=0.0)
    out1 = losses.soft_dice_loss(logits, target, smooth=1.0)
    assert torch.isclose(out0, torch.tensor(0.5)), "Dice loss sin suavizado."
    assert torch.isclose(out1, torch.tensor(1 / 3)), "Dice loss con smooth=1."


def test_soft_dice_loss_range_and_ignore() -> None:
    """
    Rango [0, 1], casi 0 con logits perfectos y los píxeles ignorados no cuentan.
    """

    target = torch.randint(0, 3, (2, 6, 6))
    perfect = 50.0 * F.one_hot(target, 3).permute(0, 3, 1, 2).float()
    assert losses.soft_dice_loss(perfect, target) < 1e-3, "Logits perfectos: ~0."
    rand = losses.soft_dice_loss(torch.randn(2, 3, 6, 6), target)
    assert 0.0 <= rand.item() <= 1.0, "La Dice loss está en [0, 1]."

    t_ign = target.clone()
    t_ign[:, 0] = IGNORE
    bad = perfect.clone()
    bad[:, :, 0] = -bad[:, :, 0]  # fila ignorada totalmente equivocada
    out = losses.soft_dice_loss(bad, t_ign, ignore_index=IGNORE)
    assert out < 1e-3, "Los píxeles ignorados no deben afectar."


def test_combined_loss_linear() -> None:
    """
    La pérdida combinada es lineal en sus pesos y tiene gradiente.
    """

    logits = torch.randn(2, 3, 4, 4, requires_grad=True)
    target = torch.randint(0, 3, (2, 4, 4))
    ce = losses.pixel_cross_entropy(logits, target)
    dice = losses.soft_dice_loss(logits, target, ignore_index=IGNORE)
    out = losses.CombinedLoss(ce_weight=0.3, dice_weight=2.0)(logits, target)
    assert_close(out, 0.3 * ce + 2.0 * dice)
    half = losses.CombinedLoss(ce_weight=0.15, dice_weight=1.0)(logits, target)
    assert_close(half, out / 2)
    out.backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0, "Sin gradiente."
