"""
CV03 · Segmentación: métricas (pixel accuracy, IoU, mIoU, Dice) — soluciones.
"""

import torch


def confusion_matrix(
    pred: torch.Tensor,
    target: torch.Tensor,
    n_classes: int,
    ignore_index: int | None = None,
) -> torch.Tensor:
    """
    Matriz de confusión de segmentación, sin bucles (``torch.bincount``).

    ``cm[i, j]`` = número de píxeles cuya clase real es ``i`` y la predicha ``j``.

    Args:
        pred: clase predicha por píxel (``int64``). Dimensions: [batch, height, width].
        target: clase real por píxel (``int64``). Dimensions: [batch, height, width].
        n_classes: número de clases.
        ignore_index: valor de ``target`` que se excluye (o ``None``).

    Returns:
        Matriz ``int64``. Dimensions: [n_classes, n_classes].
    """

    # BEGIN SOLUTION usa target * n_classes + pred y torch.bincount
    pred, target = pred.flatten(), target.flatten()
    if ignore_index is not None:
        keep = target != ignore_index
        pred, target = pred[keep], target[keep]
    idx = target * n_classes + pred
    cm = torch.bincount(idx, minlength=n_classes**2)
    return cm.reshape(n_classes, n_classes)
    # END SOLUTION


def pixel_accuracy(
    pred: torch.Tensor, target: torch.Tensor, ignore_index: int | None = None
) -> torch.Tensor:
    """
    Fracción de píxeles bien clasificados.

    Args:
        pred: clase predicha por píxel. Dimensions: [batch, height, width].
        target: clase real por píxel. Dimensions: [batch, height, width].
        ignore_index: valor de ``target`` que se excluye (o ``None``).

    Returns:
        Accuracy en ``[0, 1]`` (escalar ``float``).
    """

    # BEGIN SOLUTION
    valid = torch.ones_like(target, dtype=torch.bool)
    if ignore_index is not None:
        valid = target != ignore_index
    correct = ((pred == target) & valid).sum()
    return correct.float() / valid.sum().clamp_min(1).float()
    # END SOLUTION


def iou_per_class(
    pred: torch.Tensor,
    target: torch.Tensor,
    n_classes: int,
    ignore_index: int | None = None,
) -> torch.Tensor:
    """
    IoU de cada clase: ``TP / (TP + FP + FN)``.

    Si una clase no aparece ni en ``pred`` ni en ``target`` (unión vacía) su IoU es
    ``nan`` (para que no cuente en la media).

    Args:
        pred: clase predicha por píxel. Dimensions: [batch, height, width].
        target: clase real por píxel. Dimensions: [batch, height, width].
        n_classes: número de clases.
        ignore_index: valor de ``target`` que se excluye (o ``None``).

    Returns:
        IoU por clase (``float``). Dimensions: [n_classes].
    """

    # BEGIN SOLUTION puedes reutilizar confusion_matrix
    cm = confusion_matrix(pred, target, n_classes, ignore_index).float()
    tp = cm.diagonal()
    union = cm.sum(0) + cm.sum(1) - tp
    iou = tp / union
    return torch.where(union > 0, iou, torch.full_like(iou, float("nan")))
    # END SOLUTION


def mean_iou(
    pred: torch.Tensor,
    target: torch.Tensor,
    n_classes: int,
    ignore_index: int | None = None,
) -> torch.Tensor:
    """
    mIoU: media de las IoU por clase ignorando las ``nan``.

    Args:
        pred: clase predicha por píxel. Dimensions: [batch, height, width].
        target: clase real por píxel. Dimensions: [batch, height, width].
        n_classes: número de clases.
        ignore_index: valor de ``target`` que se excluye (o ``None``).

    Returns:
        mIoU (escalar ``float``).
    """

    # BEGIN SOLUTION
    return torch.nanmean(iou_per_class(pred, target, n_classes, ignore_index))
    # END SOLUTION


def dice_coefficient(
    pred_mask: torch.Tensor, target_mask: torch.Tensor, eps: float = 1e-7
) -> torch.Tensor:
    """
    Coeficiente de Dice entre dos máscaras binarias: ``2|A ∩ B| / (|A| + |B|)``.

    Si las dos máscaras están vacías se devuelve 1 (coinciden perfectamente).

    Args:
        pred_mask: máscara predicha (``bool`` o 0/1). Dimensions: [...].
        target_mask: máscara real (``bool`` o 0/1). Mismas dimensiones.
        eps: constante numérica para el denominador.

    Returns:
        Dice en ``[0, 1]`` (escalar ``float``).
    """

    # BEGIN SOLUTION
    a, b = pred_mask.bool(), target_mask.bool()
    inter = (a & b).sum().float()
    total = (a.sum() + b.sum()).float()
    if total == 0:
        return torch.tensor(1.0)
    return 2 * inter / (total + eps)
    # END SOLUTION
