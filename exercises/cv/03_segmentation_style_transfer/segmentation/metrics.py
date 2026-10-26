"""
CV03 · Segmentación: métricas (pixel accuracy, IoU, mIoU, Dice) — ejercicios.
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

    # TODO: usa target * n_classes + pred y torch.bincount
    raise NotImplementedError("TODO")


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

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


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

    # TODO: puedes reutilizar confusion_matrix
    raise NotImplementedError("TODO")


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

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


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

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")
