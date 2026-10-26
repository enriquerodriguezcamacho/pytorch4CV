"""
CV02 · Detección de objetos: precision/recall, AP y mAP — ejercicios.

Formato de las detecciones y del *ground truth* (igual que ``torchmetrics`` y los
modelos de ``torchvision.models.detection``): una lista con un diccionario por
imagen.

* ``preds[i] = {"boxes": [N_i, 4] xyxy, "scores": [N_i], "labels": [N_i]}``
* ``targets[i] = {"boxes": [M_i, 4] xyxy, "labels": [M_i]}``
"""

from __future__ import annotations

from typing import Literal

import torch

from .boxes import box_iou


def match_detections(
    pred_boxes: torch.Tensor,
    pred_scores: torch.Tensor,
    gt_boxes: torch.Tensor,
    iou_threshold: float = 0.5,
) -> torch.Tensor:
    """
    Marca cada predicción de **una imagen y una clase** como TP o FP.

    Se recorren las predicciones por *score* descendente; cada una se empareja
    con la caja real no usada de mayor IoU. Es TP si ese IoU es ``>=
    iou_threshold``; si no (o si todas las cajas reales ya se usaron), es FP.
    Cada caja real solo puede emparejarse una vez (las detecciones duplicadas son FP).

    Args:
        pred_boxes: cajas predichas ``xyxy``. Dimensions: [N, 4].
        pred_scores: confianzas. Dimensions: [N].
        gt_boxes: cajas reales ``xyxy``. Dimensions: [M, 4].
        iou_threshold: umbral de IoU para considerar acierto.

    Returns:
        Tensor booleano con ``True`` en los TP, en el **mismo orden** que
        ``pred_boxes``. Dimensions: [N].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def precision_recall_curve(
    scores: torch.Tensor, is_tp: torch.Tensor, num_gt: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Curva precisión-*recall* barriendo el umbral de confianza.

    Se ordenan las detecciones por *score* descendente y se acumulan TP y FP:
    ``precision_k = TP_k / k`` y ``recall_k = TP_k / num_gt``.

    Args:
        scores: confianzas de todas las detecciones de una clase. Dimensions: [N].
        is_tp: si cada detección es TP. Dimensions: [N].
        num_gt: número de objetos reales de la clase (``TP + FN``).

    Returns:
        ``(precision, recall)`` tras cada detección, en orden de *score*
        descendente. Dimensions: [N] y [N].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def average_precision(
    precision: torch.Tensor,
    recall: torch.Tensor,
    method: Literal["all_points", "11_points"] = "all_points",
) -> float:
    """
    Área bajo la curva precisión-*recall* (AP).

    * ``"all_points"`` (VOC 2010+, COCO usa 101 puntos): se añaden los extremos
      ``(r=0, p=1... )`` → en concreto ``recall = [0, r..., 1]`` y
      ``precision = [0, p..., 0]``, se hace la envolvente monótona
      ``p_interp(r) = max_{r' >= r} p(r')`` y se suma
      ``Σ (r_{k+1} - r_k) · p_interp(r_{k+1})``.
    * ``"11_points"`` (VOC 2007): media de ``p_interp(t)`` para
      ``t ∈ {0, 0.1, ..., 1}``, con ``p_interp(t) = max{p_k : r_k >= t}`` (0 si
      no hay ninguno).

    Args:
        precision: precisiones. Dimensions: [N].
        recall: *recalls* (no decrecientes). Dimensions: [N].
        method: tipo de interpolación.

    Returns:
        AP en [0, 1].

    Raises:
        ValueError: si ``method`` no es válido.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def mean_average_precision(
    preds: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
    iou_threshold: float = 0.5,
    method: Literal["all_points", "11_points"] = "all_points",
) -> float:
    """
    mAP con un umbral de IoU (p.ej. mAP@0.5 de PASCAL VOC).

    Para cada clase presente en el *ground truth*: emparejar por imagen con
    :func:`match_detections`, juntar las detecciones de todas las imágenes,
    calcular la curva con :func:`precision_recall_curve` y el AP con
    :func:`average_precision`. El mAP es la media de los AP de esas clases (las
    clases que solo aparecen en las predicciones no cuentan).

    Args:
        preds: detecciones por imagen (ver docstring del módulo).
        targets: *ground truth* por imagen (misma longitud que ``preds``).
        iou_threshold: umbral de IoU.
        method: interpolación del AP.

    Returns:
        mAP en [0, 1] (0.0 si no hay ninguna caja real).
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def coco_map(
    preds: list[dict[str, torch.Tensor]],
    targets: list[dict[str, torch.Tensor]],
) -> float:
    """
    mAP@[.5:.95] estilo COCO: media del mAP para IoU = 0.50, 0.55, ..., 0.95.

    Args:
        preds: detecciones por imagen.
        targets: *ground truth* por imagen.

    Returns:
        mAP medio sobre los 10 umbrales.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")
