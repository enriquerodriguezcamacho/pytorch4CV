"""
CV02 · Detección de objetos: precision/recall, AP y mAP — soluciones.

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

    # BEGIN SOLUTION
    n = pred_boxes.shape[0]
    is_tp = torch.zeros(n, dtype=torch.bool)
    if n == 0 or gt_boxes.shape[0] == 0:
        return is_tp
    ious = box_iou(pred_boxes, gt_boxes)  # [N, M]
    used = torch.zeros(gt_boxes.shape[0], dtype=torch.bool)
    for idx in pred_scores.argsort(descending=True).tolist():
        candidates = ious[idx].masked_fill(used, -1.0)
        best_iou, best_gt = candidates.max(dim=0)
        if best_iou >= iou_threshold:
            is_tp[idx] = True
            used[best_gt] = True
    return is_tp
    # END SOLUTION


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

    # BEGIN SOLUTION
    order = scores.argsort(descending=True)
    tp = is_tp[order].float().cumsum(0)
    ranks = torch.arange(1, scores.numel() + 1, dtype=torch.float32)
    precision = tp / ranks
    recall = tp / max(num_gt, 1)
    return precision, recall
    # END SOLUTION


def average_precision(
    precision: torch.Tensor,
    recall: torch.Tensor,
    method: Literal["all_points", "11_points"] = "all_points",
) -> float:
    """
    Área bajo la curva precisión-*recall* (AP).

    * ``"all_points"`` (VOC 2010+; COCO usa 101 puntos): se añaden centinelas
      ``recall = [0, r..., 1]`` y ``precision = [0, p..., 0]``, se hace la
      envolvente monótona
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

    # BEGIN SOLUTION
    if method == "11_points":
        total = 0.0
        for t in torch.linspace(0, 1, 11):
            mask = recall >= t - 1e-9
            total += precision[mask].max().item() if mask.any() else 0.0
        return total / 11
    if method != "all_points":
        raise ValueError(f"Método desconocido: {method!r}.")
    r = torch.cat([torch.tensor([0.0]), recall.float(), torch.tensor([1.0])])
    p = torch.cat([torch.tensor([0.0]), precision.float(), torch.tensor([0.0])])
    p = p.flip(0).cummax(0).values.flip(0)  # envolvente monótona
    return float(((r[1:] - r[:-1]) * p[1:]).sum())
    # END SOLUTION


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

    # BEGIN SOLUTION
    classes = torch.cat([t["labels"] for t in targets]).unique().tolist()
    aps = []
    for c in classes:
        all_scores, all_tp, num_gt = [], [], 0
        for pred, target in zip(preds, targets):
            p_mask = pred["labels"] == c
            gt = target["boxes"][target["labels"] == c]
            num_gt += gt.shape[0]
            scores = pred["scores"][p_mask]
            all_scores.append(scores)
            all_tp.append(
                match_detections(pred["boxes"][p_mask], scores, gt, iou_threshold)
            )
        precision, recall = precision_recall_curve(
            torch.cat(all_scores), torch.cat(all_tp), num_gt
        )
        aps.append(average_precision(precision, recall, method))
    return sum(aps) / len(aps) if aps else 0.0
    # END SOLUTION


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

    # BEGIN SOLUTION
    thresholds = torch.linspace(0.5, 0.95, 10).tolist()
    return sum(mean_average_precision(preds, targets, t) for t in thresholds) / 10
    # END SOLUTION
