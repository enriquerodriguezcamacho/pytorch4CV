"""
CV02 · Detección de objetos: operaciones con cajas (formatos, IoU, NMS) — soluciones.

Convenciones:

* ``xyxy``: ``(x1, y1, x2, y2)``, esquina superior izquierda e inferior derecha.
* ``xywh``: ``(x1, y1, w, h)``, esquina superior izquierda + tamaño (formato COCO).
* ``cxcywh``: ``(cx, cy, w, h)``, centro + tamaño (formato YOLO).
"""

from __future__ import annotations

from typing import Literal

import torch

BoxFormat = Literal["xyxy", "xywh", "cxcywh"]
_FORMATS = ("xyxy", "xywh", "cxcywh")


def box_convert(boxes: torch.Tensor, in_fmt: BoxFormat, out_fmt: BoxFormat) -> torch.Tensor:
    """
    Convierte cajas entre los formatos ``xyxy``, ``xywh`` y ``cxcywh``.

    Pasa siempre por ``xyxy`` como formato intermedio.

    Args:
        boxes: cajas. Dimensions: [..., 4].
        in_fmt: formato de entrada.
        out_fmt: formato de salida.

    Returns:
        Cajas convertidas. Dimensions: [..., 4].

    Raises:
        ValueError: si algún formato no es válido.
    """

    # BEGIN SOLUTION
    if in_fmt not in _FORMATS or out_fmt not in _FORMATS:
        raise ValueError(f"Formato no soportado: {in_fmt} -> {out_fmt}.")
    a, b, c, d = boxes.unbind(-1)
    if in_fmt == "xywh":
        a, b, c, d = a, b, a + c, b + d
    elif in_fmt == "cxcywh":
        a, b, c, d = a - c / 2, b - d / 2, a + c / 2, b + d / 2
    # ahora (a, b, c, d) = (x1, y1, x2, y2)
    if out_fmt == "xywh":
        return torch.stack([a, b, c - a, d - b], dim=-1)
    if out_fmt == "cxcywh":
        return torch.stack([(a + c) / 2, (b + d) / 2, c - a, d - b], dim=-1)
    return torch.stack([a, b, c, d], dim=-1)
    # END SOLUTION


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """
    Área de cajas en formato ``xyxy``.

    Args:
        boxes: cajas ``xyxy``. Dimensions: [..., 4].

    Returns:
        Áreas. Dimensions: [...].
    """

    # BEGIN SOLUTION
    return (boxes[..., 2] - boxes[..., 0]) * (boxes[..., 3] - boxes[..., 1])
    # END SOLUTION


def box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    IoU de todas las parejas de cajas, vectorizado con *broadcasting* (sin bucles).

    Args:
        boxes1: cajas ``xyxy``. Dimensions: [N, 4].
        boxes2: cajas ``xyxy``. Dimensions: [M, 4].

    Returns:
        Matriz de IoU con ``out[i, j] = IoU(boxes1[i], boxes2[j])``.
        Dimensions: [N, M].
    """

    # BEGIN SOLUTION
    top_left = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    bottom_right = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]
    wh = (bottom_right - top_left).clamp(min=0)
    inter = wh[..., 0] * wh[..., 1]
    union = box_area(boxes1)[:, None] + box_area(boxes2)[None, :] - inter
    return inter / union
    # END SOLUTION


def iou_midpoint(
    boxes_preds: torch.Tensor, boxes_labels: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """
    IoU elemento a elemento de cajas en formato ``cxcywh`` (como en YOLO).

    A diferencia de :func:`box_iou`, aquí no se cruzan todas con todas: se
    compara ``boxes_preds[..., k, :]`` con ``boxes_labels[..., k, :]``.

    Args:
        boxes_preds: cajas predichas ``cxcywh``. Dimensions: [..., 4]
            (p.ej. [batch, S, S, 4]).
        boxes_labels: cajas reales ``cxcywh``. Dimensions: [..., 4].
        eps: constante para evitar dividir entre cero.

    Returns:
        IoU de cada pareja. Dimensions: [...] (p.ej. [batch, S, S]).
    """

    # BEGIN SOLUTION
    b1 = box_convert(boxes_preds, "cxcywh", "xyxy")
    b2 = box_convert(boxes_labels, "cxcywh", "xyxy")
    x1 = torch.max(b1[..., 0], b2[..., 0])
    y1 = torch.max(b1[..., 1], b2[..., 1])
    x2 = torch.min(b1[..., 2], b2[..., 2])
    y2 = torch.min(b1[..., 3], b2[..., 3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = box_area(b1).abs()
    area2 = box_area(b2).abs()
    return inter / (area1 + area2 - inter + eps)
    # END SOLUTION


def nms(boxes: torch.Tensor, scores: torch.Tensor, iou_threshold: float) -> torch.Tensor:
    """
    *Non-Maximum Suppression* desde cero.

    Algoritmo: ordenar por *score* descendente; tomar la mejor caja, guardarla y
    descartar las restantes cuyo IoU con ella sea **estrictamente mayor** que
    ``iou_threshold``; repetir con las que queden.

    Args:
        boxes: cajas ``xyxy``. Dimensions: [N, 4].
        scores: confianza de cada caja. Dimensions: [N].
        iou_threshold: umbral de solapamiento para suprimir.

    Returns:
        Índices (``int64``) de las cajas que se conservan, ordenados por *score*
        descendente. Dimensions: [K].
    """

    # BEGIN SOLUTION
    order = scores.argsort(descending=True)
    keep: list[int] = []
    while order.numel() > 0:
        best = order[0]
        keep.append(int(best))
        if order.numel() == 1:
            break
        ious = box_iou(boxes[best].unsqueeze(0), boxes[order[1:]]).squeeze(0)
        order = order[1:][ious <= iou_threshold]
    return torch.tensor(keep, dtype=torch.int64, device=boxes.device)
    # END SOLUTION


def batched_nms(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    """
    NMS independiente por clase: dos cajas de clases distintas nunca se suprimen.

    Truco habitual (el de ``torchvision``): desplazar las cajas de cada clase un
    *offset* ``label * (max_coord + 1)`` para que no se solapen entre clases y
    aplicar un único :func:`nms`.

    Args:
        boxes: cajas ``xyxy``. Dimensions: [N, 4].
        scores: confianzas. Dimensions: [N].
        labels: clase de cada caja (``int64``). Dimensions: [N].
        iou_threshold: umbral de solapamiento.

    Returns:
        Índices conservados, ordenados por *score* descendente. Dimensions: [K].
    """

    # BEGIN SOLUTION
    if boxes.numel() == 0:
        return torch.empty(0, dtype=torch.int64, device=boxes.device)
    offsets = labels.to(boxes) * (boxes.max() + 1)
    return nms(boxes + offsets[:, None], scores, iou_threshold)
    # END SOLUTION
