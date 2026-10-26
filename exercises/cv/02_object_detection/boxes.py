"""
CV02 · Detección de objetos: operaciones con cajas (formatos, IoU, NMS) — ejercicios.

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

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def box_area(boxes: torch.Tensor) -> torch.Tensor:
    """
    Área de cajas en formato ``xyxy``.

    Args:
        boxes: cajas ``xyxy``. Dimensions: [..., 4].

    Returns:
        Áreas. Dimensions: [...].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


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

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


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

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


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

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


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

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")
