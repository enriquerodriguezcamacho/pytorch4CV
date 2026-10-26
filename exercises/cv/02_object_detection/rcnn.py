"""
CV02 · Detección de objetos: piezas de la familia R-CNN (anchors, codificación de
cajas, RoI pooling y RPN) — ejercicios.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn


def anchor_shapes(
    scales: list[float], ratios: list[float], all_combinations: bool = False
) -> torch.Tensor:
    """
    Anchura y altura relativas de los *anchors* de un píxel.

    Para una escala ``s`` y un *aspect ratio* ``r`` (= ancho / alto):
    ``w = s * sqrt(r)`` y ``h = s / sqrt(r)`` (relativas al tamaño de la imagen).

    * ``all_combinations=False`` (apuntes, d2l): solo las combinaciones que
      contienen ``scales[0]`` o ``ratios[0]``, en este orden:
      ``(s_1, r_1), (s_2, r_1), ..., (s_n, r_1), (s_1, r_2), ..., (s_1, r_m)``
      → ``n + m - 1`` *anchors*.
    * ``all_combinations=True`` (Faster R-CNN): producto cartesiano, con el
      ratio como índice externo: ``(s_1, r_1), (s_2, r_1), ..., (s_n, r_m)``
      → ``n · m`` *anchors*.

    Args:
        scales: escalas ``s`` en ``(0, 1]``.
        ratios: *aspect ratios* ``r > 0``.
        all_combinations: tipo de combinación.

    Returns:
        Tamaños relativos ``(w, h)``. Dimensions: [num_anchors, 2].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def generate_anchors(
    feat_h: int,
    feat_w: int,
    stride: int,
    scales: list[float],
    ratios: list[float],
    all_combinations: bool = False,
) -> torch.Tensor:
    """
    Genera los *anchors* centrados en cada celda de una rejilla ``feat_h × feat_w``.

    La imagen mide ``H = feat_h * stride`` por ``W = feat_w * stride`` píxeles. El
    centro de la celda ``(i, j)`` es ``((j + 0.5) * stride, (i + 0.5) * stride)``
    y cada *anchor* mide ``W * w_rel`` por ``H * h_rel`` (ver :func:`anchor_shapes`).

    Args:
        feat_h: alto del mapa de características.
        feat_w: ancho del mapa de características.
        stride: píxeles de imagen por celda.
        scales: escalas.
        ratios: *aspect ratios*.
        all_combinations: ver :func:`anchor_shapes`.

    Returns:
        *Anchors* en formato ``xyxy`` (píxeles), recorriendo las celdas por filas
        (``i`` externo, ``j`` interno) y, dentro de cada celda, los *anchors* en
        el orden de :func:`anchor_shapes`. Dimensions:
        [feat_h * feat_w * num_anchors, 4].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def encode_boxes(
    reference: torch.Tensor,
    targets: torch.Tensor,
    weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
) -> torch.Tensor:
    """
    Codifica cajas como desplazamientos respecto a *anchors*/propuestas (Faster R-CNN).

    Con ``(x_a, y_a, w_a, h_a)`` el centro y tamaño de la referencia y
    ``(x, y, w, h)`` los del objetivo:

    ``t_x = w_x (x - x_a) / w_a``, ``t_y = w_y (y - y_a) / h_a``,
    ``t_w = w_w log(w / w_a)``, ``t_h = w_h log(h / h_a)``.

    Args:
        reference: *anchors* o propuestas ``xyxy``. Dimensions: [N, 4].
        targets: cajas reales ``xyxy`` emparejadas. Dimensions: [N, 4].
        weights: pesos ``(w_x, w_y, w_w, w_h)``.

    Returns:
        Desplazamientos ``(t_x, t_y, t_w, t_h)``. Dimensions: [N, 4].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def decode_boxes(
    reference: torch.Tensor,
    deltas: torch.Tensor,
    weights: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0),
    clip: float = math.log(1000.0 / 16),
) -> torch.Tensor:
    """
    Operación inversa de :func:`encode_boxes`.

    ``x = t_x / w_x · w_a + x_a``, ``w = exp(t_w / w_w) · w_a`` (análogo para
    ``y`` y ``h``). Antes del ``exp`` se recorta ``t_w`` y ``t_h`` a ``clip``
    para evitar desbordamientos (como ``torchvision``).

    Args:
        reference: *anchors* o propuestas ``xyxy``. Dimensions: [N, 4].
        deltas: desplazamientos ``(t_x, t_y, t_w, t_h)``. Dimensions: [N, 4].
        weights: pesos ``(w_x, w_y, w_w, w_h)``.
        clip: valor máximo de ``t_w / w_w`` y ``t_h / w_h``.

    Returns:
        Cajas decodificadas ``xyxy``. Dimensions: [N, 4].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def roi_pool(
    features: torch.Tensor,
    rois: torch.Tensor,
    output_size: tuple[int, int],
    spatial_scale: float = 1.0,
    mode: Literal["max", "mean"] = "max",
) -> torch.Tensor:
    """
    RoI pooling (Fast R-CNN) con la misma cuantización que ``torchvision.ops.roi_pool``.

    Para cada región ``(n, x1, y1, x2, y2)`` (coordenadas de imagen):

    1. Se proyecta al mapa y se redondea: ``x_s = round(x1 · scale)``,
       ``x_e = round(x2 · scale)`` (igual para ``y``).
    2. Tamaño ``roi_w = max(x_e - x_s + 1, 1)`` (ídem ``roi_h``) y tamaño de celda
       ``bin_w = roi_w / out_w``.
    3. La celda ``(p, q)`` cubre las filas ``[floor(p·bin_h), ceil((p+1)·bin_h))`` y
       columnas ``[floor(q·bin_w), ceil((q+1)·bin_w))`` desplazadas por
       ``(y_s, x_s)`` y recortadas al mapa.
    4. Se agrega con máximo o media; una celda vacía vale 0.

    Args:
        features: mapa de características. Dimensions: [batch, C, H, W].
        rois: regiones ``(batch_index, x1, y1, x2, y2)``. Dimensions: [K, 5].
        output_size: tamaño de salida ``(out_h, out_w)``.
        spatial_scale: factor imagen → mapa (p.ej. 1/16 en VGG-16).
        mode: ``"max"`` o ``"mean"``.

    Returns:
        Regiones agregadas. Dimensions: [K, C, out_h, out_w].

    Raises:
        ValueError: si ``mode`` no es ``"max"`` ni ``"mean"``.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


class RPNHead(nn.Module):
    """
    Cabeza de la *Region Proposal Network* (Faster R-CNN).

    conv3×3 (mismos canales, padding 1) → ReLU y, a partir de ahí, dos conv1×1:
    ``A`` logits de *objectness* y ``4A`` desplazamientos (un bloque de 4 por
    *anchor*: canal ``a * 4 + k``).
    """

    def __init__(self, in_channels: int, num_anchors: int) -> None:
        """
        Constructor de la clase.

        Args:
            in_channels: canales del mapa de características.
            num_anchors: *anchors* por posición ``A``.
        """

        super().__init__()

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass, con las salidas aplanadas en el orden de :func:`generate_anchors`.

        Args:
            x: mapa de características. Dimensions: [batch, in_channels, H, W].

        Returns:
            *Objectness* (logits) y desplazamientos, ordenados por celda (por filas)
            y luego por *anchor*. Dimensions: [batch, H * W * A] y
            [batch, H * W * A, 4].
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")
