"""
CV02 · Detección de objetos: piezas de la familia R-CNN (anchors, codificación de
cajas, RoI pooling y RPN) — soluciones.
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

    # BEGIN SOLUTION
    if all_combinations:
        pairs = [(s, r) for r in ratios for s in scales]
    else:
        pairs = [(s, ratios[0]) for s in scales] + [(scales[0], r) for r in ratios[1:]]
    s = torch.tensor([p[0] for p in pairs], dtype=torch.float32)
    r = torch.tensor([p[1] for p in pairs], dtype=torch.float32)
    return torch.stack([s * r.sqrt(), s / r.sqrt()], dim=1)
    # END SOLUTION


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

    # BEGIN SOLUTION
    shapes = anchor_shapes(scales, ratios, all_combinations)  # [A, 2]
    img_h, img_w = feat_h * stride, feat_w * stride
    half_w = shapes[:, 0] * img_w / 2
    half_h = shapes[:, 1] * img_h / 2
    base = torch.stack([-half_w, -half_h, half_w, half_h], dim=1)  # [A, 4]

    cy = (torch.arange(feat_h, dtype=torch.float32) + 0.5) * stride
    cx = (torch.arange(feat_w, dtype=torch.float32) + 0.5) * stride
    grid_y, grid_x = torch.meshgrid(cy, cx, indexing="ij")
    centers = torch.stack([grid_x, grid_y, grid_x, grid_y], dim=-1).reshape(-1, 1, 4)
    return (centers + base[None]).reshape(-1, 4)
    # END SOLUTION


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

    # BEGIN SOLUTION
    wx, wy, ww, wh = weights
    aw = reference[:, 2] - reference[:, 0]
    ah = reference[:, 3] - reference[:, 1]
    ax = reference[:, 0] + 0.5 * aw
    ay = reference[:, 1] + 0.5 * ah
    gw = targets[:, 2] - targets[:, 0]
    gh = targets[:, 3] - targets[:, 1]
    gx = targets[:, 0] + 0.5 * gw
    gy = targets[:, 1] + 0.5 * gh
    return torch.stack(
        [
            wx * (gx - ax) / aw,
            wy * (gy - ay) / ah,
            ww * torch.log(gw / aw),
            wh * torch.log(gh / ah),
        ],
        dim=1,
    )
    # END SOLUTION


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

    # BEGIN SOLUTION
    wx, wy, ww, wh = weights
    aw = reference[:, 2] - reference[:, 0]
    ah = reference[:, 3] - reference[:, 1]
    ax = reference[:, 0] + 0.5 * aw
    ay = reference[:, 1] + 0.5 * ah
    dx, dy = deltas[:, 0] / wx, deltas[:, 1] / wy
    dw = (deltas[:, 2] / ww).clamp(max=clip)
    dh = (deltas[:, 3] / wh).clamp(max=clip)
    cx, cy = dx * aw + ax, dy * ah + ay
    w, h = torch.exp(dw) * aw, torch.exp(dh) * ah
    return torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], dim=1)
    # END SOLUTION


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

    # BEGIN SOLUTION
    if mode not in ("max", "mean"):
        raise ValueError(f"mode debe ser 'max' o 'mean', no {mode!r}.")
    _, channels, height, width = features.shape
    out_h, out_w = output_size
    output = features.new_zeros(rois.shape[0], channels, out_h, out_w)

    def _round(v: float) -> int:  # redondeo "half away from zero" (como en C++)
        return int(math.floor(v + 0.5)) if v >= 0 else -int(math.floor(-v + 0.5))

    for k, roi in enumerate(rois.tolist()):
        n = int(roi[0])
        x_s, y_s = _round(roi[1] * spatial_scale), _round(roi[2] * spatial_scale)
        x_e, y_e = _round(roi[3] * spatial_scale), _round(roi[4] * spatial_scale)
        bin_h = max(y_e - y_s + 1, 1) / out_h
        bin_w = max(x_e - x_s + 1, 1) / out_w
        for p in range(out_h):
            h0 = min(max(math.floor(p * bin_h) + y_s, 0), height)
            h1 = min(max(math.ceil((p + 1) * bin_h) + y_s, 0), height)
            for q in range(out_w):
                w0 = min(max(math.floor(q * bin_w) + x_s, 0), width)
                w1 = min(max(math.ceil((q + 1) * bin_w) + x_s, 0), width)
                if h1 <= h0 or w1 <= w0:
                    continue
                cell = features[n, :, h0:h1, w0:w1].flatten(1)
                output[k, :, p, q] = cell.amax(1) if mode == "max" else cell.mean(1)
    return output
    # END SOLUTION


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

        # BEGIN SOLUTION
        self.conv = nn.Conv2d(in_channels, in_channels, 3, padding=1)
        self.cls_logits = nn.Conv2d(in_channels, num_anchors, 1)
        self.bbox_pred = nn.Conv2d(in_channels, num_anchors * 4, 1)
        # END SOLUTION

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

        # BEGIN SOLUTION
        t = torch.relu(self.conv(x))
        logits = self.cls_logits(t)  # [B, A, H, W]
        deltas = self.bbox_pred(t)  # [B, 4A, H, W]
        b, a, h, w = logits.shape
        logits = logits.permute(0, 2, 3, 1).reshape(b, h * w * a)
        deltas = deltas.view(b, a, 4, h, w).permute(0, 3, 4, 1, 2).reshape(b, -1, 4)
        return logits, deltas
        # END SOLUTION
