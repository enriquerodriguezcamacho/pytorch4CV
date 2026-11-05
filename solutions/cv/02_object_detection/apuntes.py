"""
CV02 · Ejercicios de código del tema 2 de los apuntes (sección 2.7) — soluciones.

Los ejercicios 4 (NMS) y 5 (mAP) están en ``boxes.py`` y ``metrics.py``, y la
generación de anchors del ejercicio 6, en ``rcnn.py``. Los teóricos están en
``docs/SOLUCIONES_APUNTES.md``.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn
from torchvision.ops import roi_pool as tv_roi_pool

from .boxes import box_convert, box_iou


class SingleObjectDetector(nn.Module):
    """
    Ej. 1 · Detector de un solo objeto: CNN + dos MLP (clase y coordenadas).

    Args del constructor:
        backbone: CNN que devuelve un vector. Salida: [batch, feature_dim].
        feature_dim: dimensión de las características.
        num_classes: número de clases, incluida la de fondo.
    """

    def __init__(self, backbone: nn.Module, feature_dim: int, num_classes: int) -> None:
        super().__init__()

        self.backbone = backbone
        # BEGIN SOLUTION define `cls_head` y `box_head` (dos capas lineales con ReLU)
        self.cls_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, num_classes),
        )
        self.box_head = nn.Sequential(
            nn.Linear(feature_dim, feature_dim), nn.ReLU(), nn.Linear(feature_dim, 4)
        )
        # END SOLUTION

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: imágenes. Dimensions: [batch, channels, height, width].

        Returns:
            Logits [batch, num_classes] y cajas [batch, 4].
        """

        # BEGIN SOLUTION
        features = self.backbone(x)
        return self.cls_head(features), self.box_head(features)
        # END SOLUTION


def single_object_loss(
    logits: torch.Tensor,
    boxes_pred: torch.Tensor,
    labels: torch.Tensor,
    boxes_true: torch.Tensor,
    background_class: int = 0,
    lam: float = 1.0,
) -> torch.Tensor:
    """
    Ej. 1 · ``L = L_CE + λ · 1_{no fondo} · L_L2``.

    ``L_L2`` es el MSE (media de las 4 coordenadas y de las muestras) calculado solo
    sobre las muestras que no son de fondo. Si todas son de fondo, vale 0.

    Args:
        logits: Dimensions: [batch, num_classes].
        boxes_pred: Dimensions: [batch, 4].
        labels: Dimensions: [batch].
        boxes_true: Dimensions: [batch, 4].
        background_class: índice de la clase de fondo.
        lam: peso de la regresión.

    Returns:
        Pérdida escalar.
    """

    # BEGIN SOLUTION
    loss = F.cross_entropy(logits, labels)
    foreground = labels != background_class
    if foreground.any():
        loss = loss + lam * F.mse_loss(boxes_pred[foreground], boxes_true[foreground])
    return loss
    # END SOLUTION


def random_proposals(height: int, width: int, c: int) -> torch.Tensor:
    """
    Ej. 2 · ``c`` cajas aleatorias ``xyxy`` centradas en cada píxel.

    Anchura y altura uniformes en ``[1, W]`` y ``[1, H]`` (``torch.rand``), con las
    cajas recortadas a la imagen. El coste crece como ``c·H·W``, lo que hace
    inviable clasificar cada propuesta.

    Args:
        height: alto de la imagen.
        width: ancho de la imagen.
        c: cajas por píxel.

    Returns:
        Cajas. Dimensions: [c * height * width, 4].
    """

    # BEGIN SOLUTION
    ys, xs = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
    cx = xs.flatten().float().repeat_interleave(c)
    cy = ys.flatten().float().repeat_interleave(c)
    box_w = 1 + torch.rand(cx.shape) * (width - 1)
    box_h = 1 + torch.rand(cy.shape) * (height - 1)
    return torch.stack(
        [
            (cx - box_w / 2).clamp(0, width),
            (cy - box_h / 2).clamp(0, height),
            (cx + box_w / 2).clamp(0, width),
            (cy + box_h / 2).clamp(0, height),
        ],
        dim=1,
    )
    # END SOLUTION


def iou(
    box1: torch.Tensor, box2: torch.Tensor, fmt: Literal["xyxy", "cxcywh"] = "xyxy"
) -> torch.Tensor:
    """
    Ej. 3 · IoU de dos cajas con un parámetro para elegir su representación.

    Args:
        box1: primera caja. Dimensions: [4].
        box2: segunda caja. Dimensions: [4].
        fmt: ``"xyxy"`` o ``"cxcywh"`` (centro + tamaño).

    Returns:
        IoU escalar en [0, 1].
    """

    # BEGIN SOLUTION
    first = box_convert(box1[None], fmt, "xyxy")
    second = box_convert(box2[None], fmt, "xyxy")
    return box_iou(first, second)[0, 0]
    # END SOLUTION


def count_anchors(height: int, width: int, n_scales: int, n_ratios: int) -> int:
    """
    Ej. 6 · Número de anchors usando solo combinaciones con ``s_1`` o ``r_1``.

    Args:
        height: alto de la imagen.
        width: ancho de la imagen.
        n_scales: número de escalas ``n``.
        n_ratios: número de ratios ``m``.

    Returns:
        ``(n + m - 1) · H · W``.
    """

    # BEGIN SOLUTION
    return (n_scales + n_ratios - 1) * height * width
    # END SOLUTION


def roi_pool_region(
    region: torch.Tensor,
    output_size: tuple[int, int],
    mode: Literal["max", "mean"] = "max",
) -> torch.Tensor:
    """
    Ej. 7 · RoI pooling de una región completa, como lo definen los apuntes.

    Cada celda mide ``ceil(H/h) × ceil(W/w)``. La celda ``(i, j)`` empieza en
    ``(floor(i·H/h), floor(j·W/w))`` y se recorta al borde, de modo que las celdas
    cercanas a los bordes pueden ser más pequeñas.

    Args:
        region: región. Dimensions: [C, H, W].
        output_size: tamaño de salida ``(h, w)``.
        mode: agregación, ``"max"`` o ``"mean"``.

    Returns:
        Región agregada. Dimensions: [C, h, w].

    Raises:
        ValueError: si ``mode`` no es válido.
    """

    # BEGIN SOLUTION
    if mode not in ("max", "mean"):
        raise ValueError(f"mode debe ser 'max' o 'mean', no {mode!r}.")
    channels, height, width = region.shape
    out_h, out_w = output_size
    cell_h, cell_w = math.ceil(height / out_h), math.ceil(width / out_w)
    output = region.new_zeros(channels, out_h, out_w)
    for i in range(out_h):
        top = math.floor(i * height / out_h)
        for j in range(out_w):
            left = math.floor(j * width / out_w)
            cell = region[:, top : top + cell_h, left : left + cell_w].flatten(1)
            output[:, i, j] = cell.amax(dim=1) if mode == "max" else cell.mean(dim=1)
    return output
    # END SOLUTION


class RCNN(nn.Module):
    """
    Ej. 8 · *Forward* de R-CNN: el cuello de botella es ejecutar la CNN una vez
    **por propuesta**.

    Args del constructor:
        backbone: CNN. Salida: [batch, feature_dim].
        feature_dim: dimensión de las características.
        num_classes: número de clases, incluida la de fondo.
        crop_size: lado al que se redimensiona cada propuesta.
    """

    def __init__(
        self,
        backbone: nn.Module,
        feature_dim: int,
        num_classes: int,
        crop_size: int = 32,
    ) -> None:
        super().__init__()

        self.backbone = backbone
        self.crop_size = crop_size
        self.cls_head = nn.Linear(feature_dim, num_classes)
        self.box_head = nn.Linear(feature_dim, 4)

    def forward(
        self, image: torch.Tensor, proposals: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Recorta cada propuesta, la redimensiona a ``crop_size`` (bilineal) y la pasa
        **por separado** por la CNN.

        Args:
            image: imagen. Dimensions: [channels, height, width].
            proposals: propuestas ``xyxy`` en píxeles. Dimensions: [K, 4].

        Returns:
            Logits [K, num_classes] y cajas [K, 4].
        """

        # BEGIN SOLUTION
        features = []
        for x1, y1, x2, y2 in proposals.long().tolist():
            crop = image[:, y1:y2, x1:x2].unsqueeze(0)
            crop = F.interpolate(
                crop, size=(self.crop_size, self.crop_size), mode="bilinear"
            )
            features.append(self.backbone(crop))
        stacked = torch.cat(features)
        return self.cls_head(stacked), self.box_head(stacked)
        # END SOLUTION


class FastRCNN(nn.Module):
    """
    Ej. 8 · *Forward* de Fast R-CNN: la CNN se aplica **una sola vez** a toda la imagen.

    Args del constructor:
        backbone: CNN. Salida: [batch, channels, H', W'].
        channels: canales del mapa de características.
        num_classes: número de clases, incluida la de fondo.
        output_size: lado del RoI pooling.
        spatial_scale: factor imagen → mapa de características.
    """

    def __init__(
        self,
        backbone: nn.Module,
        channels: int,
        num_classes: int,
        output_size: int = 2,
        spatial_scale: float = 1.0,
    ) -> None:
        super().__init__()

        self.backbone = backbone
        self.output_size = output_size
        self.spatial_scale = spatial_scale
        feature_dim = channels * output_size**2
        self.cls_head = nn.Linear(feature_dim, num_classes)
        self.box_head = nn.Linear(feature_dim, 4)

    def forward(
        self, image: torch.Tensor, proposals: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Calcula el mapa de características una vez, aplica ``torchvision.ops.roi_pool``
        a todas las propuestas y clasifica cada región.

        Args:
            image: imagen. Dimensions: [channels, height, width].
            proposals: propuestas ``xyxy`` en píxeles. Dimensions: [K, 4].

        Returns:
            Logits [K, num_classes] y cajas [K, 4].
        """

        # BEGIN SOLUTION
        feature_map = self.backbone(image.unsqueeze(0))
        pooled = tv_roi_pool(
            feature_map,
            [proposals.float()],
            output_size=self.output_size,
            spatial_scale=self.spatial_scale,
        ).flatten(1)
        return self.cls_head(pooled), self.box_head(pooled)
        # END SOLUTION
