"""
F11 · Arquitecturas: redes siamesas y multi-tarea — ejercicios.

Una red siamesa comparte **los mismos pesos** en sus dos ramas; una red
multi-tarea comparte el *backbone* y tiene una cabeza por tarea.
"""

import torch
import torch.nn.functional as F
from torch import nn


def make_embedding_net(in_channels: int, embedding_dim: int) -> nn.Sequential:
    """
    Backbone pequeño (ya dado): dos Conv 3×3 + ReLU + MaxPool, GAP y Linear.

    Entrada [batch, in_channels, H, W] → salida [batch, embedding_dim].
    """

    return nn.Sequential(
        nn.Conv2d(in_channels, 16, 3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(32, embedding_dim),
    )


class SiameseNetwork(nn.Module):
    """
    Red siamesa: una única ``self.embedding_net`` aplicada a las dos entradas.

    Atributo dado: ``self.embedding_net``.
    """

    def __init__(self, in_channels: int = 1, embedding_dim: int = 16) -> None:
        """
        Constructor.

        Args:
            in_channels: canales de las imágenes.
            embedding_dim: dimensión del embedding.
        """

        super().__init__()
        self.embedding_net = make_embedding_net(in_channels, embedding_dim)

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """
        Calcula el embedding normalizado (norma L2 = 1).

        Args:
            x: imágenes. Dimensions: [batch, in_channels, height, width].

        Returns:
            Embeddings unitarios. Dimensions: [batch, embedding_dim].
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Distancia euclídea entre los embeddings de cada pareja.

        Args:
            x1: primeras imágenes. Dimensions: [batch, in_channels, height, width].
            x2: segundas imágenes. Dimensions: [batch, in_channels, height, width].

        Returns:
            Distancias en [0, 2]. Dimensions: [batch].
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")


def contrastive_loss(
    distance: torch.Tensor, same: torch.Tensor, margin: float = 1.0
) -> torch.Tensor:
    """
    Pérdida contrastiva (Hadsell et al., 2006), promediada en el lote.

    ``L = y · d² + (1 - y) · max(0, margin - d)²`` con ``y = 1`` si la pareja es de la
    misma clase.

    Args:
        distance: distancias. Dimensions: [batch].
        same: 1 si la pareja es "igual", 0 si es "distinta". Dimensions: [batch].
        margin: margen para las parejas distintas.

    Returns:
        Escalar.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


class MultiTaskNet(nn.Module):
    """
    Red multi-tarea: backbone compartido + cabeza de clasificación + cabeza de
    regresión de caja (4 valores en [0, 1], como la localización de un solo objeto).

    Atributos: ``self.backbone`` (dado), ``self.cls_head`` (``nn.Linear(feat_dim,
    num_classes)``) y ``self.box_head`` (``nn.Sequential(nn.Linear(feat_dim, 4),
    nn.Sigmoid())``).
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 5, feat_dim: int = 32) -> None:
        """
        Constructor.

        Args:
            in_channels: canales de las imágenes.
            num_classes: número de clases.
            feat_dim: dimensión de las características compartidas.
        """

        super().__init__()
        self.backbone = make_embedding_net(in_channels, feat_dim)

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: imágenes. Dimensions: [batch, in_channels, height, width].

        Returns:
            Logits de clase y caja ``(cx, cy, w, h)`` normalizada. Dimensions:
            [batch, num_classes] y [batch, 4].
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")


def multitask_loss(
    logits: torch.Tensor,
    boxes: torch.Tensor,
    labels: torch.Tensor,
    target_boxes: torch.Tensor,
    box_weight: float = 1.0,
) -> torch.Tensor:
    """
    Suma ponderada de las pérdidas de cada tarea:
    ``CE(logits, labels) + box_weight · SmoothL1(boxes, target_boxes)``.

    Args:
        logits: logits. Dimensions: [batch, num_classes].
        boxes: cajas predichas. Dimensions: [batch, 4].
        labels: clases verdaderas (``int64``). Dimensions: [batch].
        target_boxes: cajas verdaderas. Dimensions: [batch, 4].
        box_weight: peso de la pérdida de regresión.

    Returns:
        Escalar.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")
