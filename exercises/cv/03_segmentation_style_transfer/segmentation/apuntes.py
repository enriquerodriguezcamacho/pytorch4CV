"""
CV03 · Ejercicios de código del tema 3 de los apuntes (sección 3.4) — ejercicios.

Los ejercicios 2, 3 y 9 están en ``unet.py``, y los 11-13 en ``../style_transfer/``.
Los teóricos están en ``docs/SOLUCIONES_APUNTES.md``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision.models import resnet18


def sliding_window_segmentation(
    classifier: nn.Module, image: torch.Tensor, patch_size: int
) -> torch.Tensor:
    """
    Ej. 1 · Segmentación *naive*: clasifica un parche centrado en cada píxel.

    La imagen se rellena con ceros (``patch_size // 2`` por lado) y se hace un
    *forward* por píxel: ``O(H·W·coste(P))``, intratable en tiempo real.

    Args:
        classifier: modelo. Entrada: [1, C, P, P]. Salida: [1, num_classes].
        image: imagen. Dimensions: [C, H, W].
        patch_size: lado ``P`` del parche (impar).

    Returns:
        Clase de cada píxel. Dimensions: [H, W].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def conv_transpose2d_general(
    x: torch.Tensor, kernel: torch.Tensor, stride: int = 1, padding: int = 0
) -> torch.Tensor:
    """
    Ej. 4 · Convolución transpuesta con stride, padding y varios canales, sin
    ``nn.ConvTranspose2d``.

    Args:
        x: entrada. Dimensions: [C_in, H, W].
        kernel: kernel, con el orden de ``ConvTranspose2d.weight``.
            Dimensions: [C_in, C_out, kh, kw].
        stride: paso ``s >= 1``.
        padding: recorte ``p >= 0`` en cada borde.

    Returns:
        Salida. Dimensions: [C_out, (H - 1)s + kh - 2p, (W - 1)s + kw - 2p].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def bed_of_nails_upsample(x: torch.Tensor, scale: int) -> torch.Tensor:
    """
    Ej. 7.1 · Cada valor pasa a ``(i·scale, j·scale)`` y el resto se rellena con ceros.

    Args:
        x: mapa. Dimensions: [C, H, W].
        scale: factor entero.

    Returns:
        Mapa ampliado. Dimensions: [C, H·scale, W·scale].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def nearest_neighbor_upsample(x: torch.Tensor, scale: int) -> torch.Tensor:
    """
    Ej. 7.2 · Cada valor se repite en un bloque de ``scale × scale``.

    Args:
        x: mapa. Dimensions: [C, H, W].
        scale: factor entero.

    Returns:
        Mapa ampliado. Dimensions: [C, H·scale, W·scale].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def bilinear_upsample(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """
    Ej. 7.3 · Interpolación bilineal desde cero, equivalente a
    ``F.interpolate(mode="bilinear", align_corners=False)``.

    La coordenada de salida ``i`` corresponde a ``(i + 0.5)·H/H' - 0.5`` en la
    entrada, recortada a ``[0, H - 1]`` (ídem para columnas).

    Args:
        x: mapa. Dimensions: [C, H, W].
        size: tamaño de salida ``(H', W')``.

    Returns:
        Mapa interpolado. Dimensions: [C, H', W'].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def max_unpool(
    x: torch.Tensor, indices: torch.Tensor, output_size: tuple[int, int]
) -> torch.Tensor:
    """
    Ej. 7.4 · Coloca cada valor donde estaba su máximo y rellena el resto con ceros.

    Args:
        x: valores tras el *max pooling*. Dimensions: [C, h, w].
        indices: índices de ``F.max_pool2d(return_indices=True)``, aplanados por
            canal. Dimensions: [C, h, w].
        output_size: tamaño de salida ``(H, W)``.

    Returns:
        Mapa reconstruido. Dimensions: [C, H, W].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


class FCNResNet18(nn.Module):
    """
    Ej. 8 · FCN sobre ResNet-18: backbone sin ``avgpool`` ni ``fc``, proyección 1×1 a
    ``num_classes`` y ``ConvTranspose2d(kernel=64, stride=32, padding=16)``, que
    multiplica por 32 el tamaño del mapa.

    Args del constructor:
        num_classes: número de clases ``N``.
        weights: pesos de torchvision (``None`` para no descargar nada).
    """

    def __init__(self, num_classes: int, weights: str | None = None) -> None:
        super().__init__()

        # TODO: define `backbone`, `classifier` y `upsample`
        raise NotImplementedError("TODO")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: imágenes. Dimensions: [batch, 3, H, W] (``H`` y ``W`` múltiplos de 32).

        Returns:
            Logits. Dimensions: [batch, num_classes, H, W].
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")


def train_segmentation_epoch(
    model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer
) -> float:
    """
    Ej. 10 · Una época de entrenamiento con *Cross-Entropy* por píxel.

    Args:
        model: modelo. Salida: [batch, num_classes, H, W].
        loader: pares ``(imagen, máscara)``, con máscaras [batch, H, W].
        optimizer: optimizador.

    Returns:
        Pérdida media de los *batches*.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")
