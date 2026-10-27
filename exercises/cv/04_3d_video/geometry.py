"""
CV04 · 3D y vídeo: convolución 3D, profundidad, normales y nubes de puntos — ejercicios.

Contiene las piezas "3D" del capítulo: la convolución 3D a mano, las pérdidas para
estimar profundidad (*scale-invariant*, la del parcial) y normales, la distancia de
Chamfer con su F1@δ y dos clasificadores sencillos (nubes de puntos y vóxeles).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

# --------------------------------------------------------------------------- #
# Convolución 3D                                                              #
# --------------------------------------------------------------------------- #


def conv3d_output_size(
    size: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1
) -> int:
    """
    Tamaño de salida de una dimensión de ``Conv3d``:
    ``floor((D + 2P - dilation · (K - 1) - 1) / S + 1)``.

    Args:
        size: tamaño de entrada ``D`` (profundidad, alto o ancho).
        kernel_size: tamaño del kernel ``K``.
        stride: paso ``S``.
        padding: relleno ``P`` a cada lado.
        dilation: dilatación.

    Returns:
        Tamaño de salida.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def conv3d_single_channel(volume: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """
    Convolución 3D de **un canal**, stride 1 y sin padding, con tres bucles sobre
    la salida: ``O[d, h, w] = Σ_{i,j,k} I[d+i, h+j, w+k] · K[i, j, k]``.

    (Como en PyTorch, es en realidad una correlación cruzada: el kernel no se voltea.)

    Args:
        volume: volumen de entrada. Dimensions: [depth, height, width].
        kernel: kernel. Dimensions: [k_d, k_h, k_w].

    Returns:
        Volumen de salida. Dimensions:
        [depth - k_d + 1, height - k_h + 1, width - k_w + 1].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def conv3d_manual(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int = 1,
    padding: int = 0,
) -> torch.Tensor:
    """
    ``Conv3d`` multicanal desde cero (debe coincidir con ``F.conv3d``).

    Se aplica el padding con ``F.pad`` y se recorre la salida con bucles sobre
    ``(d, h, w)``. Para cada posición, el bloque de entrada
    ``[batch, C_in, k_d, k_h, k_w]`` se multiplica por todos los filtros a la vez
    (``einsum`` o *broadcasting*) para obtener los ``C_out`` canales.

    Args:
        x: entrada. Dimensions: [batch, in_channels, depth, height, width].
        weight: filtros. Dimensions: [out_channels, in_channels, k_d, k_h, k_w].
        bias: sesgo opcional. Dimensions: [out_channels].
        stride: paso (igual en las tres dimensiones).
        padding: relleno (igual en las tres dimensiones).

    Returns:
        Salida. Dimensions: [batch, out_channels, D_out, H_out, W_out].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


# --------------------------------------------------------------------------- #
# Profundidad y normales                                                      #
# --------------------------------------------------------------------------- #


def scale_invariant_loss(
    pred: torch.Tensor, target: torch.Tensor, lam: float = 1.0, eps: float = 1e-8
) -> torch.Tensor:
    """
    Pérdida *scale-invariant* para profundidad (Eigen et al., 2014). Con
    ``d = log ŷ - log y`` y ``n = H · W`` píxeles **por imagen**:

    ``L = (1/n) Σ d² - (λ/n²) (Σ d)²``

    calculada por imagen y promediada sobre el lote. Con ``λ = 1`` multiplicar la
    predicción por una constante no cambia la pérdida.

    Args:
        pred: profundidad predicha (positiva). Dimensions: [batch, 1, height, width].
        target: profundidad real (positiva). Dimensions: igual que ``pred``.
        lam: peso ``λ`` del término de escala (0 → MSE en log).
        eps: constante que se suma antes del ``log``.

    Returns:
        Escalar.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def surface_normal_loss(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-8
) -> torch.Tensor:
    """
    Pérdida de normales: ``L = mean(1 - cos(ŷ, y))`` sobre todos los píxeles, con
    el coseno calculado en la dimensión de canales (las 3 componentes).

    Args:
        pred: normales predichas (sin normalizar). Dimensions: [batch, 3, height, width].
        target: normales reales. Dimensions: igual que ``pred``.
        eps: constante de ``F.cosine_similarity``.

    Returns:
        Escalar en [0, 2].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


# --------------------------------------------------------------------------- #
# Nubes de puntos                                                             #
# --------------------------------------------------------------------------- #


def chamfer_distance(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """
    Distancia de Chamfer (con distancias **al cuadrado**), vectorizada con
    ``torch.cdist``:

    ``d(P, Q) = (1/|P|) Σ_p min_q ||p - q||² + (1/|Q|) Σ_q min_p ||q - p||²``.

    Args:
        p: nubes de puntos. Dimensions: [batch, N, 3].
        q: nubes de puntos. Dimensions: [batch, M, 3].

    Returns:
        Distancia de cada pareja. Dimensions: [batch].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def f1_at_delta(pred: torch.Tensor, target: torch.Tensor, delta: float) -> torch.Tensor:
    """
    F1@δ entre dos nubes de puntos (distancia euclídea **sin** cuadrado):

    * precisión: fracción de puntos predichos a distancia ``<= δ`` de algún punto real;
    * *recall*: fracción de puntos reales a distancia ``<= δ`` de algún punto predicho;
    * ``F1 = 2PR / (P + R)`` (0 si ``P + R = 0``).

    Args:
        pred: nubes predichas. Dimensions: [batch, N, 3].
        target: nubes reales. Dimensions: [batch, M, 3].
        delta: umbral de distancia.

    Returns:
        F1 de cada pareja. Dimensions: [batch].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


class PointCloudClassifier(nn.Module):
    """
    Clasificador tipo PointNet (sin *T-nets*): un MLP compartido por punto
    (``Conv1d`` con kernel 1 o ``Linear``) y **max-pool** sobre los puntos, que
    hace la red invariante al orden.

    Arquitectura: ``3 → hidden → hidden`` (ReLU tras cada capa) por punto,
    máximo sobre los N puntos y ``Linear(hidden, num_classes)``.
    """

    def __init__(self, num_classes: int, hidden: int = 64) -> None:
        """
        Constructor de la clase.

        Args:
            num_classes: número de clases.
            hidden: tamaño de las features por punto.
        """

        super().__init__()

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            points: nubes de puntos. Dimensions: [batch, N, 3].

        Returns:
            Logits. Dimensions: [batch, num_classes].
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")


class VoxelClassifier(nn.Module):
    """
    Clasificador de vóxeles de ocupación con ``Conv3d``:

    ``Conv3d(1, c, 3, padding=1) → ReLU → MaxPool3d(2) → Conv3d(c, 2c, 3, padding=1)
    → ReLU → AdaptiveAvgPool3d(1) → Linear(2c, num_classes)``.
    """

    def __init__(self, num_classes: int, channels: int = 8) -> None:
        """
        Constructor de la clase.

        Args:
            num_classes: número de clases.
            channels: canales ``c`` de la primera convolución.
        """

        super().__init__()

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def forward(self, voxels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            voxels: rejillas de ocupación. Dimensions: [batch, 1, depth, height, width].

        Returns:
            Logits. Dimensions: [batch, num_classes].
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")
