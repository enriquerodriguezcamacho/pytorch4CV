"""
F11 · Arquitecturas: bloques de construcción — ejercicios.

Bloques de las CNN clásicas: VGG, ResNet (básico y bottleneck), Inception,
DenseNet y Squeeze-and-Excitation.
"""

import torch
import torch.nn.functional as F
from torch import nn


class VGGBlock(nn.Module):
    """
    Bloque VGG: ``num_convs`` × (Conv 3×3 ``padding=1`` + [BN] + ReLU) + MaxPool 2×2.

    Atributo que hay que crear: ``self.block`` (``nn.Sequential``).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_convs: int = 2,
        batch_norm: bool = False,
    ) -> None:
        """
        Constructor.

        Args:
            in_channels: canales de entrada.
            out_channels: canales de salida (de todas las convoluciones).
            num_convs: número de convoluciones 3×3.
            batch_norm: si se añade ``nn.BatchNorm2d`` tras cada convolución (en ese
                caso las convoluciones van sin sesgo).
        """

        super().__init__()

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: entrada. Dimensions: [batch, in_channels, height, width].

        Returns:
            Salida. Dimensions: [batch, out_channels, height // 2, width // 2].
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")


class ResidualBlock(nn.Module):
    """
    Bloque residual básico de ResNet-18/34.

    ``out = ReLU(BN(conv2(ReLU(BN(conv1(x))))) + shortcut(x))``

    Atributos que hay que crear:
        * ``self.conv1``: Conv 3×3, ``stride``, ``padding=1``, sin sesgo; ``self.bn1``.
        * ``self.conv2``: Conv 3×3, ``stride=1``, ``padding=1``, sin sesgo; ``self.bn2``.
        * ``self.shortcut``: ``nn.Identity()`` si no cambian ni canales ni resolución;
          si cambian, proyección ``nn.Sequential(Conv 1×1 con stride y sin sesgo, BN)``.
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        """
        Constructor.

        Args:
            in_channels: canales de entrada.
            out_channels: canales de salida.
            stride: paso de la primera convolución (2 para reducir la resolución).
        """

        super().__init__()

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: entrada. Dimensions: [batch, in_channels, height, width].

        Returns:
            Salida. Dimensions: [batch, out_channels, height / stride, width / stride].
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")


class BottleneckBlock(nn.Module):
    """
    Bloque *bottleneck* de ResNet-50/101/152: 1×1 (reduce) → 3×3 → 1×1 (expande).

    Los canales de salida son ``mid_channels * expansion``. Todas las convoluciones
    van sin sesgo y seguidas de BN; ReLU tras las dos primeras y tras la suma.

    Atributos que hay que crear: ``self.conv1``, ``self.bn1``, ``self.conv2``
    (3×3 con el ``stride``), ``self.bn2``, ``self.conv3``, ``self.bn3`` y
    ``self.shortcut`` (identidad o proyección 1×1 + BN, como en ``ResidualBlock``).
    """

    def __init__(
        self, in_channels: int, mid_channels: int, stride: int = 1, expansion: int = 4
    ) -> None:
        """
        Constructor.

        Args:
            in_channels: canales de entrada.
            mid_channels: canales del cuello de botella.
            stride: paso de la convolución 3×3.
            expansion: factor de expansión de la última 1×1.
        """

        super().__init__()
        out_channels = mid_channels * expansion
        self.out_channels = out_channels

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: entrada. Dimensions: [batch, in_channels, height, width].

        Returns:
            Salida. Dimensions: [batch, mid_channels * expansion, height / stride,
            width / stride].
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")


class InceptionModule(nn.Module):
    """
    Módulo Inception (GoogLeNet, versión con reducción de dimensión 1×1).

    Cuatro ramas en paralelo, cada convolución con sesgo y seguida de ReLU (sin BN):

    1. ``self.branch1``: Conv 1×1 → ``c1``.
    2. ``self.branch3``: Conv 1×1 → ``c3_reduce``, Conv 3×3 (``padding=1``) → ``c3``.
    3. ``self.branch5``: Conv 1×1 → ``c5_reduce``, Conv 5×5 (``padding=2``) → ``c5``.
    4. ``self.branch_pool``: MaxPool 3×3 (``stride=1``, ``padding=1``), Conv 1×1 →
       ``pool_proj``.

    Las salidas se concatenan por canales en ese orden.
    """

    def __init__(
        self,
        in_channels: int,
        c1: int,
        c3_reduce: int,
        c3: int,
        c5_reduce: int,
        c5: int,
        pool_proj: int,
    ) -> None:
        """
        Constructor.

        Args:
            in_channels: canales de entrada.
            c1: canales de la rama 1×1.
            c3_reduce: canales de la 1×1 previa a la 3×3.
            c3: canales de la rama 3×3.
            c5_reduce: canales de la 1×1 previa a la 5×5.
            c5: canales de la rama 5×5.
            pool_proj: canales de la 1×1 tras el max pooling.
        """

        super().__init__()
        self.out_channels = c1 + c3 + c5 + pool_proj

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: entrada. Dimensions: [batch, in_channels, height, width].

        Returns:
            Concatenación de las 4 ramas. Dimensions:
            [batch, c1 + c3 + c5 + pool_proj, height, width].
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")


class DenseLayer(nn.Module):
    """
    Capa de DenseNet (ya dada): BN → ReLU → Conv 3×3 (``growth_rate``, sin sesgo).
    """

    def __init__(self, in_channels: int, growth_rate: int) -> None:
        super().__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, growth_rate, 3, padding=1, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(F.relu(self.bn(x)))


class DenseBlock(nn.Module):
    """
    Bloque denso de DenseNet: la capa ``l`` recibe la concatenación de la entrada y
    de las salidas de todas las capas anteriores.

    Atributo que hay que crear: ``self.layers`` (``nn.ModuleList`` de ``DenseLayer``;
    la capa ``l`` tiene ``in_channels + l * growth_rate`` canales de entrada).
    """

    def __init__(self, in_channels: int, growth_rate: int, num_layers: int) -> None:
        """
        Constructor.

        Args:
            in_channels: canales de entrada ``C``.
            growth_rate: canales que añade cada capa ``k``.
            num_layers: número de capas ``L``.
        """

        super().__init__()
        self.out_channels = in_channels + num_layers * growth_rate

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: entrada. Dimensions: [batch, in_channels, height, width].

        Returns:
            Entrada concatenada con las salidas de todas las capas (la entrada va
            primero). Dimensions: [batch, in_channels + L * k, height, width].
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")


class SEBlock(nn.Module):
    """
    Bloque Squeeze-and-Excitation (recalibración de canales).

    ``s = sigmoid(W2 · ReLU(W1 · GAP(x)))`` y ``out = x * s`` canal a canal.

    Atributo que hay que crear: ``self.fc = nn.Sequential(Linear(C, C // r), ReLU,
    Linear(C // r, C), Sigmoid)``.
    """

    def __init__(self, channels: int, reduction: int = 16) -> None:
        """
        Constructor.

        Args:
            channels: número de canales ``C``.
            reduction: factor de reducción ``r`` (mínimo 1 canal oculto).
        """

        super().__init__()
        hidden = max(1, channels // reduction)

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: entrada. Dimensions: [batch, channels, height, width].

        Returns:
            Entrada reescalada por canal. Dimensions: [batch, channels, height, width].
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")
