"""
CV01 · Bloques de las arquitecturas clásicas (ResNet, GoogLeNet, DenseNet) — soluciones.
"""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    """
    Bloque básico de ResNet-18/34: ``relu(F(x) + shortcut(x))``.

    ``F`` = conv3×3 (stride) → BN → ReLU → conv3×3 → BN. El *shortcut* es la
    identidad si la forma no cambia; si cambia (``stride != 1`` o distinto número
    de canales) es conv1×1 (stride) → BN (*projection shortcut*).
    """

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        """
        Constructor de la clase.

        Args:
            in_channels: canales de entrada.
            out_channels: canales de salida.
            stride: *stride* de la primera convolución (2 reduce la resolución).
        """

        super().__init__()

        # BEGIN SOLUTION
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut: nn.Module = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        # END SOLUTION

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: entrada. Dimensions: [batch, in_channels, height, width].

        Returns:
            Salida. Dimensions: [batch, out_channels, height / stride, width / stride].
        """

        # BEGIN SOLUTION
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.shortcut(x))
        # END SOLUTION


class InceptionModule(nn.Module):
    """
    Módulo Inception con reducción de dimensión (GoogLeNet).

    Cuatro ramas en paralelo que se concatenan por canales:

    1. conv1×1 → ``c1``
    2. conv1×1 → ``c3_reduce`` → ReLU → conv3×3 (padding 1) → ``c3``
    3. conv1×1 → ``c5_reduce`` → ReLU → conv5×5 (padding 2) → ``c5``
    4. maxpool3×3 (stride 1, padding 1) → conv1×1 → ``pool_proj``

    Cada convolución va seguida de ReLU.
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
        Constructor de la clase.

        Args:
            in_channels: canales de entrada.
            c1: salidas de la rama 1×1.
            c3_reduce: canales tras la reducción 1×1 de la rama 3×3.
            c3: salidas de la rama 3×3.
            c5_reduce: canales tras la reducción 1×1 de la rama 5×5.
            c5: salidas de la rama 5×5.
            pool_proj: salidas de la rama de *pooling*.
        """

        super().__init__()

        # BEGIN SOLUTION
        self.branch1 = nn.Sequential(nn.Conv2d(in_channels, c1, 1), nn.ReLU())
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, c3_reduce, 1),
            nn.ReLU(),
            nn.Conv2d(c3_reduce, c3, 3, padding=1),
            nn.ReLU(),
        )
        self.branch5 = nn.Sequential(
            nn.Conv2d(in_channels, c5_reduce, 1),
            nn.ReLU(),
            nn.Conv2d(c5_reduce, c5, 5, padding=2),
            nn.ReLU(),
        )
        self.branch_pool = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, 1),
            nn.ReLU(),
        )
        # END SOLUTION

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: entrada. Dimensions: [batch, in_channels, height, width].

        Returns:
            Concatenación de las ramas, en el orden 1×1, 3×3, 5×5, pool.
            Dimensions: [batch, c1 + c3 + c5 + pool_proj, height, width].
        """

        # BEGIN SOLUTION
        branches = [self.branch1, self.branch3, self.branch5, self.branch_pool]
        return torch.cat([b(x) for b in branches], dim=1)
        # END SOLUTION


class DenseBlock(nn.Module):
    """
    Bloque denso de DenseNet: cada capa recibe la concatenación de todas las
    anteriores y aporta ``growth_rate`` canales nuevos.

    Capa ``i``: BN → ReLU → conv3×3 (padding 1, sin *bias*) de
    ``in_channels + i * growth_rate`` a ``growth_rate`` canales.
    """

    def __init__(self, in_channels: int, growth_rate: int, num_layers: int) -> None:
        """
        Constructor de la clase.

        Args:
            in_channels: canales de entrada.
            growth_rate: canales que añade cada capa (``k`` en el paper).
            num_layers: número de capas del bloque.
        """

        super().__init__()

        # BEGIN SOLUTION
        self.layers = nn.ModuleList(
            nn.Sequential(
                nn.BatchNorm2d(in_channels + i * growth_rate),
                nn.ReLU(),
                nn.Conv2d(
                    in_channels + i * growth_rate, growth_rate, 3, padding=1, bias=False
                ),
            )
            for i in range(num_layers)
        )
        # END SOLUTION

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: entrada. Dimensions: [batch, in_channels, height, width].

        Returns:
            ``cat([x, y_1, ..., y_L])``. Dimensions:
            [batch, in_channels + num_layers * growth_rate, height, width].
        """

        # BEGIN SOLUTION
        features = [x]
        for layer in self.layers:
            features.append(layer(torch.cat(features, dim=1)))
        return torch.cat(features, dim=1)
        # END SOLUTION
