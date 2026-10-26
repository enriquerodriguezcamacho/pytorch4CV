"""
CV07 · Modelos autorregresivos: convolución enmascarada y PixelCNN — soluciones.
"""

import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn


def build_causal_mask(kernel_size: int, mask_type: Literal["A", "B"]) -> torch.Tensor:
    """
    Máscara espacial ``[kernel_size, kernel_size]`` de PixelCNN (orden *raster*).

    Vale 1 en las filas por encima del centro y en la fila central a la izquierda
    del centro. El píxel central vale 1 solo en la máscara ``"B"``; en la ``"A"``
    (primera capa) vale 0 para que la salida no vea el propio píxel de entrada.

    Args:
        kernel_size: tamaño impar del kernel.
        mask_type: ``"A"`` o ``"B"``.

    Returns:
        Máscara ``float32``. Dimensions: [kernel_size, kernel_size].

    Raises:
        ValueError: si ``mask_type`` no es ``"A"``/``"B"`` o ``kernel_size`` es par.
    """

    # BEGIN SOLUTION
    if mask_type not in ("A", "B"):
        raise ValueError(f"mask_type debe ser 'A' o 'B', no {mask_type!r}")
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size debe ser impar")
    c = kernel_size // 2
    mask = torch.zeros(kernel_size, kernel_size)
    mask[:c, :] = 1.0
    mask[c, :c] = 1.0
    if mask_type == "B":
        mask[c, c] = 1.0
    return mask
    # END SOLUTION


class MaskedConv2d(nn.Conv2d):
    """
    ``Conv2d`` con máscara causal (PixelCNN, una sola "cadena" de canales).

    La máscara se guarda como *buffer* (no es un parámetro) con forma
    ``[1, 1, k, k]`` y se multiplica por ``self.weight`` en cada forward, de modo
    que la salida en ``(i, j)`` solo depende de píxeles anteriores en orden raster.
    El padding es ``kernel_size // 2`` para conservar el tamaño.

    Args:
        in_channels: canales de entrada.
        out_channels: canales de salida.
        kernel_size: tamaño impar del kernel.
        mask_type: ``"A"`` (primera capa) o ``"B"`` (resto).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        mask_type: Literal["A", "B"],
    ) -> None:
        super().__init__(in_channels, out_channels, kernel_size, padding=kernel_size // 2)

        # BEGIN SOLUTION
        mask = build_causal_mask(kernel_size, mask_type)
        self.register_buffer("mask", mask[None, None])
        # END SOLUTION

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convolución con los pesos enmascarados.

        Args:
            x: entrada. Dimensions: [batch, in_channels, height, width].

        Returns:
            Salida. Dimensions: [batch, out_channels, height, width].
        """

        # BEGIN SOLUTION
        return F.conv2d(x, self.weight * self.mask, self.bias, padding=self.padding)
        # END SOLUTION


class PixelCNN(nn.Module):
    """
    PixelCNN mínima para imágenes en escala de grises cuantizadas a ``n_levels``.

    Arquitectura:
        - ``MaskedConv2d(1, hidden, 7, "A")`` + ReLU.
        - ``n_layers`` × (``MaskedConv2d(hidden, hidden, 3, "B")`` + ReLU).
        - ``Conv2d(hidden, n_levels, 1)`` → logits por píxel (softmax de
          ``n_levels`` valores, discreto).

    Args:
        hidden: canales ocultos.
        n_layers: número de capas de tipo B.
        n_levels: número de niveles de intensidad (256 en el paper).
    """

    def __init__(self, hidden: int = 16, n_layers: int = 2, n_levels: int = 256) -> None:
        super().__init__()

        self.n_levels = n_levels

        # BEGIN SOLUTION
        layers: list[nn.Module] = [MaskedConv2d(1, hidden, 7, "A"), nn.ReLU()]
        for _ in range(n_layers):
            layers += [MaskedConv2d(hidden, hidden, 3, "B"), nn.ReLU()]
        layers.append(nn.Conv2d(hidden, n_levels, 1))
        self.net = nn.Sequential(*layers)
        # END SOLUTION

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Logits de p(x_ij | x_<ij).

        Args:
            x: imágenes normalizadas en [0, 1]. Dimensions: [batch, 1, height, width].

        Returns:
            Logits. Dimensions: [batch, n_levels, height, width].
        """

        # BEGIN SOLUTION
        return self.net(x)
        # END SOLUTION


def pixelcnn_nll(logits: torch.Tensor, x_int: torch.Tensor) -> torch.Tensor:
    """
    NLL media por píxel (en *nats*) = entropía cruzada de ``n_levels`` clases.

    Args:
        logits: salida de PixelCNN. Dimensions: [batch, n_levels, height, width].
        x_int: imagen con enteros en ``[0, n_levels)``. Dimensions:
            [batch, 1, height, width] (``int64``).

    Returns:
        Escalar.
    """

    # BEGIN SOLUTION
    return F.cross_entropy(logits, x_int[:, 0])
    # END SOLUTION


def bits_per_dim(nll_nats: torch.Tensor) -> torch.Tensor:
    """
    Convierte una NLL media por dimensión de nats a bits (``nll / ln 2``).

    Args:
        nll_nats: NLL en nats. Escalar.

    Returns:
        NLL en bits por dimensión. Escalar.
    """

    # BEGIN SOLUTION
    return nll_nats / math.log(2)
    # END SOLUTION


@torch.no_grad()
def sample(model: PixelCNN, height: int, width: int, n: int = 1) -> torch.Tensor:
    """
    Generación **secuencial** píxel a píxel en orden raster.

    Empieza con una imagen de ceros; para cada ``(i, j)`` ejecuta el modelo,
    muestrea el nivel con ``torch.multinomial`` de ``softmax(logits[:, :, i, j])``
    y escribe ``nivel / (n_levels - 1)`` en la imagen.

    Args:
        model: PixelCNN (se pone en ``eval``).
        height: alto de la imagen.
        width: ancho de la imagen.
        n: número de muestras.

    Returns:
        Niveles enteros muestreados ``int64``. Dimensions: [n, 1, height, width].
    """

    # BEGIN SOLUTION
    model.eval()
    device = next(model.parameters()).device
    x = torch.zeros(n, 1, height, width, device=device)
    levels = torch.zeros(n, 1, height, width, dtype=torch.long, device=device)
    for i in range(height):
        for j in range(width):
            probs = F.softmax(model(x)[:, :, i, j], dim=1)
            lvl = torch.multinomial(probs, 1).squeeze(1)
            levels[:, 0, i, j] = lvl
            x[:, 0, i, j] = lvl.float() / (model.n_levels - 1)
    return levels
    # END SOLUTION
