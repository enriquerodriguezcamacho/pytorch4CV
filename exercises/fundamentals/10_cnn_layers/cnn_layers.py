"""
F10 · Capas convolucionales — ejercicios.

Funciones y módulos para entender qué hace por dentro ``nn.Conv2d``,
``nn.MaxPool2d``, ``nn.ConvTranspose2d`` y ``nn.PixelShuffle``.
"""

import torch
import torch.nn.functional as F
from torch import nn


def conv_output_size(
    n: int, kernel_size: int, stride: int = 1, padding: int = 0, dilation: int = 1
) -> int:
    """
    Tamaño espacial de la salida de una convolución (o de un pooling).

    Fórmula: floor((n + 2p - d(k - 1) - 1) / s) + 1.

    Args:
        n: tamaño de entrada en una dimensión (alto o ancho).
        kernel_size: tamaño del kernel ``k``.
        stride: paso ``s``.
        padding: relleno ``p`` por cada lado.
        dilation: dilatación ``d``.

    Returns:
        Tamaño de la salida en esa dimensión.

    Raises:
        ValueError: si el kernel (dilatado) no cabe en la entrada rellenada.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def conv_transpose_output_size(
    n: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    output_padding: int = 0,
    dilation: int = 1,
) -> int:
    """
    Tamaño espacial de la salida de una convolución transpuesta.

    Fórmula: (n - 1)s - 2p + d(k - 1) + output_padding + 1.

    Args:
        n: tamaño de entrada en una dimensión.
        kernel_size: tamaño del kernel ``k``.
        stride: paso ``s``.
        padding: relleno ``p`` (aquí *recorta* la salida).
        output_padding: filas/columnas extra añadidas a un lado de la salida.
        dilation: dilatación ``d``.

    Returns:
        Tamaño de la salida en esa dimensión.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def count_conv_params(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    groups: int = 1,
    bias: bool = True,
) -> int:
    """
    Número de parámetros entrenables de un ``nn.Conv2d`` con kernel cuadrado.

    Args:
        in_channels: canales de entrada ``C_in``.
        out_channels: canales de salida ``C_out``.
        kernel_size: lado ``k`` del kernel.
        groups: número de grupos ``g`` (``C_in`` y ``C_out`` deben ser múltiplos).
        bias: si la capa tiene sesgo.

    Returns:
        ``C_out * (C_in / g) * k * k`` (+ ``C_out`` si hay sesgo).

    Raises:
        ValueError: si ``in_channels`` u ``out_channels`` no son múltiplos de ``groups``.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def count_conv_macs(
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    out_height: int,
    out_width: int,
    groups: int = 1,
) -> int:
    """
    Multiplicaciones-acumulaciones (MACs) de una convolución 2D sin contar el sesgo.

    Los FLOPs se suelen aproximar como ``2 * MACs``.

    Args:
        in_channels: canales de entrada ``C_in``.
        out_channels: canales de salida ``C_out``.
        kernel_size: lado ``k`` del kernel.
        out_height: alto de la salida ``H_out``.
        out_width: ancho de la salida ``W_out``.
        groups: número de grupos ``g``.

    Returns:
        ``H_out * W_out * C_out * (C_in / g) * k * k``.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def conv2d_naive(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int = 1,
    padding: int = 0,
) -> torch.Tensor:
    """
    Convolución 2D (correlación cruzada) con bucles sobre las posiciones de salida.

    Equivale a ``F.conv2d(x, weight, bias, stride, padding)`` con ``groups=1`` y
    ``dilation=1``. Se permite vectorizar sobre lote y canales, pero recorre las
    posiciones ``(i, j)`` de la salida con bucles ``for``.

    Args:
        x: entrada. Dimensions: [batch, in_channels, height, width].
        weight: kernels. Dimensions: [out_channels, in_channels, k_h, k_w].
        bias: sesgo opcional. Dimensions: [out_channels].
        stride: paso (igual en alto y ancho).
        padding: relleno con ceros (igual en alto y ancho).

    Returns:
        Salida. Dimensions: [batch, out_channels, height_out, width_out].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def conv2d_im2col(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    stride: int = 1,
    padding: int = 0,
) -> torch.Tensor:
    """
    Convolución 2D como un producto matricial (im2col) usando ``F.unfold``.

    Args:
        x: entrada. Dimensions: [batch, in_channels, height, width].
        weight: kernels. Dimensions: [out_channels, in_channels, k_h, k_w].
        bias: sesgo opcional. Dimensions: [out_channels].
        stride: paso.
        padding: relleno con ceros.

    Returns:
        Salida. Dimensions: [batch, out_channels, height_out, width_out].
    """

    # TODO: F.unfold -> [B, C*kh*kw, L]; luego W.view(C_out, -1) @ cols
    raise NotImplementedError("TODO")


def max_pool2d(x: torch.Tensor, kernel_size: int, stride: int | None = None) -> torch.Tensor:
    """
    Max pooling 2D sin padding implementado con ``F.unfold``.

    Args:
        x: entrada. Dimensions: [batch, channels, height, width].
        kernel_size: lado de la ventana.
        stride: paso; si es ``None`` vale ``kernel_size`` (como en ``nn.MaxPool2d``).

    Returns:
        Salida. Dimensions: [batch, channels, height_out, width_out].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def receptive_field(layers: list[tuple[int, ...]]) -> int:
    """
    Campo receptivo (en píxeles de la entrada) de una pila de capas.

    Recurrencia: ``r_l = r_{l-1} + (k_eff - 1) * j_{l-1}`` y ``j_l = j_{l-1} * s``,
    con ``r_0 = 1``, ``j_0 = 1`` y ``k_eff = d(k - 1) + 1``.

    Args:
        layers: lista de tuplas ``(kernel, stride)`` o ``(kernel, stride, dilation)``,
            en orden desde la entrada. Sirve para convoluciones y poolings.

    Returns:
        Lado del campo receptivo de una neurona de la última capa.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


class DepthwiseSeparableConv(nn.Module):
    """
    Convolución separable en profundidad (MobileNet): depthwise ``k×k`` + pointwise 1×1.

    Atributos que hay que crear:
        * ``self.depthwise``: ``nn.Conv2d(in, in, k, stride, padding=k // 2,
          groups=in, bias=bias)``.
        * ``self.pointwise``: ``nn.Conv2d(in, out, 1, bias=bias)``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        bias: bool = True,
    ) -> None:
        """
        Constructor.

        Args:
            in_channels: canales de entrada.
            out_channels: canales de salida.
            kernel_size: lado (impar) del kernel depthwise.
            stride: paso de la convolución depthwise.
            bias: si las dos convoluciones llevan sesgo.
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


def transposed_conv2d(
    x: torch.Tensor,
    weight: torch.Tensor,
    stride: int = 1,
    padding: int = 0,
    output_padding: int = 0,
) -> torch.Tensor:
    """
    Convolución transpuesta 2D de un solo canal, "desde cero" (scatter-add).

    Cada píxel de entrada ``x[i, j]`` pinta una copia del kernel escalada por su valor
    en la posición ``(i * s, j * s)`` de un lienzo de tamaño ``(H - 1)s + k +
    output_padding``; las contribuciones solapadas se suman. Después se recortan
    ``padding`` filas/columnas por arriba/izquierda y se toma el tamaño final.
    Equivale a ``F.conv_transpose2d(x[None, None], weight[None, None], ...)``.

    Args:
        x: entrada de un canal. Dimensions: [height, width].
        weight: kernel. Dimensions: [k, k].
        stride: paso ``s``.
        padding: relleno ``p``.
        output_padding: tamaño extra añadido al final.

    Returns:
        Salida. Dimensions: [(H - 1)s - 2p + k + output_padding,
        (W - 1)s - 2p + k + output_padding].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def pixel_shuffle(x: torch.Tensor, upscale_factor: int) -> torch.Tensor:
    """
    Reordena canales en espacio (sub-pixel convolution, ESPCN) con ``view``/``permute``.

    Equivale a ``F.pixel_shuffle``: ``out[b, c, h*r + i, w*r + j] =
    x[b, c*r*r + i*r + j, h, w]``.

    Args:
        x: entrada. Dimensions: [batch, channels * r * r, height, width].
        upscale_factor: factor ``r``.

    Returns:
        Salida. Dimensions: [batch, channels, height * r, width * r].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


class UpsampleConv(nn.Module):
    """
    Alternativa a ``ConvTranspose2d`` sin artefactos *checkerboard*:
    ``F.interpolate`` (nearest o bilinear) + ``Conv2d`` 3×3 con ``padding=1``.

    Atributos: ``self.scale_factor``, ``self.mode`` y ``self.conv``.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        scale_factor: int = 2,
        mode: str = "nearest",
    ) -> None:
        """
        Constructor.

        Args:
            in_channels: canales de entrada.
            out_channels: canales de salida.
            scale_factor: factor de aumento espacial.
            mode: ``"nearest"`` o ``"bilinear"``.
        """

        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: entrada. Dimensions: [batch, in_channels, height, width].

        Returns:
            Salida. Dimensions: [batch, out_channels, height * s, width * s].
        """

        # TODO: usa align_corners=False solo si mode es "bilinear"
        raise NotImplementedError("TODO")
