"""
F10 · Capas convolucionales — soluciones.

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

    # BEGIN SOLUTION
    effective = dilation * (kernel_size - 1) + 1
    if n + 2 * padding < effective:
        raise ValueError("El kernel no cabe en la entrada con ese padding.")
    return (n + 2 * padding - effective) // stride + 1
    # END SOLUTION


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

    # BEGIN SOLUTION
    return (
        (n - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1
    )
    # END SOLUTION


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

    # BEGIN SOLUTION
    if in_channels % groups or out_channels % groups:
        raise ValueError("in_channels y out_channels deben ser múltiplos de groups.")
    params = out_channels * (in_channels // groups) * kernel_size * kernel_size
    return params + (out_channels if bias else 0)
    # END SOLUTION


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

    # BEGIN SOLUTION
    per_pixel = out_channels * (in_channels // groups) * kernel_size * kernel_size
    return out_height * out_width * per_pixel
    # END SOLUTION


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

    # BEGIN SOLUTION
    x = F.pad(x, (padding, padding, padding, padding))
    b, _, h, w = x.shape
    c_out, _, kh, kw = weight.shape
    h_out = (h - kh) // stride + 1
    w_out = (w - kw) // stride + 1
    out = x.new_zeros(b, c_out, h_out, w_out)
    for i in range(h_out):
        for j in range(w_out):
            hs, ws = i * stride, j * stride
            patch = x[:, :, hs : hs + kh, ws : ws + kw]  # [B, C_in, kh, kw]
            out[:, :, i, j] = torch.einsum("bchw,ochw->bo", patch, weight)
    if bias is not None:
        out = out + bias.view(1, -1, 1, 1)
    return out
    # END SOLUTION


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

    # BEGIN SOLUTION F.unfold -> [B, C*kh*kw, L]; luego W.view(C_out, -1) @ cols
    b, _, h, w = x.shape
    c_out, _, kh, kw = weight.shape
    h_out = (h + 2 * padding - kh) // stride + 1
    w_out = (w + 2 * padding - kw) // stride + 1
    cols = F.unfold(x, (kh, kw), padding=padding, stride=stride)  # [B, C*kh*kw, L]
    out = weight.view(c_out, -1) @ cols  # [B, C_out, L]
    if bias is not None:
        out = out + bias.view(1, -1, 1)
    return out.view(b, c_out, h_out, w_out)
    # END SOLUTION


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

    # BEGIN SOLUTION
    stride = kernel_size if stride is None else stride
    b, c, h, w = x.shape
    h_out = (h - kernel_size) // stride + 1
    w_out = (w - kernel_size) // stride + 1
    cols = F.unfold(x, kernel_size, stride=stride)  # [B, C*k*k, L]
    cols = cols.view(b, c, kernel_size * kernel_size, -1)
    return cols.max(dim=2).values.view(b, c, h_out, w_out)
    # END SOLUTION


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

    # BEGIN SOLUTION
    r, jump = 1, 1
    for layer in layers:
        k, s = layer[0], layer[1]
        d = layer[2] if len(layer) > 2 else 1
        r += d * (k - 1) * jump
        jump *= s
    return r
    # END SOLUTION


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

        # BEGIN SOLUTION
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=in_channels,
            bias=bias,
        )
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)
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
        return self.pointwise(self.depthwise(x))
        # END SOLUTION


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

    # BEGIN SOLUTION
    h, w = x.shape
    k = weight.shape[0]
    canvas = x.new_zeros((h - 1) * stride + k + output_padding, (w - 1) * stride + k + output_padding)
    for i in range(h):
        for j in range(w):
            hs, ws = i * stride, j * stride
            canvas[hs : hs + k, ws : ws + k] += x[i, j] * weight
    h_out = (h - 1) * stride - 2 * padding + k + output_padding
    w_out = (w - 1) * stride - 2 * padding + k + output_padding
    return canvas[padding : padding + h_out, padding : padding + w_out]
    # END SOLUTION


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

    # BEGIN SOLUTION
    b, c_r2, h, w = x.shape
    r = upscale_factor
    c = c_r2 // (r * r)
    x = x.view(b, c, r, r, h, w)
    x = x.permute(0, 1, 4, 2, 5, 3)  # [B, C, H, r, W, r]
    return x.reshape(b, c, h * r, w * r)
    # END SOLUTION


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

        # BEGIN SOLUTION
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        # END SOLUTION

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: entrada. Dimensions: [batch, in_channels, height, width].

        Returns:
            Salida. Dimensions: [batch, out_channels, height * s, width * s].
        """

        # BEGIN SOLUTION usa align_corners=False solo si mode es "bilinear"
        kwargs = {"align_corners": False} if self.mode == "bilinear" else {}
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode, **kwargs)
        return self.conv(x)
        # END SOLUTION
