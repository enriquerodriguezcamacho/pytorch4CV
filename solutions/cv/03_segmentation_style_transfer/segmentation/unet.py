"""
CV03 · Segmentación: convolución transpuesta, U-Net y cabeza FCN — soluciones.
"""

import torch
import torch.nn.functional as F
from torch import nn

# ---------------------------------------------------------------------------
# Convolución transpuesta
# ---------------------------------------------------------------------------


def transposed_conv_output_size(
    size: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    output_padding: int = 0,
    dilation: int = 1,
) -> int:
    """
    Tamaño de salida (en una dimensión) de una convolución transpuesta.

    Fórmula de ``nn.ConvTranspose2d``:
    ``(size - 1) * stride - 2 * padding + dilation * (kernel_size - 1)
    + output_padding + 1``.

    Args:
        size: tamaño de entrada (alto o ancho).
        kernel_size: tamaño del kernel.
        stride: paso.
        padding: filas/columnas que se eliminan de CADA lado de la salida.
        output_padding: filas/columnas extra que se añaden a un solo lado.
        dilation: dilatación del kernel.

    Returns:
        Tamaño de la salida en esa dimensión.
    """

    # BEGIN SOLUTION
    return (
        (size - 1) * stride
        - 2 * padding
        + dilation * (kernel_size - 1)
        + output_padding
        + 1
    )
    # END SOLUTION


def conv_transpose2d_manual(
    x: torch.Tensor, kernel: torch.Tensor, stride: int = 1, padding: int = 0
) -> torch.Tensor:
    """
    Convolución transpuesta 2D de un solo canal implementada "a mano".

    Cada elemento ``x[i, j]`` multiplica al kernel completo y el resultado se SUMA en
    la salida a partir de la posición ``(i * stride, j * stride)``. Al final se quitan
    ``padding`` filas/columnas de cada borde.

    Args:
        x: entrada. Dimensions: [height, width].
        kernel: kernel. Dimensions: [kh, kw].
        stride: paso.
        padding: padding de la convolución transpuesta (recorta la salida).

    Returns:
        Salida. Dimensions: [(height - 1) * stride + kh - 2 * padding,
        (width - 1) * stride + kw - 2 * padding].
    """

    # BEGIN SOLUTION
    h, w = x.shape
    kh, kw = kernel.shape
    out = x.new_zeros((h - 1) * stride + kh, (w - 1) * stride + kw)
    for i in range(h):
        for j in range(w):
            r, c = i * stride, j * stride
            out[r : r + kh, c : c + kw] += x[i, j] * kernel
    if padding > 0:
        out = out[padding:-padding, padding:-padding]
    return out
    # END SOLUTION


# ---------------------------------------------------------------------------
# Bloques de la U-Net
# ---------------------------------------------------------------------------


def center_crop(x: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """
    Recorta el centro de un mapa de activación (U-Net original, convs sin padding).

    Args:
        x: tensor. Dimensions: [batch, channels, height, width].
        size: ``(height', width')`` deseados, con ``height' <= height`` y
            ``width' <= width``.

    Returns:
        Tensor recortado. Dimensions: [batch, channels, height', width'].
    """

    # BEGIN SOLUTION
    th, tw = size
    h, w = x.shape[-2:]
    top, left = (h - th) // 2, (w - tw) // 2
    return x[..., top : top + th, left : left + tw]
    # END SOLUTION


class ConvBlock(nn.Module):
    """
    Dos capas ``Conv2d(3x3, padding=1) -> [BatchNorm2d] -> ReLU``.
    """

    def __init__(self, in_channels: int, out_channels: int, use_bn: bool = True) -> None:
        """
        Constructor de la clase.

        Args:
            in_channels: canales de entrada del bloque.
            out_channels: canales de salida del bloque.
            use_bn: si es ``True`` se añade ``BatchNorm2d`` tras cada convolución
                (y la convolución va sin sesgo); si es ``False``, no hay BN.
        """

        super().__init__()

        # BEGIN SOLUTION
        layers: list[nn.Module] = []
        channels = [in_channels, out_channels, out_channels]
        for c_in, c_out in zip(channels[:-1], channels[1:]):
            layers.append(nn.Conv2d(c_in, c_out, 3, padding=1, bias=not use_bn))
            if use_bn:
                layers.append(nn.BatchNorm2d(c_out))
            layers.append(nn.ReLU())
        self.block = nn.Sequential(*layers)
        # END SOLUTION

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: entrada. Dimensions: [batch, in_channels, height, width].

        Returns:
            Salida (no negativa por la ReLU). Dimensions:
            [batch, out_channels, height, width].
        """

        # BEGIN SOLUTION
        return self.block(x)
        # END SOLUTION


class EncoderBlock(nn.Module):
    """
    Bloque del encoder: ``ConvBlock`` seguido de ``MaxPool2d(2)``.
    """

    def __init__(self, in_channels: int, out_channels: int) -> None:
        """
        Constructor de la clase.

        Args:
            in_channels: canales de entrada.
            out_channels: canales de salida.
        """

        super().__init__()

        # BEGIN SOLUTION
        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # END SOLUTION

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: entrada. Dimensions: [batch, in_channels, height, width].

        Returns:
            Tensor antes del max pooling (skip connection) y después. Dimensions:
            [batch, out_channels, height, width] y
            [batch, out_channels, height // 2, width // 2].
        """

        # BEGIN SOLUTION
        skip = self.conv(x)
        return skip, self.pool(skip)
        # END SOLUTION


class DecoderBlock(nn.Module):
    """
    Bloque del decoder: ``ConvTranspose2d(k=2, s=2)`` + concatenación con la skip +
    ``ConvBlock``.
    """

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        """
        Constructor de la clase.

        Args:
            in_channels: canales del tensor que llega desde abajo (más profundo).
            skip_channels: canales de la skip connection del encoder.
            out_channels: canales de salida. La conv transpuesta pasa de
                ``in_channels`` a ``out_channels`` y el ``ConvBlock`` de
                ``out_channels + skip_channels`` a ``out_channels``.
        """

        super().__init__()

        # BEGIN SOLUTION
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels + skip_channels, out_channels)
        # END SOLUTION

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Si tras el upsampling el tamaño no coincide con el de la skip
        (entradas con tamaño impar), se rellena el tensor subido con ceros por la
        derecha/abajo (``F.pad``) hasta igualarlo. La concatenación es
        ``torch.cat([skip, x], dim=1)`` (primero la skip).

        Args:
            x: tensor del nivel inferior. Dimensions: [batch, in_channels, h, w].
            skip: skip connection. Dimensions: [batch, skip_channels, H, W] con
                ``H in {2h, 2h + 1}`` y ``W in {2w, 2w + 1}``.

        Returns:
            Salida. Dimensions: [batch, out_channels, H, W].
        """

        # BEGIN SOLUTION
        x = self.up(x)
        dh = skip.shape[-2] - x.shape[-2]
        dw = skip.shape[-1] - x.shape[-1]
        if dh or dw:
            x = F.pad(x, [0, dw, 0, dh])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)
        # END SOLUTION


class UNet(nn.Module):
    """
    U-Net para segmentación semántica (devuelve logits por píxel).

    Con ``channels = [16, 32, 64]`` y ``in_channels = 3``::

        enc 3->16, enc 16->32, bottleneck ConvBlock 32->64,
        dec 64->32 (skip 32), dec 32->16 (skip 16), clasificador Conv1x1 16->n_classes
    """

    def __init__(
        self,
        in_channels: int = 3,
        n_classes: int = 2,
        channels: tuple[int, ...] | list[int] = (16, 32, 64),
    ) -> None:
        """
        Constructor de la clase. Debes definir ``self.encoders`` (``nn.ModuleList`` de
        ``EncoderBlock``), ``self.bottleneck`` (``ConvBlock``), ``self.decoders``
        (``nn.ModuleList`` de ``DecoderBlock``) y ``self.classifier``
        (``Conv2d`` 1x1).

        Args:
            in_channels: canales de la imagen de entrada.
            n_classes: número de clases (canales de salida).
            channels: canales de cada nivel; el último es el del bottleneck.
        """

        super().__init__()

        self.n_classes = n_classes
        channels = list(channels)

        # BEGIN SOLUTION
        enc_in = [in_channels, *channels[:-2]]
        self.encoders = nn.ModuleList(
            EncoderBlock(c_in, c_out) for c_in, c_out in zip(enc_in, channels[:-1])
        )
        self.bottleneck = ConvBlock(channels[-2], channels[-1])
        rev = channels[::-1]
        self.decoders = nn.ModuleList(
            DecoderBlock(c_in, c_out, c_out) for c_in, c_out in zip(rev[:-1], rev[1:])
        )
        self.classifier = nn.Conv2d(channels[0], n_classes, kernel_size=1)
        # END SOLUTION

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: imagen. Dimensions: [batch, in_channels, height, width].

        Returns:
            Logits por píxel (sin softmax). Dimensions:
            [batch, n_classes, height, width].
        """

        # BEGIN SOLUTION
        skips = []
        for encoder in self.encoders:
            skip, x = encoder(x)
            skips.append(skip)
        x = self.bottleneck(x)
        for decoder, skip in zip(self.decoders, reversed(skips)):
            x = decoder(x, skip)
        return self.classifier(x)
        # END SOLUTION

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predice la máscara de clases de un lote de imágenes.

        Args:
            x: imágenes. Dimensions: [batch, in_channels, height, width].

        Returns:
            Clase de cada píxel (``int64``). Dimensions: [batch, height, width].
        """

        self.eval()
        with torch.no_grad():
            return self(x).argmax(dim=1)


# ---------------------------------------------------------------------------
# FCN
# ---------------------------------------------------------------------------


class FCNHead(nn.Module):
    """
    Cabeza de una FCN: ``Conv2d 1x1`` (canales -> clases) y reescalado bilineal a la
    resolución de la imagen original (``F.interpolate``).
    """

    def __init__(self, in_channels: int, n_classes: int) -> None:
        """
        Constructor de la clase.

        Args:
            in_channels: canales del mapa de features del backbone.
            n_classes: número de clases.
        """

        super().__init__()

        # BEGIN SOLUTION
        self.classifier = nn.Conv2d(in_channels, n_classes, kernel_size=1)
        # END SOLUTION

    def forward(self, features: torch.Tensor, output_size: tuple[int, int]) -> torch.Tensor:
        """
        Forward pass.

        Args:
            features: features del backbone. Dimensions: [batch, in_channels, h, w].
            output_size: ``(height, width)`` de la imagen original.

        Returns:
            Logits reescalados con ``mode="bilinear"`` y ``align_corners=False``.
            Dimensions: [batch, n_classes, height, width].
        """

        # BEGIN SOLUTION
        logits = self.classifier(features)
        return F.interpolate(
            logits, size=output_size, mode="bilinear", align_corners=False
        )
        # END SOLUTION
