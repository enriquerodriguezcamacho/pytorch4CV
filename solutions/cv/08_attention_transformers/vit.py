"""
CV08 · Atención y Transformers: Vision Transformer mínimo — soluciones.
"""

from __future__ import annotations

import torch
from torch import nn

from .attention import TransformerEncoderBlock


class PatchEmbedding(nn.Module):
    """
    Divide la imagen en parches ``P × P`` y proyecta cada uno a ``d_model`` con una
    ``Conv2d(in_channels, d_model, kernel_size=P, stride=P)`` (equivale a aplanar
    cada parche y aplicar una capa lineal).

    Salida: ``conv(x).flatten(2).transpose(1, 2)``, con los parches por filas.
    """

    def __init__(self, image_size: int, patch_size: int, in_channels: int, d_model: int) -> None:
        """
        Constructor de la clase.

        Args:
            image_size: lado de la imagen (cuadrada).
            patch_size: lado ``P`` de cada parche (divide a ``image_size``).
            in_channels: canales de la imagen.
            d_model: dimensión del *embedding*.

        Raises:
            ValueError: si ``patch_size`` no divide a ``image_size``.
        """

        super().__init__()

        # BEGIN SOLUTION
        if image_size % patch_size != 0:
            raise ValueError("patch_size debe dividir a image_size.")
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, d_model, kernel_size=patch_size, stride=patch_size)
        # END SOLUTION

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: imágenes. Dimensions: [batch, in_channels, image_size, image_size].

        Returns:
            Secuencia de parches. Dimensions: [batch, num_patches, d_model].
        """

        # BEGIN SOLUTION
        return self.proj(x).flatten(2).transpose(1, 2)
        # END SOLUTION


class VisionTransformer(nn.Module):
    """
    ViT mínimo (Dosovitskiy et al., 2021):

    1. :class:`PatchEmbedding` → ``[B, N, d]``.
    2. Se antepone el token ``[CLS]`` (``nn.Parameter`` de forma ``[1, 1, d]``,
       expandido al lote) → ``[B, N + 1, d]``.
    3. Se suma el *embedding* posicional aprendido (``nn.Parameter`` ``[1, N + 1, d]``).
    4. ``depth`` bloques :class:`TransformerEncoderBlock` (*Pre-LN*).
    5. ``nn.LayerNorm`` final y cabeza ``Linear(d, num_classes)`` sobre el token 0.

    ``cls_token`` y ``pos_embedding`` se inicializan con ``torch.randn · 0.02``.
    """

    def __init__(
        self,
        image_size: int,
        patch_size: int,
        num_classes: int,
        d_model: int = 64,
        depth: int = 2,
        num_heads: int = 4,
        in_channels: int = 3,
    ) -> None:
        """
        Constructor de la clase.

        Args:
            image_size: lado de la imagen.
            patch_size: lado de cada parche.
            num_classes: número de clases.
            d_model: dimensión del modelo.
            depth: número de bloques.
            num_heads: cabezas de atención.
            in_channels: canales de la imagen.
        """

        super().__init__()

        # BEGIN SOLUTION
        self.patch_embed = PatchEmbedding(image_size, patch_size, in_channels, d_model)
        num_tokens = self.patch_embed.num_patches + 1
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_tokens, d_model) * 0.02)
        self.blocks = nn.ModuleList(
            [TransformerEncoderBlock(d_model, num_heads, norm_first=True) for _ in range(depth)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        # END SOLUTION

    def tokens(self, x: torch.Tensor) -> torch.Tensor:
        """
        Secuencia de tokens tras los bloques y la ``LayerNorm`` final.

        Args:
            x: imágenes. Dimensions: [batch, in_channels, image_size, image_size].

        Returns:
            Dimensions: [batch, num_patches + 1, d_model].
        """

        # BEGIN SOLUTION
        z = self.patch_embed(x)
        cls = self.cls_token.expand(z.shape[0], -1, -1)
        z = torch.cat([cls, z], dim=1) + self.pos_embedding
        for block in self.blocks:
            z = block(z)
        return self.norm(z)
        # END SOLUTION

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: imágenes. Dimensions: [batch, in_channels, image_size, image_size].

        Returns:
            Logits. Dimensions: [batch, num_classes].
        """

        # BEGIN SOLUTION
        return self.head(self.tokens(x)[:, 0])
        # END SOLUTION
