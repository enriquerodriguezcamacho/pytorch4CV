"""
CV08 · Atención y Transformers: atención, *multi-head*, codificación posicional y
bloque *encoder* — ejercicios.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

# --------------------------------------------------------------------------- #
# Nadaraya-Watson: la atención antes de la atención                           #
# --------------------------------------------------------------------------- #


def nadaraya_watson(
    queries: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    kernel: Literal["gaussian", "boxcar", "epanechnikov"] = "gaussian",
    bandwidth: float = 1.0,
) -> torch.Tensor:
    """
    Regresión de Nadaraya-Watson, la forma más simple de atención:

    ``f(q) = Σ_i K((q - k_i) / h) · v_i / Σ_i K((q - k_i) / h)``

    con ``K(u) = exp(-u²/2)`` (gaussiano; la constante se cancela),
    ``K(u) = 1[|u| < 1]`` (*boxcar*) o ``K(u) = max(1 - |u|, 0)`` (Epanechnikov).
    Si para una *query* todos los pesos son 0 se devuelve 0 (se suma ``1e-12`` al
    denominador).

    Args:
        queries: puntos donde se evalúa. Dimensions: [n].
        keys: entradas de entrenamiento ``x_i``. Dimensions: [m].
        values: salidas de entrenamiento ``y_i``. Dimensions: [m].
        kernel: tipo de kernel.
        bandwidth: ancho de banda ``h``.

    Returns:
        Predicciones. Dimensions: [n].

    Raises:
        ValueError: si el kernel no es válido.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


# --------------------------------------------------------------------------- #
# Scaled dot-product attention                                                #
# --------------------------------------------------------------------------- #


def causal_mask(size: int) -> torch.Tensor:
    """
    Máscara causal: la posición ``i`` solo puede atender a ``j <= i``.

    Args:
        size: longitud de la secuencia.

    Returns:
        Máscara booleana triangular inferior (``True`` = se puede atender).
        Dimensions: [size, size].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    ``Attention(Q, K, V) = softmax(Q Kᵀ / sqrt(d)) V``, con ``d`` la dimensión de
    las *queries*.

    Las posiciones con ``mask == False`` se rellenan con ``-inf`` (``masked_fill``)
    **antes** del softmax.

    Args:
        q: *queries*. Dimensions: [..., n, d].
        k: *keys*. Dimensions: [..., m, d].
        v: *values*. Dimensions: [..., m, d_v].
        mask: máscara booleana que se difunde a [..., n, m] (``True`` = atender).

    Returns:
        Salida y pesos de atención (cada fila suma 1). Dimensions: [..., n, d_v]
        y [..., n, m].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


class MultiHeadAttention(nn.Module):
    """
    *Multi-head attention* desde cero:

    ``MHA(Q, K, V) = [head_1; ...; head_h] W_O``,
    ``head_i = Attention(Q W_i^Q, K W_i^K, V W_i^V)``.

    Las proyecciones de todas las cabezas se hacen a la vez con tres
    ``nn.Linear(d_model, d_model)`` (``w_q``, ``w_k``, ``w_v``) y la salida con
    ``w_o``. Para separar las cabezas: ``view(B, L, h, d_k).transpose(1, 2)``
    (``d_k = d_model / h``); para juntarlas, la operación inversa.
    """

    def __init__(self, d_model: int, num_heads: int) -> None:
        """
        Constructor de la clase.

        Args:
            d_model: dimensión del modelo.
            num_heads: número de cabezas ``h``.

        Raises:
            ValueError: si ``num_heads`` no divide a ``d_model``.
        """

        super().__init__()

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Separa las cabezas.

        Args:
            x: Dimensions: [batch, length, d_model].

        Returns:
            Dimensions: [batch, num_heads, length, d_k].
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """
        Junta las cabezas (inversa de :meth:`_split_heads`).

        Args:
            x: Dimensions: [batch, num_heads, length, d_k].

        Returns:
            Dimensions: [batch, length, d_model].
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            query: Dimensions: [batch, n, d_model].
            key: Dimensions: [batch, m, d_model].
            value: Dimensions: [batch, m, d_model].
            mask: máscara booleana que se difunde a [batch, heads, n, m].

        Returns:
            Salida y pesos de atención por cabeza. Dimensions: [batch, n, d_model]
            y [batch, num_heads, n, m].
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")


def sinusoidal_positional_encoding(length: int, d_model: int) -> torch.Tensor:
    """
    Codificación posicional sinusoidal (Vaswani et al., 2017):

    ``PE[pos, 2i] = sin(pos / 10000^(2i/d))``, ``PE[pos, 2i+1] = cos(pos / 10000^(2i/d))``.

    Args:
        length: número de posiciones.
        d_model: dimensión (par).

    Returns:
        Dimensions: [length, d_model].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


class TransformerEncoderBlock(nn.Module):
    """
    Bloque *encoder* con FFN ``Linear(d, 4d) → GELU → Linear(4d, d)``.

    * ``norm_first=True`` (*Pre-LN*, ViT): ``x = x + MHA(LN₁(x))``;
      ``x = x + FFN(LN₂(x))``.
    * ``norm_first=False`` (*Post-LN*, Transformer original):
      ``x = LN₁(x + MHA(x))``; ``x = LN₂(x + FFN(x))``.

    Atributos: ``attn`` (:class:`MultiHeadAttention`), ``norm1``, ``norm2``
    (``nn.LayerNorm``) y ``ffn`` (``nn.Sequential``).
    """

    def __init__(self, d_model: int, num_heads: int, norm_first: bool = True) -> None:
        """
        Constructor de la clase.

        Args:
            d_model: dimensión del modelo.
            num_heads: número de cabezas.
            norm_first: *Pre-LN* (``True``) o *Post-LN*.
        """

        super().__init__()

        self.norm_first = norm_first
        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Forward pass (autoatención).

        Args:
            x: Dimensions: [batch, length, d_model].
            mask: máscara opcional para la atención.

        Returns:
            Dimensions: [batch, length, d_model].
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")


def hard_distillation_loss(
    student_logits: torch.Tensor, teacher_logits: torch.Tensor, targets: torch.Tensor
) -> torch.Tensor:
    """
    *Hard distillation* (DeiT): ``½ CE(z_s, y) + ½ CE(z_s, argmax z_t)``. La versión
    *soft* (con temperatura) está en F05 (``DistillationLoss``).

    Args:
        student_logits: Dimensions: [batch, num_classes].
        teacher_logits: Dimensions: [batch, num_classes].
        targets: etiquetas reales. Dimensions: [batch].

    Returns:
        Escalar (sin gradiente hacia el profesor).
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")
