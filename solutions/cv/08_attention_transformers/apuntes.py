"""
CV08 · Ejercicios de código del tema 8 de los apuntes (sección 8.4) — soluciones.

*Patch embedding* (Ej. 6) y el *encoder* Pre-LN (Ej. 7) están en ``vit.py`` y
``attention.py``.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn.functional as F


def batched_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    additive_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Ej. 5 · Atención multi-cabeza por *batch* con máscara **aditiva** (0 = permitido,
    ``-inf`` = bloqueado), que se suma a los *scores* antes del softmax.

    Args:
        q: Dimensions: [B, h, N_q, d_k].
        k: Dimensions: [B, h, N_k, d_k].
        v: Dimensions: [B, h, N_k, d_v].
        additive_mask: máscara difundible a [B, h, N_q, N_k].

    Returns:
        Salida. Dimensions: [B, h, N_q, d_v].
    """

    # BEGIN SOLUTION
    scores = q @ k.transpose(-2, -1) / math.sqrt(k.shape[-1])
    if additive_mask is not None:
        scores = scores + additive_mask
    return torch.softmax(scores, dim=-1) @ v
    # END SOLUTION


def deit_loss(
    cls_logits: torch.Tensor,
    dist_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    mode: Literal["hard", "soft"] = "hard",
    lam: float = 0.5,
) -> torch.Tensor:
    """
    Ej. 8 · Pérdida de DeiT, con el profesor como objetivo fijo (``detach``):

    * ``hard``: ``λ CE(z_cls, y) + (1 - λ) CE(z_dist, argmax z_t)``.
    * ``soft``: ``λ CE(z_cls, y) + (1 - λ) KL(p_t || p_dist)``, con KL ``batchmean``.

    Args:
        cls_logits: *logits* de clasificación. Dimensions: [batch, num_classes].
        dist_logits: *logits* de destilación. Dimensions: [batch, num_classes].
        teacher_logits: *logits* del profesor. Dimensions: [batch, num_classes].
        labels: etiquetas. Dimensions: [batch].
        mode: ``"hard"`` o ``"soft"``.
        lam: peso ``λ`` en [0, 1].

    Returns:
        Pérdida escalar.

    Raises:
        ValueError: si ``mode`` no es válido.
    """

    # BEGIN SOLUTION
    teacher = teacher_logits.detach()
    if mode == "hard":
        distillation = F.cross_entropy(dist_logits, teacher.argmax(dim=1))
    elif mode == "soft":
        distillation = F.kl_div(
            F.log_softmax(dist_logits, dim=1),
            F.softmax(teacher, dim=1),
            reduction="batchmean",
        )
    else:
        raise ValueError(f"mode debe ser 'hard' o 'soft', no {mode!r}.")
    return lam * F.cross_entropy(cls_logits, labels) + (1 - lam) * distillation
    # END SOLUTION


def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    """
    Ej. 10 · Divide un mapa en ventanas sin solape (Swin), de izquierda a derecha y de
    arriba abajo.

    Args:
        x: mapa. Dimensions: [B, H, W, C]; ``H`` y ``W`` divisibles por ``M``.
        window_size: lado ``M`` de la ventana.

    Returns:
        Ventanas. Dimensions: [B·H·W / M², M, M, C].
    """

    # BEGIN SOLUTION
    batch, height, width, channels = x.shape
    size = window_size
    x = x.view(batch, height // size, size, width // size, size, channels)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, size, size, channels)
    # END SOLUTION


def window_reverse(windows: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """
    Ej. 10 · Operación inversa de :func:`window_partition`.

    Args:
        windows: ventanas. Dimensions: [B·H·W / M², M, M, C].
        height: alto ``H`` del mapa original.
        width: ancho ``W`` del mapa original.

    Returns:
        Mapa original. Dimensions: [B, H, W, C].
    """

    # BEGIN SOLUTION
    size = windows.shape[1]
    channels = windows.shape[-1]
    rows, cols = height // size, width // size
    batch = windows.shape[0] // (rows * cols)
    x = windows.view(batch, rows, cols, size, size, channels)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(batch, height, width, channels)
    # END SOLUTION
