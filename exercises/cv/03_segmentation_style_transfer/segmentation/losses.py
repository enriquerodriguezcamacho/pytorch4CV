"""
CV03 · Segmentación: pérdidas por píxel (CE con ignore_index, Dice, combinada) — ejercicios.
"""

import torch
import torch.nn.functional as F
from torch import nn


def pixel_cross_entropy(
    logits: torch.Tensor,
    target: torch.Tensor,
    ignore_index: int = 255,
    class_weights: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Cross-entropy por píxel implementada a mano (sin ``F.cross_entropy``).

    Se calcula ``-log softmax(logits)[y]`` en cada píxel, se descartan los píxeles con
    ``target == ignore_index`` y se promedia. Con pesos por clase, la media es
    ponderada: ``sum(w_y * nll) / sum(w_y)`` (igual que PyTorch).

    Args:
        logits: salida de la red. Dimensions: [batch, n_classes, height, width].
        target: clase de cada píxel (``int64``). Dimensions: [batch, height, width].
        ignore_index: valor de ``target`` que no cuenta (bordes, "void").
        class_weights: peso de cada clase o ``None``. Dimensions: [n_classes].

    Returns:
        Pérdida media (escalar). Si no hay ningún píxel válido devuelve 0.
    """

    # TODO: usa log_softmax(dim=1) y gather sobre los píxeles válidos
    raise NotImplementedError("TODO")


def soft_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    smooth: float = 1.0,
    ignore_index: int | None = None,
) -> torch.Tensor:
    """
    Soft Dice loss multiclase.

    Con ``p = softmax(logits)`` y ``g = one_hot(target)``, para cada clase ``k``::

        dice_k = (2 * sum(p_k * g_k) + smooth) / (sum(p_k) + sum(g_k) + smooth)

    donde las sumas recorren el lote y los píxeles válidos. La pérdida es
    ``1 - mean_k(dice_k)``.

    Args:
        logits: salida de la red. Dimensions: [batch, n_classes, height, width].
        target: clase de cada píxel (``int64``). Dimensions: [batch, height, width].
        smooth: término de suavizado (evita 0/0 y da gradiente estable).
        ignore_index: valor de ``target`` que se excluye (o ``None``).

    Returns:
        Pérdida en ``[0, 1]`` (escalar).
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


class CombinedLoss(nn.Module):
    """
    Pérdida combinada ``ce_weight * CE + dice_weight * Dice`` (muy usada en imagen
    médica, donde las clases están desbalanceadas).
    """

    def __init__(
        self, ce_weight: float = 1.0, dice_weight: float = 1.0, ignore_index: int = 255
    ) -> None:
        """
        Constructor de la clase.

        Args:
            ce_weight: peso de la cross-entropy.
            dice_weight: peso de la Dice loss.
            ignore_index: índice ignorado por ambas pérdidas.
        """

        super().__init__()

        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            logits: salida de la red. Dimensions: [batch, n_classes, height, width].
            target: clase de cada píxel. Dimensions: [batch, height, width].

        Returns:
            Pérdida combinada (escalar).
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")
