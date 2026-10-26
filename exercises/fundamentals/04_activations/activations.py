"""
F04 · Funciones de activación — ejercicios.

Implementaciones desde cero (solo operaciones de tensor, sin ``torch.nn.functional``
ni ``torch.relu``/``torch.sigmoid``/``torch.softmax``...) y numéricamente estables
de las activaciones más usadas en visión por computador.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn


def relu(x: torch.Tensor) -> torch.Tensor:
    """
    ``ReLU(x) = max(0, x)``.

    Args:
        x: entrada. Dimensions: [*].

    Returns:
        Salida. Dimensions: [*].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def leaky_relu(x: torch.Tensor, negative_slope: float = 0.01) -> torch.Tensor:
    """
    ``LeakyReLU(x) = x`` si ``x > 0``; ``negative_slope · x`` si no.

    Args:
        x: entrada. Dimensions: [*].
        negative_slope: pendiente en la parte negativa (0.2 en discriminadores GAN).

    Returns:
        Salida. Dimensions: [*].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def elu(x: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
    """
    ``ELU(x) = x`` si ``x > 0``; ``α(eˣ - 1)`` si no.

    Cuidado: ``torch.where`` evalúa **las dos ramas**. Si calculas ``exp(x)``
    con ``x = 1000`` sale ``inf`` y en el backward ``0 · inf = nan``. Usa
    ``expm1`` sobre ``x`` recortado a ``≤ 0``.

    Args:
        x: entrada. Dimensions: [*].
        alpha: valor de saturación negativa (``-α``).

    Returns:
        Salida. Dimensions: [*].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def sigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    Sigmoide estable ``σ(x) = 1 / (1 + e^{-x})``.

    Con ``z = e^{-|x|} ∈ (0, 1]``: ``σ(x) = 1/(1+z)`` si ``x ≥ 0`` y ``z/(1+z)``
    si ``x < 0``. Nunca se calcula la exponencial de un número positivo grande.

    Args:
        x: entrada. Dimensions: [*].

    Returns:
        Salida en (0, 1). Dimensions: [*].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def sigmoid_derivative(x: torch.Tensor) -> torch.Tensor:
    """
    Derivada a mano de la sigmoide: ``σ'(x) = σ(x)(1 - σ(x))`` (máximo 0.25 en 0).

    Args:
        x: entrada. Dimensions: [*].

    Returns:
        ``σ'(x)``. Dimensions: [*].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def tanh(x: torch.Tensor) -> torch.Tensor:
    """
    ``tanh(x) = 2σ(2x) - 1`` (estable gracias a ``sigmoid``).

    Args:
        x: entrada. Dimensions: [*].

    Returns:
        Salida en (-1, 1). Dimensions: [*].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def gelu(
    x: torch.Tensor, approximate: Literal["none", "tanh"] = "none"
) -> torch.Tensor:
    """
    GELU (Hendrycks & Gimpel, 2016), activación de ViT y BERT.

    * Exacta: ``GELU(x) = x · Φ(x) = 0.5 · x · (1 + erf(x / √2))``.
    * Aproximada: ``0.5 · x · (1 + tanh(√(2/π) · (x + 0.044715 x³)))``.

    Args:
        x: entrada. Dimensions: [*].
        approximate: ``"none"`` (exacta, con ``torch.erf``) o ``"tanh"``.

    Returns:
        Salida. Dimensions: [*].

    Raises:
        ValueError: si ``approximate`` no es ``"none"`` ni ``"tanh"``.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def silu(x: torch.Tensor) -> torch.Tensor:
    """
    SiLU / Swish: ``x · σ(x)`` (YOLOv5/v8, EfficientNet).

    Args:
        x: entrada. Dimensions: [*].

    Returns:
        Salida. Dimensions: [*].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def softplus(x: torch.Tensor, beta: float = 1.0) -> torch.Tensor:
    """
    Softplus estable: ``(1/β) · log(1 + e^{βx})``.

    Identidad estable: ``log(1 + eᵗ) = max(t, 0) + log1p(e^{-|t|})`` con ``t = βx``.

    Args:
        x: entrada. Dimensions: [*].
        beta: inclinación (``β → ∞`` se parece a ReLU).

    Returns:
        Salida positiva. Dimensions: [*].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def mish(x: torch.Tensor) -> torch.Tensor:
    """
    Mish (Misra, 2019), usada en YOLOv4: ``x · tanh(softplus(x))``.

    Args:
        x: entrada. Dimensions: [*].

    Returns:
        Salida. Dimensions: [*].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def hardsigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    ``Hardsigmoid(x) = clip(x/6 + 1/2, 0, 1)`` (definición de PyTorch / MobileNetV3).

    Args:
        x: entrada. Dimensions: [*].

    Returns:
        Salida en [0, 1]. Dimensions: [*].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def hardswish(x: torch.Tensor) -> torch.Tensor:
    """
    ``Hardswish(x) = x · ReLU6(x + 3) / 6`` (MobileNetV3).

    Args:
        x: entrada. Dimensions: [*].

    Returns:
        Salida. Dimensions: [*].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Softmax estable: ``exp(x - max) / Σ exp(x - max)`` a lo largo de ``dim``.

    Restar el máximo no cambia el resultado (se cancela) y evita ``exp(1000) = inf``.

    Args:
        x: logits. Dimensions: [*].
        dim: eje de las clases (``1`` para ``[batch, classes, H, W]``).

    Returns:
        Probabilidades que suman 1 en ``dim``. Dimensions: [*].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def log_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Log-softmax estable: ``x - max - log Σ exp(x - max)`` (truco *log-sum-exp*).

    No calcules ``log(softmax(x))``: con logits extremos da ``log(0) = -inf``.

    Args:
        x: logits. Dimensions: [*].
        dim: eje de las clases.

    Returns:
        Log-probabilidades (≤ 0). Dimensions: [*].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


class PReLU(nn.Module):
    """
    PReLU (He et al., 2015): LeakyReLU con pendiente negativa **aprendible**.

    ``PReLU(x) = x`` si ``x > 0``; ``a · x`` si no. Con ``num_parameters = C`` hay
    una pendiente por canal (eje 1 de la entrada), como ``nn.PReLU``.
    """

    def __init__(self, num_parameters: int = 1, init: float = 0.25) -> None:
        """
        Args:
            num_parameters: 1 (pendiente compartida) o número de canales.
            init: valor inicial de la pendiente.
        """

        super().__init__()
        self.num_parameters = num_parameters

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: entrada. Dimensions: [batch, channels, *] (o [*] si
                ``num_parameters == 1``).

        Returns:
            Salida. Dimensions: iguales a la entrada.
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")
