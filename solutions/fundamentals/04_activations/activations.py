"""
F04 · Funciones de activación — soluciones.

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

    # BEGIN SOLUTION
    return x.clamp_min(0.0)
    # END SOLUTION


def leaky_relu(x: torch.Tensor, negative_slope: float = 0.01) -> torch.Tensor:
    """
    ``LeakyReLU(x) = x`` si ``x > 0``; ``negative_slope · x`` si no.

    Args:
        x: entrada. Dimensions: [*].
        negative_slope: pendiente en la parte negativa (0.2 en discriminadores GAN).

    Returns:
        Salida. Dimensions: [*].
    """

    # BEGIN SOLUTION
    return torch.where(x > 0, x, negative_slope * x)
    # END SOLUTION


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

    # BEGIN SOLUTION
    return torch.where(x > 0, x, alpha * torch.expm1(x.clamp_max(0.0)))
    # END SOLUTION


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

    # BEGIN SOLUTION
    z = torch.exp(-x.abs())
    return torch.where(x >= 0, 1.0 / (1.0 + z), z / (1.0 + z))
    # END SOLUTION


def sigmoid_derivative(x: torch.Tensor) -> torch.Tensor:
    """
    Derivada a mano de la sigmoide: ``σ'(x) = σ(x)(1 - σ(x))`` (máximo 0.25 en 0).

    Args:
        x: entrada. Dimensions: [*].

    Returns:
        ``σ'(x)``. Dimensions: [*].
    """

    # BEGIN SOLUTION
    s = sigmoid(x)
    return s * (1.0 - s)
    # END SOLUTION


def tanh(x: torch.Tensor) -> torch.Tensor:
    """
    ``tanh(x) = 2σ(2x) - 1`` (estable gracias a ``sigmoid``).

    Args:
        x: entrada. Dimensions: [*].

    Returns:
        Salida en (-1, 1). Dimensions: [*].
    """

    # BEGIN SOLUTION
    return 2.0 * sigmoid(2.0 * x) - 1.0
    # END SOLUTION


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

    # BEGIN SOLUTION
    if approximate == "none":
        return 0.5 * x * (1.0 + torch.erf(x / math.sqrt(2.0)))
    if approximate == "tanh":
        inner = math.sqrt(2.0 / math.pi) * (x + 0.044715 * x**3)
        return 0.5 * x * (1.0 + tanh(inner))
    raise ValueError(f"approximate desconocido: {approximate}")
    # END SOLUTION


def silu(x: torch.Tensor) -> torch.Tensor:
    """
    SiLU / Swish: ``x · σ(x)`` (YOLOv5/v8, EfficientNet).

    Args:
        x: entrada. Dimensions: [*].

    Returns:
        Salida. Dimensions: [*].
    """

    # BEGIN SOLUTION
    return x * sigmoid(x)
    # END SOLUTION


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

    # BEGIN SOLUTION
    t = beta * x
    return (t.clamp_min(0.0) + torch.log1p(torch.exp(-t.abs()))) / beta
    # END SOLUTION


def mish(x: torch.Tensor) -> torch.Tensor:
    """
    Mish (Misra, 2019), usada en YOLOv4: ``x · tanh(softplus(x))``.

    Args:
        x: entrada. Dimensions: [*].

    Returns:
        Salida. Dimensions: [*].
    """

    # BEGIN SOLUTION
    return x * tanh(softplus(x))
    # END SOLUTION


def hardsigmoid(x: torch.Tensor) -> torch.Tensor:
    """
    ``Hardsigmoid(x) = clip(x/6 + 1/2, 0, 1)`` (definición de PyTorch / MobileNetV3).

    Args:
        x: entrada. Dimensions: [*].

    Returns:
        Salida en [0, 1]. Dimensions: [*].
    """

    # BEGIN SOLUTION
    return (x / 6.0 + 0.5).clamp(0.0, 1.0)
    # END SOLUTION


def hardswish(x: torch.Tensor) -> torch.Tensor:
    """
    ``Hardswish(x) = x · ReLU6(x + 3) / 6`` (MobileNetV3).

    Args:
        x: entrada. Dimensions: [*].

    Returns:
        Salida. Dimensions: [*].
    """

    # BEGIN SOLUTION
    return x * (x + 3.0).clamp(0.0, 6.0) / 6.0
    # END SOLUTION


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

    # BEGIN SOLUTION
    shifted = x - x.amax(dim=dim, keepdim=True).detach()
    e = torch.exp(shifted)
    return e / e.sum(dim=dim, keepdim=True)
    # END SOLUTION


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

    # BEGIN SOLUTION
    shifted = x - x.amax(dim=dim, keepdim=True).detach()
    return shifted - torch.log(torch.exp(shifted).sum(dim=dim, keepdim=True))
    # END SOLUTION


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

        # BEGIN SOLUTION
        self.weight = nn.Parameter(torch.full((num_parameters,), float(init)))
        # END SOLUTION

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: entrada. Dimensions: [batch, channels, *] (o [*] si
                ``num_parameters == 1``).

        Returns:
            Salida. Dimensions: iguales a la entrada.
        """

        # BEGIN SOLUTION
        if self.num_parameters == 1:
            a = self.weight
        else:
            shape = [1] * x.dim()
            shape[1] = self.num_parameters
            a = self.weight.view(shape)
        return torch.where(x > 0, x, a * x)
        # END SOLUTION
