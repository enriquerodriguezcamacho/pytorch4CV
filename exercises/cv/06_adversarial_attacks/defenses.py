"""
CV06 · Ataques adversarios: defensas y evaluación (adversarial training, VAT,
robustez y transferibilidad) — ejercicios.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from .white_box import fgsm

Attack = Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor]


@torch.no_grad()
def accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    """
    Precisión del modelo (en modo ``eval``) sobre un lote. Ya dada.

    Args:
        model: clasificador.
        x: imágenes. Dimensions: [batch, channels, height, width].
        y: etiquetas. Dimensions: [batch].

    Returns:
        Fracción de aciertos.
    """

    model.eval()
    return (model(x).argmax(1) == y).float().mean().item()


def robust_accuracy(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, attack: Attack
) -> tuple[float, float]:
    """
    *Clean accuracy* y *robust accuracy* (precisión sobre las imágenes atacadas).

    El modelo se pone en modo ``eval`` **antes** de atacar.

    Args:
        model: clasificador.
        x: imágenes limpias. Dimensions: [batch, channels, height, width].
        y: etiquetas. Dimensions: [batch].
        attack: función ``attack(model, x, y) -> x_adv``.

    Returns:
        ``(clean_acc, robust_acc)``.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def transfer_success_rate(
    source: nn.Module, target: nn.Module, x: torch.Tensor, y: torch.Tensor, attack: Attack
) -> float:
    """
    Transferibilidad: se generan los adversarios sobre ``source`` y se mide qué
    fracción de ellos **engaña** a ``target`` (predicción distinta de ``y``),
    contando solo las muestras que ``target`` clasificaba bien sin atacar.

    Args:
        source: modelo con el que se genera el ataque.
        target: modelo atacado (no se usan sus gradientes).
        x: imágenes limpias. Dimensions: [batch, channels, height, width].
        y: etiquetas. Dimensions: [batch].
        attack: función ``attack(model, x, y) -> x_adv``.

    Returns:
        Fracción en [0, 1]; 0.0 si ``target`` no acierta ninguna limpia.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def adversarial_training_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    adv_weight: float = 0.5,
) -> float:
    """
    Un paso de *adversarial training* (Goodfellow et al.) con FGSM:

    ``L = (1 - w) · CE(f(x), y) + w · CE(f(x_adv), y)``.

    Los adversarios se generan con el modelo en modo ``eval`` (con
    :func:`fgsm`); después se pasa a ``train`` para calcular la pérdida y dar el
    paso del optimizador.

    Args:
        model: clasificador.
        optimizer: optimizador de ``model``.
        x: imágenes limpias. Dimensions: [batch, channels, height, width].
        y: etiquetas. Dimensions: [batch].
        epsilon: presupuesto de FGSM.
        adv_weight: peso ``w`` de la parte adversaria.

    Returns:
        Valor de la pérdida total.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def _l2_normalize(d: torch.Tensor) -> torch.Tensor:
    """Normaliza cada muestra a norma ℓ2 unidad. Ya dado."""

    norm = d.flatten(1).norm(dim=1).view(-1, *([1] * (d.dim() - 1)))
    return d / (norm + 1e-12)


def vat_loss(
    model: nn.Module,
    x: torch.Tensor,
    epsilon: float = 1.0,
    xi: float = 1e-6,
    n_power: int = 1,
) -> torch.Tensor:
    """
    *Virtual Adversarial Training* (Miyato et al.): pérdida no supervisada (no usa
    etiquetas).

    1. ``p = softmax(f(x))`` **sin gradiente** (es el objetivo).
    2. ``d ~ N(0, I)`` normalizada por muestra (``torch.randn_like``).
    3. ``n_power`` iteraciones de potencia: ``d ← normalizar(∇_d KL(p ‖ f(x + ξ d)))``.
    4. ``r_adv = ε · d`` y la pérdida es ``KL(p ‖ f(x + r_adv))``.

    La KL se calcula como ``F.kl_div(log_softmax(f(x')), p, reduction="batchmean")``.

    Args:
        model: clasificador.
        x: imágenes (sin etiquetar). Dimensions: [batch, channels, height, width].
        epsilon: norma de la perturbación adversaria.
        xi: tamaño de la perturbación en las iteraciones de potencia.
        n_power: número de iteraciones de potencia.

    Returns:
        Escalar derivable respecto a los parámetros del modelo.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")
