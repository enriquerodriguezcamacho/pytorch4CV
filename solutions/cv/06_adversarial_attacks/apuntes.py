"""
CV06 · Ejercicios de código del tema 6 de los apuntes (sección 6.5) — soluciones.

FGSM (Ej. 5) y VAT (Ej. 10) están en ``white_box.py`` y ``defenses.py``, y CutMix
(Ej. 8), en ``fundamentals/07_regularization_normalization``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader


class PGDAttack:
    """
    Ej. 6 · Ataque PGD ℓ∞ sin objetivo, con inicio aleatorio.

    Args del constructor:
        model: clasificador.
        epsilon: radio ``ε`` de la bola.
        alpha: tamaño de paso ``α``.
        steps: número de iteraciones ``T``.
    """

    def __init__(
        self, model: nn.Module, epsilon: float, alpha: float, steps: int
    ) -> None:
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps

    def perturb(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        Inicia en ``clamp(x + U(-ε, ε), 0, 1)`` y ejecuta las ``T`` iteraciones. Tras
        cada paso de signo, proyecta a la bola ``ε`` de ``x`` y recorta a [0, 1].
        No acumula gradientes en los parámetros del modelo.

        Args:
            x: imágenes en [0, 1]. Dimensions: [batch, channels, height, width].
            y: etiquetas. Dimensions: [batch].

        Returns:
            Imágenes adversarias, sin grafo. Mismas dimensiones que ``x``.
        """

        # BEGIN SOLUTION
        noise = torch.empty_like(x).uniform_(-self.epsilon, self.epsilon)
        x_adv = (x + noise).clamp(0, 1).detach()
        for _ in range(self.steps):
            x_adv.requires_grad_(True)
            loss = F.cross_entropy(self.model(x_adv), y)
            grad = torch.autograd.grad(loss, x_adv)[0]
            x_adv = x_adv.detach() + self.alpha * grad.sign()
            x_adv = x + (x_adv - x).clamp(-self.epsilon, self.epsilon)
            x_adv = x_adv.clamp(0, 1).detach()
        return x_adv
        # END SOLUTION


def boundary_attack(
    model: nn.Module,
    x: torch.Tensor,
    x_init: torch.Tensor,
    target_class: int,
    max_queries: int,
    delta: float = 0.1,
    epsilon: float = 0.1,
) -> torch.Tensor:
    """
    Ej. 7 · *Boundary Attack* simplificado y dirigido (algoritmo 2 de los apuntes).

    En cada una de las ``max_queries`` iteraciones:

    1. Paso ortogonal: ``η ~ N(0, I)`` (``torch.randn_like``), se elimina su
       componente en la dirección ``x_curr - x`` y
       ``x_cand = x_curr + δ · η⊥ / ||η⊥||``.
    2. Paso hacia el original: ``x_cand = x_cand - ε (x_cand - x)``, recortado a [0, 1].
    3. Se acepta si se predice ``target_class`` **y** se reduce la distancia ℓ2 a
       ``x``; entonces ``δ`` y ``ε`` se multiplican por 1.05 y, si no, por 0.95.

    Solo usa la clase predicha: ni *logits* ni gradientes.

    Args:
        model: clasificador.
        x: imagen original. Dimensions: [channels, height, width].
        x_init: imagen inicial, ya clasificada como ``target_class``. Mismas
            dimensiones.
        target_class: clase objetivo.
        max_queries: presupuesto de consultas.
        delta: paso ortogonal inicial.
        epsilon: paso inicial hacia el original.

    Returns:
        Imagen adversaria final.
    """

    # BEGIN SOLUTION
    x_curr = x_init.clone()
    with torch.no_grad():
        for _ in range(max_queries):
            direction = (x_curr - x).flatten()
            noise = torch.randn_like(x).flatten()
            norm_sq = direction.dot(direction)
            if norm_sq > 0:
                noise = noise - noise.dot(direction) / norm_sq * direction
            step = (delta * noise / noise.norm()).view_as(x)
            candidate = x_curr + step
            candidate = (candidate - epsilon * (candidate - x)).clamp(0, 1)
            predicted = model(candidate.unsqueeze(0)).argmax(dim=1).item()
            closer = (candidate - x).norm() < (x_curr - x).norm()
            if predicted == target_class and closer:
                x_curr = candidate
                delta, epsilon = delta * 1.05, epsilon * 1.05
            else:
                delta, epsilon = delta * 0.95, epsilon * 0.95
    return x_curr
    # END SOLUTION


def pgd_adversarial_training_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    attack: PGDAttack,
) -> float:
    """
    Ej. 9 · Una época de *adversarial training* con PGD.

    Los adversarios se generan en modo ``eval`` y ya salen desacoplados del grafo; el
    modelo vuelve a ``train``, se hace ``zero_grad`` y se actualiza con la pérdida
    sobre esos adversarios.

    Args:
        model: clasificador (el mismo que usa ``attack``).
        loader: pares ``(imágenes, etiquetas)``.
        optimizer: optimizador.
        attack: ataque PGD.

    Returns:
        Pérdida media de los *batches*.
    """

    # BEGIN SOLUTION
    total = 0.0
    for images, labels in loader:
        model.eval()
        adversarial = attack.perturb(images, labels)
        model.train()
        optimizer.zero_grad()
        loss = F.cross_entropy(model(adversarial), labels)
        loss.backward()
        optimizer.step()
        total += loss.item()
    return total / len(loader)
    # END SOLUTION


def robustness_curve(
    model: nn.Module,
    loader: DataLoader,
    budgets: list[float],
    steps: int = 10,
) -> list[float]:
    """
    Ej. 11 · *Accuracy* Top-1 frente a PGD ℓ∞ para varios presupuestos, con
    ``α = 2.5·ε / steps``. Con ``ε = 0`` coincide con la *clean accuracy*.

    Args:
        model: clasificador; se evalúa en modo ``eval``.
        loader: pares ``(imágenes, etiquetas)``.
        budgets: valores de ``ε``.
        steps: iteraciones de PGD.

    Returns:
        *Accuracy* para cada presupuesto.
    """

    # BEGIN SOLUTION
    model.eval()
    accuracies = []
    for epsilon in budgets:
        attack = PGDAttack(model, epsilon, 2.5 * epsilon / steps, steps)
        correct, total = 0, 0
        for images, labels in loader:
            adversarial = attack.perturb(images, labels)
            with torch.no_grad():
                correct += (model(adversarial).argmax(dim=1) == labels).sum().item()
            total += labels.shape[0]
        accuracies.append(correct / total)
    return accuracies
    # END SOLUTION
