"""
CV06 · Ataques adversarios: ataques de caja blanca (FGSM, PGD, I-FGSM, C&W) — soluciones.

Convención: las imágenes están en [0, 1] y los ataques **no** modifican ni los
parámetros ni los gradientes del modelo (se usa ``torch.autograd.grad`` respecto
a la imagen, no ``loss.backward()`` sobre el modelo).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _input_gradient(
    model: nn.Module, x: torch.Tensor, y: torch.Tensor, loss_fn: nn.Module | None = None
) -> torch.Tensor:
    """
    Gradiente de la pérdida respecto a la entrada. Ya dado.

    Args:
        model: clasificador.
        x: imágenes. Dimensions: [batch, channels, height, width].
        y: etiquetas. Dimensions: [batch].
        loss_fn: pérdida; por defecto ``nn.CrossEntropyLoss()``.

    Returns:
        ``∇_x L(f(x), y)``. Dimensions: igual que ``x``.
    """

    loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()
    x = x.clone().detach().requires_grad_(True)
    loss = loss_fn(model(x), y)
    (grad,) = torch.autograd.grad(loss, x)
    return grad


def fgsm(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    targeted: bool = False,
) -> torch.Tensor:
    """
    *Fast Gradient Sign Method* (Goodfellow et al., 2015):

    * sin objetivo: ``x_adv = x + ε · sign(∇_x L(x, y))`` (sube la pérdida de la
      clase real ``y``);
    * con objetivo: ``x_adv = x - ε · sign(∇_x L(x, t))`` (baja la pérdida de la
      clase objetivo ``t``, que se pasa en ``y``).

    Después se recorta a [0, 1].

    Args:
        model: clasificador.
        x: imágenes limpias. Dimensions: [batch, channels, height, width].
        y: clase real (o clase objetivo si ``targeted``). Dimensions: [batch].
        epsilon: tamaño de la perturbación (norma ℓ∞).
        targeted: tipo de ataque.

    Returns:
        Imágenes adversarias sin gradiente. Dimensions: igual que ``x``.
    """

    # BEGIN SOLUTION
    sign = -1.0 if targeted else 1.0
    grad = _input_gradient(model, x, y)
    return (x + sign * epsilon * grad.sign()).clamp(0.0, 1.0).detach()
    # END SOLUTION


def pgd_linf(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    alpha: float,
    steps: int,
    random_start: bool = False,
    targeted: bool = False,
) -> torch.Tensor:
    """
    *Projected Gradient Descent* en ℓ∞ (Madry et al., 2018). Sin ``random_start``
    es exactamente I-FGSM / BIM.

    1. Inicio: ``x' = x`` o, con ``random_start``,
       ``x' = clamp(x + U(-ε, ε), 0, 1)`` (con ``torch.empty_like(x).uniform_``).
    2. ``steps`` veces: paso de signo de tamaño ``alpha`` (``+`` sin objetivo,
       ``-`` con objetivo), proyección a la bola
       ``x + clamp(x' - x, -ε, ε)`` y recorte a [0, 1].

    Args:
        model: clasificador.
        x: imágenes limpias. Dimensions: [batch, channels, height, width].
        y: clase real (o clase objetivo si ``targeted``). Dimensions: [batch].
        epsilon: radio de la bola ℓ∞.
        alpha: tamaño de cada paso.
        steps: número de iteraciones.
        random_start: si se empieza en un punto aleatorio de la bola.
        targeted: tipo de ataque.

    Returns:
        Imágenes adversarias sin gradiente. Dimensions: igual que ``x``.
    """

    # BEGIN SOLUTION
    sign = -1.0 if targeted else 1.0
    x = x.detach()
    x_adv = x.clone()
    if random_start:
        x_adv = (x_adv + torch.empty_like(x).uniform_(-epsilon, epsilon)).clamp(0.0, 1.0)
    for _ in range(steps):
        grad = _input_gradient(model, x_adv, y)
        x_adv = x_adv + sign * alpha * grad.sign()
        x_adv = (x + (x_adv - x).clamp(-epsilon, epsilon)).clamp(0.0, 1.0)
    return x_adv.detach()
    # END SOLUTION


def pgd_l2(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    alpha: float,
    steps: int,
) -> torch.Tensor:
    """
    PGD en ℓ2 (sin objetivo, sin inicio aleatorio).

    En cada paso, **por muestra**: ``x' ← x' + α · g / ||g||₂`` (``g`` el
    gradiente; se suma ``1e-12`` a la norma) y se proyecta
    ``δ = x' - x`` a la bola ℓ2: ``δ ← δ · min(1, ε / ||δ||₂)``. Luego se recorta
    ``x + δ`` a [0, 1].

    Args:
        model: clasificador.
        x: imágenes limpias. Dimensions: [batch, channels, height, width].
        y: clase real. Dimensions: [batch].
        epsilon: radio de la bola ℓ2.
        alpha: tamaño de cada paso (en norma ℓ2).
        steps: número de iteraciones.

    Returns:
        Imágenes adversarias sin gradiente. Dimensions: igual que ``x``.
    """

    # BEGIN SOLUTION
    x = x.detach()
    x_adv = x.clone()
    view = (-1,) + (1,) * (x.dim() - 1)
    for _ in range(steps):
        grad = _input_gradient(model, x_adv, y)
        grad_norm = grad.flatten(1).norm(dim=1).view(view) + 1e-12
        x_adv = x_adv + alpha * grad / grad_norm
        delta = x_adv - x
        delta_norm = delta.flatten(1).norm(dim=1).view(view)
        factor = torch.clamp(epsilon / (delta_norm + 1e-12), max=1.0)
        x_adv = (x + delta * factor).clamp(0.0, 1.0)
    return x_adv.detach()
    # END SOLUTION


class IFGSMAttack:
    """
    I-FGSM sobre **una** imagen, con la estructura del examen final de diciembre.

    Guarda la imagen original en ``self.img`` y la adversaria en
    ``self.perturbed_img`` (empieza siendo una copia), y el historial de pérdidas
    en ``self.loss_history``.
    """

    def __init__(
        self,
        img: torch.Tensor,
        model: nn.Module,
        true_class_idx: int,
        loss_fn: nn.Module | None = None,
        targeted: bool = False,
        target_class_idx: int | None = None,
    ) -> None:
        """
        Constructor de la clase.

        Args:
            img: imagen a atacar. Dimensions: [channels, height, width].
            model: modelo a atacar (se pone en modo ``eval``).
            true_class_idx: clase real.
            loss_fn: pérdida; por defecto ``nn.CrossEntropyLoss()``.
            targeted: si el ataque es dirigido.
            target_class_idx: clase objetivo (obligatoria si ``targeted``).

        Raises:
            ValueError: si ``targeted`` y no se da ``target_class_idx``.
        """

        # BEGIN SOLUTION
        if targeted and target_class_idx is None:
            raise ValueError("Un ataque dirigido necesita target_class_idx.")
        self.img = img.clone().detach()
        self.perturbed_img = self.img.clone()
        self.model = model.eval()
        self.true_class_idx = true_class_idx
        self.loss_fn = loss_fn if loss_fn is not None else nn.CrossEntropyLoss()
        self.targeted = targeted
        self.target_class_idx = target_class_idx
        self.loss_history: list[float] = []
        # END SOLUTION

    def _compute_loss(self, img: torch.Tensor) -> torch.Tensor:
        """
        Pérdida respecto a la clase objetivo (si ``targeted``) o a la real.

        Args:
            img: imagen (con gradiente). Dimensions: [channels, height, width].

        Returns:
            Escalar con grafo (para derivar respecto a ``img``).
        """

        # BEGIN SOLUTION
        idx = self.target_class_idx if self.targeted else self.true_class_idx
        logits = self.model(img.unsqueeze(0))
        return self.loss_fn(logits, torch.tensor([idx], device=img.device))
        # END SOLUTION

    def _gradient_step(self, epsilon: float, alpha: float) -> None:
        """
        Un paso de I-FGSM sobre ``self.perturbed_img``: calcula la pérdida (y la
        añade a ``self.loss_history``), da un paso de signo de tamaño ``alpha``
        (descenso si es dirigido, ascenso si no), proyecta a
        ``[img - ε, img + ε]`` y recorta a [0, 1].

        Args:
            epsilon: máxima perturbación ℓ∞.
            alpha: tamaño del paso.
        """

        # BEGIN SOLUTION
        img = self.perturbed_img.clone().detach().requires_grad_(True)
        loss = self._compute_loss(img)
        (grad,) = torch.autograd.grad(loss, img)
        self.loss_history.append(loss.item())
        sign = -1.0 if self.targeted else 1.0
        step = img.detach() + sign * alpha * grad.sign()
        step = torch.max(torch.min(step, self.img + epsilon), self.img - epsilon)
        self.perturbed_img = step.clamp(0.0, 1.0).detach()
        # END SOLUTION

    def get_perturbation(self) -> torch.Tensor:
        """
        Perturbación actual ``perturbed_img - img``. Ya dada.

        Returns:
            Dimensions: [channels, height, width].
        """

        return self.perturbed_img - self.img

    def get_norm(self, norm_type: str = "linf") -> float:
        """
        Norma de la perturbación: ``"linf"`` (máximo absoluto), ``"l2"`` o ``"l1"``.

        Args:
            norm_type: tipo de norma.

        Returns:
            Valor de la norma.

        Raises:
            ValueError: si la norma no es válida.
        """

        # BEGIN SOLUTION
        delta = self.get_perturbation().flatten()
        match norm_type:
            case "linf":
                return delta.abs().max().item()
            case "l2":
                return delta.norm(p=2).item()
            case "l1":
                return delta.abs().sum().item()
            case _:
                raise ValueError(f"Norma desconocida: {norm_type!r}.")
        # END SOLUTION

    @torch.no_grad()
    def get_attack_success(self) -> bool:
        """
        Éxito del ataque: si es dirigido, el modelo predice la clase objetivo; si no,
        el modelo **ya no** predice la clase real.

        Returns:
            ``True`` si el ataque ha tenido éxito.
        """

        # BEGIN SOLUTION
        pred = int(self.model(self.perturbed_img.unsqueeze(0)).argmax(1).item())
        if self.targeted:
            return pred == self.target_class_idx
        return pred != self.true_class_idx
        # END SOLUTION

    def fit(
        self, epochs: int = 100, epsilon: float = 8 / 255, alpha: float = 1 / 255
    ) -> torch.Tensor:
        """
        Ejecuta ``epochs`` pasos de :meth:`_gradient_step`.

        Args:
            epochs: número de iteraciones.
            epsilon: máxima perturbación ℓ∞.
            alpha: tamaño del paso.

        Returns:
            Imagen adversaria. Dimensions: [channels, height, width].
        """

        # BEGIN SOLUTION
        for _ in range(epochs):
            self._gradient_step(epsilon, alpha)
        return self.perturbed_img
        # END SOLUTION


def carlini_wagner_l2(
    model: nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    c: float = 1.0,
    kappa: float = 0.0,
    steps: int = 100,
    lr: float = 0.01,
) -> torch.Tensor:
    """
    Ataque de Carlini & Wagner en ℓ2 (dirigido), con el cambio de variable
    ``x' = ½ (tanh(w) + 1)`` que mantiene ``x'`` en [0, 1] sin recortes.

    Se minimiza con Adam sobre ``w`` (inicializado en
    ``atanh(clamp(2x - 1, -1 + 1e-6, 1 - 1e-6))``, de modo que ``x'₀ ≈ x``):

    ``Σ_batch ||x' - x||²₂ + c · f(x')``, con
    ``f(x') = max(max_{i≠t} Z_i - Z_t, -κ)`` y ``Z`` los logits.

    Args:
        model: clasificador.
        x: imágenes limpias. Dimensions: [batch, channels, height, width].
        target: clase objetivo. Dimensions: [batch].
        c: peso del término de clasificación.
        kappa: margen de confianza ``κ``.
        steps: iteraciones de Adam.
        lr: *learning rate* de Adam.

    Returns:
        Imágenes adversarias en [0, 1], sin gradiente. Dimensions: igual que ``x``.
    """

    # BEGIN SOLUTION
    x = x.detach()
    w = torch.atanh((2 * x - 1).clamp(-1 + 1e-6, 1 - 1e-6)).requires_grad_(True)
    optimizer = torch.optim.Adam([w], lr=lr)
    one_hot = F.one_hot(target, num_classes=model(x[:1]).shape[1]).bool()
    for _ in range(steps):
        x_adv = 0.5 * (torch.tanh(w) + 1)
        logits = model(x_adv)
        target_logit = logits[one_hot]
        other_logit = logits.masked_fill(one_hot, float("-inf")).amax(dim=1)
        f = torch.clamp(other_logit - target_logit, min=-kappa)
        loss = ((x_adv - x) ** 2).flatten(1).sum(1) + c * f
        optimizer.zero_grad()
        loss.sum().backward()
        optimizer.step()
    return (0.5 * (torch.tanh(w) + 1)).detach()
    # END SOLUTION


def linear_worst_case(weight: torch.Tensor, epsilon: float) -> tuple[torch.Tensor, float]:
    """
    Ejemplo de juguete del capítulo (modelo lineal ``f(x) = wᵀx + b``): la
    perturbación ``δ`` con ``||δ||∞ <= ε`` que más sube la salida es
    ``δ = ε · sign(w)``, y el cambio es ``wᵀδ = ε ||w||₁``, que crece con la
    dimensión aunque ``ε`` sea imperceptible.

    Args:
        weight: pesos ``w``. Dimensions: [n].
        epsilon: presupuesto ℓ∞.

    Returns:
        ``(δ, cambio de la salida)``. Dimensions: [n] y escalar ``float``.
    """

    # BEGIN SOLUTION
    delta = epsilon * weight.sign()
    return delta, float((weight * delta).sum())
    # END SOLUTION
