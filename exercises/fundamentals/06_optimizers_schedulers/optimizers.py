"""
F06 · Optimizadores y schedulers — ejercicios.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable, Iterable
from typing import Literal

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler

NORM_LAYERS: tuple[type[nn.Module], ...] = (
    nn.modules.batchnorm._NormBase,  # BatchNorm*, InstanceNorm*
    nn.GroupNorm,
    nn.LayerNorm,
    nn.LocalResponseNorm,
)


class SGDMomentum(Optimizer):
    """
    SGD con momentum (heavy ball), Nesterov y weight decay L2.

    Misma semántica que ``torch.optim.SGD`` con ``dampening=0``:

    ``g ← ∇θ + λ θ`` ; ``b ← μ b + g`` (en el primer paso ``b = g``) ;
    ``d = g + μ b`` si Nesterov, ``d = b`` si no ; ``θ ← θ - η d``.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict],
        lr: float = 1e-2,
        momentum: float = 0.9,
        weight_decay: float = 0.0,
        nesterov: bool = False,
    ) -> None:
        """
        Constructor.

        Args:
            params: parámetros o grupos de parámetros (lista de dicts).
            lr: learning rate ``η``.
            momentum: coeficiente ``μ ∈ [0, 1)``.
            weight_decay: coeficiente L2 ``λ ≥ 0`` (se suma al gradiente).
            nesterov: si se usa el momentum de Nesterov.

        Raises:
            ValueError: si algún hiperparámetro está fuera de rango.
        """

        if lr <= 0:
            raise ValueError("lr debe ser positivo.")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum debe estar en [0, 1).")
        if weight_decay < 0:
            raise ValueError("weight_decay no puede ser negativo.")
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        """
        Realiza un paso de optimización.

        Args:
            closure: función que recalcula la pérdida (opcional).

        Returns:
            La pérdida devuelta por ``closure`` o ``None``.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # TODO: recorre self.param_groups y guarda el buffer en self.state[p]
        raise NotImplementedError("TODO")
        return loss


class AdamFromScratch(Optimizer):
    """
    Adam / AdamW desde cero.

    ``m ← β1 m + (1 - β1) g`` ; ``v ← β2 v + (1 - β2) g²`` ;
    ``m̂ = m / (1 - β1^t)`` ; ``v̂ = v / (1 - β2^t)`` ;
    ``θ ← θ - η m̂ / (√v̂ + ε)``.

    Con ``decoupled_weight_decay=False`` (Adam) el weight decay se suma al
    gradiente (``g ← g + λ θ``, regularización L2). Con ``True`` (AdamW) se
    aplica directamente sobre los pesos antes del paso: ``θ ← θ (1 - η λ)``.
    """

    def __init__(
        self,
        params: Iterable[torch.Tensor] | Iterable[dict],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        decoupled_weight_decay: bool = False,
    ) -> None:
        """
        Constructor.

        Args:
            params: parámetros o grupos de parámetros.
            lr: learning rate ``η``.
            betas: ``(β1, β2)`` de las medias móviles del gradiente y de su cuadrado.
            eps: término ``ε`` del denominador.
            weight_decay: coeficiente ``λ``.
            decoupled_weight_decay: ``True`` para AdamW.

        Raises:
            ValueError: si algún hiperparámetro está fuera de rango.
        """

        if lr <= 0:
            raise ValueError("lr debe ser positivo.")
        if not all(0.0 <= b < 1.0 for b in betas):
            raise ValueError("Las betas deben estar en [0, 1).")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
            "decoupled_weight_decay": decoupled_weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        """
        Realiza un paso de optimización.

        Args:
            closure: función que recalcula la pérdida (opcional).

        Returns:
            La pérdida devuelta por ``closure`` o ``None``.
        """

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # TODO: guarda "step", "exp_avg" y "exp_avg_sq" en self.state[p]
        raise NotImplementedError("TODO")
        return loss


def make_param_groups(model: nn.Module, weight_decay: float) -> list[dict]:
    """
    Separa los parámetros en dos grupos: con y sin weight decay.

    No llevan weight decay los ``bias`` ni ningún parámetro de capas de
    normalización (BatchNorm, GroupNorm, LayerNorm, InstanceNorm), ni en general
    los tensores de 1 dimensión. Los parámetros congelados
    (``requires_grad=False``) se omiten.

    Args:
        model: modelo.
        weight_decay: coeficiente del grupo con decay.

    Returns:
        ``[{"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0}]``.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def make_finetuning_param_groups(
    model: nn.Module, head_name: str, backbone_lr: float, head_lr: float
) -> list[dict]:
    """
    Grupos con learning rate distinto para el backbone y la cabeza.

    Típico en *fine-tuning*: el backbone preentrenado con lr pequeño (p.ej. 1e-4)
    y la cabeza nueva con lr grande (p.ej. 1e-2).

    Args:
        model: modelo completo.
        head_name: nombre del submódulo cabeza (p.ej. ``"fc"`` en ResNet).
        backbone_lr: lr del resto de parámetros.
        head_lr: lr de la cabeza.

    Returns:
        ``[{"params": backbone, "lr": backbone_lr}, {"params": head, "lr": head_lr}]``.

    Raises:
        ValueError: si no hay ningún parámetro bajo ``head_name``.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def warmup_cosine_lambda(
    warmup_steps: int, total_steps: int, min_factor: float = 0.0
) -> Callable[[int], float]:
    """
    Factor multiplicativo para ``LambdaLR``: warmup lineal + decaimiento coseno.

    * ``step < warmup_steps``: ``(step + 1) / warmup_steps``.
    * después: ``min_factor + (1 - min_factor) · 0.5 · (1 + cos(π · t))`` con
      ``t = (step - warmup_steps) / (total_steps - warmup_steps)`` recortado a
      ``[0, 1]``.

    Args:
        warmup_steps: pasos de warmup (``≥ 0``).
        total_steps: pasos totales (``> warmup_steps``).
        min_factor: factor final (lr mínimo = ``min_factor · lr``).

    Returns:
        Función ``step -> factor``.

    Raises:
        ValueError: si ``total_steps <= warmup_steps`` o ``warmup_steps < 0``.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def run_schedule(
    optimizer: Optimizer, scheduler: LRScheduler, num_steps: int
) -> list[float]:
    """
    Simula ``num_steps`` iteraciones y devuelve el lr usado en cada una.

    En cada iteración: se lee el lr del primer grupo, se llama a
    ``optimizer.step()`` (con gradientes nulos, solo para respetar el orden) y
    después a ``scheduler.step()``. Este orden es el correcto desde PyTorch 1.1.

    Args:
        optimizer: optimizador ya construido.
        scheduler: scheduler asociado a ``optimizer``.
        num_steps: número de iteraciones.

    Returns:
        Lista de ``num_steps`` learning rates.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


class ExponentialMovingAverage:
    """
    Media móvil exponencial (EMA) de los pesos de un modelo.

    ``θ_ema ← d · θ_ema + (1 - d) · θ`` tras cada paso del optimizador. Los
    buffers no flotantes (p.ej. ``num_batches_tracked``) se copian tal cual.
    Se usa en YOLOv5/v8, DINO, Mean Teacher, modelos de difusión...
    """

    def __init__(self, model: nn.Module, decay: float = 0.999) -> None:
        """
        Constructor: copia el estado inicial del modelo.

        Args:
            model: modelo a promediar.
            decay: factor ``d ∈ [0, 1]``.

        Raises:
            ValueError: si ``decay`` no está en ``[0, 1]``.
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        """
        Actualiza la copia EMA con los pesos actuales del modelo.

        Args:
            model: modelo (mismo tipo que el del constructor).
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def copy_to(self, model: nn.Module) -> None:
        """
        Carga los pesos EMA en ``model`` (para evaluar o guardar).

        Args:
            model: modelo destino.
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")


def minimize_with_lbfgs(
    fn: Callable[[torch.Tensor], torch.Tensor], x0: torch.Tensor, steps: int = 20
) -> torch.Tensor:
    """
    Minimiza ``fn(x)`` con ``torch.optim.LBFGS`` optimizando directamente ``x``.

    Es el mismo patrón que el *style transfer* de Gatys: se optimiza la imagen
    (no los pesos) y LBFGS necesita una ``closure`` que haga ``zero_grad``,
    calcule la pérdida, llame a ``backward`` y la devuelva.

    Args:
        fn: función escalar a minimizar.
        x0: punto inicial (no se modifica). Dimensions: [*].
        steps: número de llamadas a ``optimizer.step(closure)``.

    Returns:
        Punto final (sin gradiente). Dimensions: [*].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def train_linear_regression(
    x: torch.Tensor,
    y: torch.Tensor,
    optimizer_name: Literal["sgd", "adam", "adamw"] = "sgd",
    lr: float = 0.1,
    epochs: int = 200,
) -> tuple[nn.Linear, list[float]]:
    """
    Ajusta una regresión lineal con descenso de gradiente a lote completo.

    Bucle canónico: ``optimizer.zero_grad()`` → ``loss = MSE(model(x), y)`` →
    ``loss.backward()`` → ``optimizer.step()``.

    Args:
        x: entradas. Dimensions: [N, in_features].
        y: salidas. Dimensions: [N, out_features].
        optimizer_name: ``"sgd"`` (momentum 0.9), ``"adam"`` o ``"adamw"``.
        lr: learning rate.
        epochs: número de pasos.

    Returns:
        El modelo ``nn.Linear`` entrenado y la pérdida de cada paso.

    Raises:
        ValueError: si ``optimizer_name`` no es válido.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def clone_model(model: nn.Module) -> nn.Module:
    """Copia profunda de un modelo (dada; útil para comparar optimizadores)."""

    return copy.deepcopy(model)
