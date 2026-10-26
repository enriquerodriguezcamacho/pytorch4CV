"""
F02 · Autograd — soluciones.

Ejercicios sobre el motor de diferenciación automática de PyTorch: derivadas
analíticas frente a autograd, gradientes respecto a la entrada (saliency, FGSM),
backprop manual, ``torch.autograd.Function`` personalizadas, hooks, segundas
derivadas, congelado de parámetros y acumulación de gradientes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable

import torch
import torch.nn.functional as F
from torch import nn

# ---------------------------------------------------------------------------
# 1. Derivada analítica frente a autograd
# ---------------------------------------------------------------------------


def f(x: torch.Tensor) -> torch.Tensor:
    """
    Función de ejemplo ``f(x) = x² · sin(x) + exp(-x)`` (ya dada).

    Args:
        x: tensor de entrada. Dimensions: [*].

    Returns:
        ``f(x)`` elemento a elemento. Dimensions: [*].
    """

    return x**2 * torch.sin(x) + torch.exp(-x)


def df_analytic(x: torch.Tensor) -> torch.Tensor:
    """
    Derivada analítica de ``f``: ``f'(x) = 2x·sin(x) + x²·cos(x) - exp(-x)``.

    Args:
        x: tensor de entrada. Dimensions: [*].

    Returns:
        ``f'(x)`` elemento a elemento, sin usar autograd. Dimensions: [*].
    """

    # BEGIN SOLUTION
    return 2 * x * torch.sin(x) + x**2 * torch.cos(x) - torch.exp(-x)
    # END SOLUTION


def df_autograd(
    fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor
) -> torch.Tensor:
    """
    Derivada elemento a elemento de ``fn`` calculada con autograd.

    No debe modificar ``x`` (ni su ``requires_grad`` ni su ``.grad``): trabaja
    sobre una copia desacoplada del grafo. Como ``fn`` actúa elemento a elemento,
    el gradiente de ``fn(x).sum()`` coincide con la derivada en cada punto.

    Args:
        fn: función elemento a elemento.
        x: puntos donde se evalúa la derivada. Dimensions: [*].

    Returns:
        ``fn'(x)`` sin grafo asociado (``requires_grad=False``). Dimensions: [*].
    """

    # BEGIN SOLUTION
    x = x.detach().clone().requires_grad_(True)
    y = fn(x)
    y.backward(torch.ones_like(y))
    return x.grad.detach()
    # END SOLUTION


# ---------------------------------------------------------------------------
# 2. Gradientes respecto a la entrada (base de saliency y ataques adversarios)
# ---------------------------------------------------------------------------


def input_gradient(
    model: nn.Module, images: torch.Tensor, labels: torch.Tensor
) -> torch.Tensor:
    """
    Gradiente de la cross-entropy respecto a las imágenes de entrada.

    El modelo se pone en modo ``eval`` y **no** deben quedar gradientes
    acumulados en sus parámetros (usa ``torch.autograd.grad`` con
    ``inputs=`` la imagen, o limpia los ``.grad``). Las imágenes originales no
    se modifican.

    Args:
        model: clasificador que devuelve logits.
        images: lote de imágenes. Dimensions: [batch, channels, height, width].
        labels: clases verdaderas. Dimensions: [batch].

    Returns:
        ``∂L/∂images`` (``L`` = cross-entropy media). Dimensions:
        [batch, channels, height, width].
    """

    # BEGIN SOLUTION
    model.eval()
    x = images.detach().clone().requires_grad_(True)
    loss = F.cross_entropy(model(x), labels)
    (grad,) = torch.autograd.grad(loss, x)
    return grad
    # END SOLUTION


def saliency_map(
    model: nn.Module, images: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """
    Mapa de saliencia vanilla (Simonyan et al., 2013).

    ``S = max_c |∂ y_target / ∂ x_c|``: gradiente del *logit* de la clase objetivo
    respecto a la entrada, en valor absoluto y con máximo sobre los canales.
    El modelo se pone en ``eval``.

    Args:
        model: clasificador que devuelve logits. Dimensions de salida: [batch, classes].
        images: lote de imágenes. Dimensions: [batch, channels, height, width].
        target: clase a explicar de cada imagen. Dimensions: [batch].

    Returns:
        Mapa de saliencia no negativo. Dimensions: [batch, height, width].
    """

    # BEGIN SOLUTION
    model.eval()
    x = images.detach().clone().requires_grad_(True)
    logits = model(x)
    score = logits.gather(1, target.view(-1, 1)).sum()
    (grad,) = torch.autograd.grad(score, x)
    return grad.abs().amax(dim=1)
    # END SOLUTION


def fgsm(
    model: nn.Module, images: torch.Tensor, labels: torch.Tensor, epsilon: float
) -> torch.Tensor:
    """
    Fast Gradient Sign Method (Goodfellow et al., 2015).

    ``x_adv = clip(x + ε · sign(∇_x L(f(x), y)), 0, 1)``.

    Args:
        model: clasificador que devuelve logits.
        images: imágenes en [0, 1]. Dimensions: [batch, channels, height, width].
        labels: clases verdaderas. Dimensions: [batch].
        epsilon: tamaño de la perturbación (norma L∞).

    Returns:
        Imágenes adversarias sin grafo, en [0, 1]. Dimensions:
        [batch, channels, height, width].
    """

    # BEGIN SOLUTION
    grad = input_gradient(model, images, labels)
    return (images + epsilon * grad.sign()).clamp(0.0, 1.0).detach()
    # END SOLUTION


# ---------------------------------------------------------------------------
# 3. Backprop a mano: Linear + ReLU + MSE
# ---------------------------------------------------------------------------


def manual_linear_relu_mse(
    x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Forward y backward a mano de ``L = mean((relu(x Wᵀ + b) - y)²)``.

    Sin usar autograd (ni ``.backward`` ni ``torch.autograd.grad``): aplica la
    regla de la cadena capa a capa.

    Args:
        x: entrada. Dimensions: [batch, in_features].
        weight: pesos ``W``. Dimensions: [out_features, in_features].
        bias: sesgo ``b``. Dimensions: [out_features].
        target: objetivo ``y``. Dimensions: [batch, out_features].

    Returns:
        Tupla ``(loss, dL/dW, dL/db, dL/dx)``. Dimensions: [], [out_features,
        in_features], [out_features] y [batch, in_features].
    """

    # BEGIN SOLUTION
    z = x @ weight.T + bias  # (B, O)
    a = z.clamp_min(0.0)
    diff = a - target
    loss = (diff**2).mean()
    grad_a = 2.0 * diff / diff.numel()  # dL/da
    grad_z = grad_a * (z > 0).to(z.dtype)  # ReLU'
    grad_w = grad_z.T @ x  # (O, I)
    grad_b = grad_z.sum(dim=0)
    grad_x = grad_z @ weight  # (B, I)
    return loss, grad_w, grad_b, grad_x
    # END SOLUTION


# ---------------------------------------------------------------------------
# 4. torch.autograd.Function personalizadas
# ---------------------------------------------------------------------------


class MySigmoid(torch.autograd.Function):
    """
    Sigmoide con backward propio: ``σ'(x) = σ(x)(1 - σ(x))``.

    Guarda la **salida** en ``ctx`` (es lo que necesita la derivada).
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, x: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            ctx: contexto para guardar tensores.
            x: entrada. Dimensions: [*].

        Returns:
            ``σ(x)``. Dimensions: [*].
        """

        # BEGIN SOLUTION
        out = torch.sigmoid(x)
        ctx.save_for_backward(out)
        return out
        # END SOLUTION

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            ctx: contexto con la salida guardada.
            grad_output: ``∂L/∂σ``. Dimensions: [*].

        Returns:
            ``∂L/∂x``. Dimensions: [*].
        """

        # BEGIN SOLUTION
        (out,) = ctx.saved_tensors
        return grad_output * out * (1 - out)
        # END SOLUTION


class GuidedReLU(torch.autograd.Function):
    """
    ReLU para *Guided Backpropagation* (Springenberg et al., 2015).

    * forward: ``max(x, 0)``, igual que una ReLU.
    * backward: solo deja pasar el gradiente donde **la entrada era positiva y
      el gradiente que llega es positivo**: ``g_in = g_out · 1[x > 0] · 1[g_out > 0]``.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, x: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            ctx: contexto para guardar tensores.
            x: entrada. Dimensions: [*].

        Returns:
            ``max(x, 0)``. Dimensions: [*].
        """

        # BEGIN SOLUTION
        ctx.save_for_backward(x)
        return x.clamp_min(0.0)
        # END SOLUTION

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            ctx: contexto con la entrada guardada.
            grad_output: gradiente que llega desde arriba. Dimensions: [*].

        Returns:
            Gradiente guiado. Dimensions: [*].
        """

        # BEGIN SOLUTION
        (x,) = ctx.saved_tensors
        return grad_output.clamp_min(0.0) * (x > 0).to(grad_output.dtype)
        # END SOLUTION


class GradientReversal(torch.autograd.Function):
    """
    Capa de inversión de gradiente (Ganin & Lempitsky, 2015, DANN).

    * forward: identidad.
    * backward: multiplica el gradiente por ``-lambd``. ``lambd`` no es un
      tensor, así que su "gradiente" es ``None``.
    """

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx, x: torch.Tensor, lambd: float
    ) -> torch.Tensor:
        """
        Args:
            ctx: contexto donde guardar ``lambd``.
            x: entrada. Dimensions: [*].
            lambd: factor de inversión (≥ 0).

        Returns:
            ``x`` sin cambios (usa ``view_as`` para devolver un tensor nuevo).
            Dimensions: [*].
        """

        # BEGIN SOLUTION
        ctx.lambd = lambd
        return x.view_as(x)
        # END SOLUTION

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor
    ) -> tuple[torch.Tensor, None]:
        """
        Args:
            ctx: contexto con ``lambd``.
            grad_output: gradiente que llega. Dimensions: [*].

        Returns:
            ``(-lambd · grad_output, None)``: un gradiente por cada entrada de forward.
        """

        # BEGIN SOLUTION
        return -ctx.lambd * grad_output, None
        # END SOLUTION


def grad_reverse(x: torch.Tensor, lambd: float = 1.0) -> torch.Tensor:
    """Atajo funcional para ``GradientReversal`` (ya dado)."""

    return GradientReversal.apply(x, lambd)


# ---------------------------------------------------------------------------
# 5. Hooks sobre tensores
# ---------------------------------------------------------------------------


def register_gradient_scaler(
    tensor: torch.Tensor, factor: float
) -> torch.utils.hooks.RemovableHandle:
    """
    Registra un hook que multiplica por ``factor`` el gradiente de ``tensor``.

    Args:
        tensor: tensor con ``requires_grad=True`` (hoja o intermedio).
        factor: factor de escala.

    Returns:
        *Handle* del hook (``handle.remove()`` lo desactiva).
    """

    # BEGIN SOLUTION
    return tensor.register_hook(lambda grad: grad * factor)
    # END SOLUTION


# ---------------------------------------------------------------------------
# 6. Segundo orden: create_graph
# ---------------------------------------------------------------------------


def second_derivative(
    fn: Callable[[torch.Tensor], torch.Tensor], x: torch.Tensor
) -> torch.Tensor:
    """
    Segunda derivada elemento a elemento de ``fn`` con ``create_graph=True``.

    Args:
        fn: función elemento a elemento.
        x: puntos de evaluación. Dimensions: [*].

    Returns:
        ``fn''(x)`` sin grafo asociado. Dimensions: [*].
    """

    # BEGIN SOLUTION
    x = x.detach().clone().requires_grad_(True)
    y = fn(x)
    (dy,) = torch.autograd.grad(y.sum(), x, create_graph=True)
    (d2y,) = torch.autograd.grad(dy.sum(), x)
    return d2y.detach()
    # END SOLUTION


def gradient_penalty(
    critic: nn.Module, real: torch.Tensor, fake: torch.Tensor, alpha: torch.Tensor
) -> torch.Tensor:
    """
    Penalización de gradiente de WGAN-GP (Gulrajani et al., 2017).

    ``x̂ = α·real + (1-α)·fake`` y ``GP = mean_i (‖∇_x̂ D(x̂)_i‖₂ - 1)²``.
    El resultado debe ser diferenciable respecto a los parámetros del crítico
    (usa ``create_graph=True``).

    Args:
        critic: red que devuelve una puntuación por muestra. Dimensions de
            salida: [batch, 1] o [batch].
        real: muestras reales. Dimensions: [batch, channels, height, width].
        fake: muestras generadas. Dimensions: [batch, channels, height, width].
        alpha: pesos de interpolación. Dimensions: [batch, 1, 1, 1].

    Returns:
        Escalar con la penalización. Dimensions: [].
    """

    # BEGIN SOLUTION
    x_hat = (alpha * real + (1 - alpha) * fake.detach()).requires_grad_(True)
    scores = critic(x_hat)
    (grad,) = torch.autograd.grad(scores.sum(), x_hat, create_graph=True)
    norms = grad.flatten(1).norm(2, dim=1)
    return ((norms - 1) ** 2).mean()
    # END SOLUTION


# ---------------------------------------------------------------------------
# 7. Congelar parámetros y acumulación de gradientes
# ---------------------------------------------------------------------------


def freeze(module: nn.Module) -> nn.Module:
    """
    Congela todos los parámetros de ``module`` (``requires_grad=False``).

    Args:
        module: módulo a congelar.

    Returns:
        El mismo módulo (para encadenar).
    """

    # BEGIN SOLUTION
    for p in module.parameters():
        p.requires_grad_(False)
    return module
    # END SOLUTION


def train_with_accumulation(
    model: nn.Module,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    optimizer: torch.optim.Optimizer,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    accumulation_steps: int,
) -> list[float]:
    """
    Entrena una época acumulando gradientes durante ``accumulation_steps`` micro-lotes.

    Para cada micro-lote: ``loss / accumulation_steps`` → ``backward``. Cada
    ``accumulation_steps`` micro-lotes (y al final si sobran) se hace
    ``optimizer.step()`` y ``optimizer.zero_grad()``. Con pérdidas *mean* y
    micro-lotes del mismo tamaño equivale a entrenar con un lote
    ``accumulation_steps`` veces mayor.

    Args:
        model: modelo a entrenar (se pone en ``train``).
        loss_fn: pérdida ``loss_fn(pred, target)`` con reducción media.
        optimizer: optimizador sobre los parámetros de ``model``.
        batches: iterable de ``(inputs, targets)``.
        accumulation_steps: número de micro-lotes por paso del optimizador (≥ 1).

    Returns:
        Lista con la pérdida (sin escalar) de cada micro-lote.

    Raises:
        ValueError: si ``accumulation_steps < 1``.
    """

    # BEGIN SOLUTION
    if accumulation_steps < 1:
        raise ValueError("accumulation_steps debe ser >= 1")
    model.train()
    optimizer.zero_grad()
    losses: list[float] = []
    pending = 0
    for inputs, targets in batches:
        loss = loss_fn(model(inputs), targets)
        (loss / accumulation_steps).backward()
        losses.append(loss.item())
        pending += 1
        if pending == accumulation_steps:
            optimizer.step()
            optimizer.zero_grad()
            pending = 0
    if pending:
        optimizer.step()
        optimizer.zero_grad()
    return losses
    # END SOLUTION
