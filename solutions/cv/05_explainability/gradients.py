"""
CV05 · Explicabilidad: métodos basados en gradientes — soluciones.

Saliency (Simonyan), Input × Gradient, SmoothGrad, Integrated Gradients,
Guided Backpropagation y DeconvNet.
"""

from typing import Literal

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

from .utils import normalize_map, select_target, target_scores


def input_gradient(
    model: nn.Module, x: torch.Tensor, target_class: int | None = None
) -> torch.Tensor:
    """
    Gradiente del *score* (logit, sin softmax) de la clase objetivo respecto a la
    entrada: $\\partial f_c / \\partial x$.

    Pone el modelo en modo ``eval`` y **no** acumula gradientes en los parámetros
    (usa ``torch.autograd.grad``). Como las muestras del lote son independientes,
    basta con derivar la suma de los scores de todo el lote.

    Args:
        model: modelo a explicar.
        x: imágenes de entrada. Dimensions: [batch, channels, height, width].
        target_class: clase a explicar; ``None`` usa la clase predicha.

    Returns:
        Gradiente respecto a la entrada. Dimensions: [batch, channels, height, width].
    """

    # BEGIN SOLUTION
    model.eval()
    x = x.detach().clone().requires_grad_(True)
    logits = model(x)
    targets = select_target(logits, target_class)
    scores = target_scores(logits, targets)
    (grad,) = torch.autograd.grad(scores.sum(), x)
    return grad
    # END SOLUTION


def saliency_map(
    model: nn.Module, x: torch.Tensor, target_class: int | None = None
) -> torch.Tensor:
    """
    Saliency map de Simonyan et al.: $M_{ij} = \\max_k |\\partial f_c/\\partial x_{ijk}|$.

    Args:
        model: modelo a explicar.
        x: imágenes de entrada. Dimensions: [batch, channels, height, width].
        target_class: clase a explicar; ``None`` usa la clase predicha.

    Returns:
        Mapa de saliencia (sin normalizar, no negativo). Dimensions:
        [batch, height, width].
    """

    # BEGIN SOLUTION
    grad = input_gradient(model, x, target_class)
    return grad.abs().amax(dim=1)
    # END SOLUTION


def input_x_gradient(
    model: nn.Module, x: torch.Tensor, target_class: int | None = None
) -> torch.Tensor:
    """
    Input × Gradient: $M_{ij} = \\max_k |x_{ijk} \\cdot \\partial f_c/\\partial x_{ijk}|$.

    Args:
        model: modelo a explicar.
        x: imágenes de entrada. Dimensions: [batch, channels, height, width].
        target_class: clase a explicar; ``None`` usa la clase predicha.

    Returns:
        Mapa de atribución (sin normalizar). Dimensions: [batch, height, width].
    """

    # BEGIN SOLUTION
    grad = input_gradient(model, x, target_class)
    return (x.detach() * grad).abs().amax(dim=1)
    # END SOLUTION


def smoothgrad(
    model: nn.Module,
    x: torch.Tensor,
    target_class: int | None = None,
    n_samples: int = 25,
    noise_level: float = 0.15,
) -> torch.Tensor:
    """
    SmoothGrad: media de ``n_samples`` saliency maps calculados sobre
    $x + \\varepsilon_i$, con $\\varepsilon_i \\sim \\mathcal{N}(0, \\sigma^2)$ y
    $\\sigma$ = ``noise_level`` · (max(x) − min(x)) **de cada muestra**.

    La clase objetivo (si es ``None``) se fija con la predicción sobre la imagen
    limpia, para que todas las muestras ruidosas expliquen la misma clase.

    Args:
        model: modelo a explicar.
        x: imágenes de entrada. Dimensions: [batch, channels, height, width].
        target_class: clase a explicar; ``None`` usa la clase predicha sobre ``x``.
        n_samples: número de copias ruidosas.
        noise_level: fracción del rango de la imagen usada como desviación típica.

    Returns:
        Mapa promedio (sin normalizar). Dimensions: [batch, height, width].
    """

    # BEGIN SOLUTION
    model.eval()
    with torch.no_grad():
        targets = select_target(model(x), target_class)
    value_range = x.amax(dim=(1, 2, 3), keepdim=True) - x.amin(
        dim=(1, 2, 3), keepdim=True
    )
    sigma = noise_level * value_range
    total = torch.zeros(x.shape[0], x.shape[2], x.shape[3], device=x.device)
    for _ in range(n_samples):
        noisy = (x + sigma * torch.randn_like(x)).detach().requires_grad_(True)
        scores = target_scores(model(noisy), targets)
        (grad,) = torch.autograd.grad(scores.sum(), noisy)
        total += grad.abs().amax(dim=1)
    return total / n_samples
    # END SOLUTION


def integrated_gradients(
    model: nn.Module,
    x: torch.Tensor,
    target_class: int | None = None,
    baseline: Literal["zero", "random"] = "zero",
    n_steps: int = 50,
) -> torch.Tensor:
    """
    Integrated Gradients con aproximación de Riemann por la derecha:

    $IG(x) = (x - x') \\cdot \\frac{1}{M}\\sum_{k=1}^{M}
    \\nabla_x f_c\\big(x' + \\tfrac{k}{M}(x - x')\\big)$.

    Cumple (aproximadamente) el axioma de **completitud**:
    $\\sum IG(x) \\approx f_c(x) - f_c(x')$.

    Args:
        model: modelo a explicar.
        x: imágenes de entrada. Dimensions: [batch, channels, height, width].
        target_class: clase a explicar; ``None`` usa la clase predicha sobre ``x``.
        baseline: ``"zero"`` (imagen negra) o ``"random"`` (ruido uniforme en [0, 1)).
        n_steps: número de pasos $M$ de la aproximación.

    Returns:
        Atribuciones con signo. Dimensions: [batch, channels, height, width].

    Raises:
        ValueError: si ``baseline`` no es ``"zero"`` ni ``"random"``.
    """

    # BEGIN SOLUTION
    if baseline == "zero":
        x_base = torch.zeros_like(x)
    elif baseline == "random":
        x_base = torch.rand_like(x)
    else:
        raise ValueError(f"Baseline desconocido: {baseline!r}.")

    model.eval()
    with torch.no_grad():
        targets = select_target(model(x), target_class)

    total_grad = torch.zeros_like(x)
    for k in range(1, n_steps + 1):
        point = (x_base + (k / n_steps) * (x - x_base)).detach().requires_grad_(True)
        scores = target_scores(model(point), targets)
        (grad,) = torch.autograd.grad(scores.sum(), point)
        total_grad += grad
    return (x - x_base).detach() * total_grad / n_steps
    # END SOLUTION


class GuidedBackprop:
    """
    Guided Backpropagation (Springenberg et al., 2015).

    En el *backward* de cada ``nn.ReLU`` solo deja pasar el gradiente si la
    activación del *forward* era positiva **y** el gradiente que llega es
    positivo. IMPORTANTE: las ReLU deben ser módulos ``nn.ReLU`` (no ``F.relu``),
    si no, los *hooks* no se pueden registrar.
    """

    def __init__(self, model: nn.Module) -> None:
        """
        Constructor de la clase.

        Args:
            model: modelo a explicar.
        """

        self.model = model
        self.handles: list[RemovableHandle] = []
        self.activations: list[torch.Tensor] = []

    def _forward_hook(
        self, module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        """
        *Forward hook*: guarda la activación de la ReLU (útil para visualizarla).
        Ya dado.
        """

        self.activations.append(output.detach())

    def _backward_hook(
        self,
        module: nn.Module,
        grad_input: tuple[torch.Tensor, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor]:
        """
        *Backward hook* de la ReLU con la regla de Guided Backprop.

        ``grad_input[0]`` ya es ``grad_output[0] · 1(activación > 0)``; solo falta
        anular la parte negativa del gradiente.

        Args:
            module: la ReLU.
            grad_input: gradiente respecto a la entrada de la ReLU.
            grad_output: gradiente respecto a la salida de la ReLU.

        Returns:
            Tupla con el nuevo gradiente respecto a la entrada.
        """

        # BEGIN SOLUTION
        return (torch.clamp(grad_input[0], min=0.0),)
        # END SOLUTION

    def _register_hooks(self) -> None:
        """
        Registra un *forward hook* y un *full backward hook* en cada ``nn.ReLU``
        del modelo y guarda los *handles* en ``self.handles``.

        Raises:
            RuntimeError: si el modelo no contiene ningún ``nn.ReLU``.
        """

        # BEGIN SOLUTION
        for module in self.model.modules():
            if isinstance(module, nn.ReLU):
                self.handles.append(module.register_forward_hook(self._forward_hook))
                self.handles.append(
                    module.register_full_backward_hook(self._backward_hook)
                )
        if not self.handles:
            raise RuntimeError("El modelo no contiene módulos nn.ReLU.")
        # END SOLUTION

    def _remove_hooks(self) -> None:
        """Elimina todos los *hooks* y vacía la lista de activaciones."""

        # BEGIN SOLUTION
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.activations.clear()
        # END SOLUTION

    def attribute(
        self, x: torch.Tensor, target_class: int | None = None
    ) -> torch.Tensor:
        """
        Gradiente "guiado" respecto a la entrada. Los *hooks* se eliminan siempre al
        terminar (también si hay una excepción).

        Args:
            x: imágenes de entrada. Dimensions: [batch, channels, height, width].
            target_class: clase a explicar; ``None`` usa la clase predicha.

        Returns:
            Gradiente modificado. Dimensions: [batch, channels, height, width].

        Raises:
            RuntimeError: si el modelo no contiene ningún ``nn.ReLU``.
        """

        # BEGIN SOLUTION
        self._register_hooks()
        try:
            return input_gradient(self.model, x, target_class)
        finally:
            self._remove_hooks()
        # END SOLUTION

    def explain(self, x: torch.Tensor, target_class: int | None = None) -> torch.Tensor:
        """
        Mapa 2D normalizado: máximo del valor absoluto sobre los canales. Ya dado.

        Args:
            x: imágenes de entrada. Dimensions: [batch, channels, height, width].
            target_class: clase a explicar; ``None`` usa la clase predicha.

        Returns:
            Explicación en [0, 1]. Dimensions: [batch, height, width].
        """

        return normalize_map(self.attribute(x, target_class).abs().amax(dim=1))


class DeconvNet(GuidedBackprop):
    """
    DeconvNet (Zeiler & Fergus): en la ReLU del *backward* se aplica una ReLU al
    gradiente que llega e **ignora** la máscara del *forward*.
    """

    def _backward_hook(
        self,
        module: nn.Module,
        grad_input: tuple[torch.Tensor, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> tuple[torch.Tensor]:
        """
        *Backward hook* de la ReLU con la regla de DeconvNet.

        Args:
            module: la ReLU.
            grad_input: gradiente respecto a la entrada de la ReLU.
            grad_output: gradiente respecto a la salida de la ReLU.

        Returns:
            Tupla con el nuevo gradiente respecto a la entrada.
        """

        # BEGIN SOLUTION
        return (torch.clamp(grad_output[0], min=0.0),)
        # END SOLUTION


def deconvnet(
    model: nn.Module, x: torch.Tensor, target_class: int | None = None
) -> torch.Tensor:
    """
    Atajo funcional de :class:`DeconvNet`. Ya dado.

    Args:
        model: modelo a explicar.
        x: imágenes de entrada. Dimensions: [batch, channels, height, width].
        target_class: clase a explicar; ``None`` usa la clase predicha.

    Returns:
        Reconstrucción DeconvNet. Dimensions: [batch, channels, height, width].
    """

    return DeconvNet(model).attribute(x, target_class)
