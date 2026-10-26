"""
F12 · Hooks y extracción de características — ejercicios.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.hooks import RemovableHandle


# ---------------------------------------------------------------------------
# Ejercicio 1 · CustomModel (estilo Final: style transfer con hooks en VGG)
# ---------------------------------------------------------------------------
class CustomModel(nn.Module):
    """
    Envuelve un modelo y captura con *forward hooks* la salida de cada
    ``nn.Conv2d`` (en el orden en que se ejecutan).

    Es la pieza del examen Final que se usaba para obtener las activaciones de
    VGG en *style transfer*.
    """

    def __init__(self, model: nn.Module) -> None:
        """
        Constructor (dado).

        Args:
            model: modelo cuyas convoluciones se quieren observar.
        """

        super().__init__()
        self.model = model
        self.activations: list[torch.Tensor] = []
        self.hooks: list[RemovableHandle] = []
        self._register_hooks()

    def _register_hooks(self) -> None:
        """
        Registra un *forward hook* en **cada** ``nn.Conv2d`` de ``self.model``.

        El hook debe añadir la salida de la capa a ``self.activations`` y el
        ``RemovableHandle`` devuelto por ``register_forward_hook`` debe guardarse en
        ``self.hooks``.
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward del modelo envuelto (dado).

        Args:
            x: entrada. Dimensions: [batch, channels, height, width].

        Returns:
            Salida del modelo.
        """

        return self.model(x)

    def obtain_activations(
        self, x: torch.Tensor, layers: Sequence[int], use_grad: bool
    ) -> dict[int, torch.Tensor]:
        """
        Ejecuta el modelo y devuelve las activaciones de las convoluciones pedidas.

        Hay que vaciar ``self.activations`` antes del forward (si no, se mezclan
        llamadas anteriores) y usar ``torch.set_grad_enabled(use_grad)``: en *style
        transfer* las activaciones de la imagen de estilo/contenido son
        constantes (``use_grad=False``), pero las de la imagen que se optimiza
        necesitan gradiente (``use_grad=True``).

        Args:
            x: entrada. Dimensions: [batch, channels, height, width].
            layers: índices (0, 1, 2...) de las convoluciones, en orden de
                ejecución.
            use_grad: si se construye el grafo de autograd.

        Returns:
            Diccionario ``{índice: activación}``; cada activación con Dimensions:
            [batch, out_channels, height', width'].
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def _remove_hooks_and_activations(self) -> None:
        """Quita todos los hooks y vacía las listas (dado)."""

        for hook in self.hooks:
            hook.remove()
        self.hooks.clear()
        self.activations.clear()


# ---------------------------------------------------------------------------
# Ejercicio 2 · closures: un hook por capa con su propio nombre
# ---------------------------------------------------------------------------
def register_activation_hooks(
    model: nn.Module, layer_names: Sequence[str]
) -> tuple[dict[str, torch.Tensor], list[RemovableHandle]]:
    """
    Registra un *forward hook* en cada submódulo de ``layer_names``.

    Cada hook guarda la salida (``detach()``) en un diccionario bajo **su**
    nombre. Cuidado con el *late binding* de las lambdas en un bucle: usa un
    argumento por defecto (``lambda m, i, o, k=name: ...``) o una función fábrica.

    Args:
        model: modelo.
        layer_names: nombres tal y como aparecen en ``model.named_modules()``
            (p.ej. ``"features.0"``).

    Returns:
        El diccionario (vacío; se rellena en cada forward) y la lista de
        ``RemovableHandle``.

    Raises:
        KeyError: si algún nombre no existe en el modelo.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def count_activation_shapes(
    model: nn.Module, x: torch.Tensor, layer_type: type[nn.Module] = nn.Conv2d
) -> dict[str, tuple[int, ...]]:
    """
    Forma de la salida de cada capa de tipo ``layer_type``.

    Registra hooks **temporales**, hace un forward sin gradientes y los quita
    siempre (usa ``try/finally``), de modo que el modelo queda como estaba.

    Args:
        model: modelo.
        x: entrada. Dimensions: [batch, channels, height, width].
        layer_type: clase de las capas a inspeccionar.

    Returns:
        Diccionario ``{nombre de la capa: tuple(shape de la salida)}`` en el orden
        de ``named_modules()``.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


# ---------------------------------------------------------------------------
# Ejercicio 3 · forward pre-hook que modifica la entrada
# ---------------------------------------------------------------------------
def add_input_normalization_hook(
    module: nn.Module, mean: Sequence[float], std: Sequence[float]
) -> RemovableHandle:
    """
    Añade un *forward pre-hook* que normaliza la entrada por canal.

    Un pre-hook recibe ``(module, args)`` y, si devuelve algo, sustituye a los
    argumentos del ``forward``. Debe devolver una tupla con
    ``(x - mean) / std`` (``mean`` y ``std`` con *broadcasting* sobre
    ``[batch, channels, height, width]``).

    Args:
        module: módulo al que se le añade el pre-hook.
        mean: media por canal.
        std: desviación típica por canal.

    Returns:
        ``RemovableHandle`` para quitar el hook.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


# ---------------------------------------------------------------------------
# Ejercicio 4 · forward + backward hooks: Grad-CAM
# ---------------------------------------------------------------------------
class GradientCapture:
    """
    Captura activaciones (*forward hook*) y gradientes respecto a la salida
    (*full backward hook*) de una capa objetivo. Base de Grad-CAM.
    """

    def __init__(self, target_layer: nn.Module) -> None:
        """
        Constructor: registra ambos hooks.

        Atributos: ``self.activations`` (salida de la capa, ``detach()``),
        ``self.gradients`` (``grad_output[0]``, ``detach()``) y ``self.handles``
        (lista con los dos ``RemovableHandle``).

        Args:
            target_layer: capa a observar (p.ej. ``model.layer4`` en ResNet).
        """

        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.handles: list[RemovableHandle] = []

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def _forward_hook(
        self, module: nn.Module, inputs: tuple[torch.Tensor, ...], output: torch.Tensor
    ) -> None:
        """
        Guarda la salida de la capa.

        Args:
            module: la capa.
            inputs: entradas del forward.
            output: salida. Dimensions: [batch, channels, height, width].
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def _backward_hook(
        self,
        module: nn.Module,
        grad_input: tuple[torch.Tensor | None, ...],
        grad_output: tuple[torch.Tensor | None, ...],
    ) -> None:
        """
        Guarda $\\partial y / \\partial A$ (el gradiente respecto a la **salida**).

        Args:
            module: la capa.
            grad_input: gradientes respecto a las entradas.
            grad_output: gradientes respecto a las salidas; ``grad_output[0]`` con
                Dimensions: [batch, channels, height, width].
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def remove(self) -> None:
        """Quita los hooks (dado)."""

        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def grad_cam(
    model: nn.Module,
    target_layer: nn.Module,
    x: torch.Tensor,
    class_idx: torch.Tensor | None = None,
) -> torch.Tensor:
    """
    Grad-CAM (Selvaraju et al., 2017).

    $\\alpha_k = \\frac{1}{HW}\\sum_{ij} \\partial y^c / \\partial A^k_{ij}$,
    $L = \\mathrm{ReLU}(\\sum_k \\alpha_k A^k)$; después se interpola
    (bilineal, ``align_corners=False``) al tamaño de ``x`` y se normaliza cada
    mapa a [0, 1] dividiendo por su máximo (``+ 1e-8``). Usa ``GradientCapture``
    y quita los hooks al terminar (``try/finally``). Si ``class_idx`` es ``None``,
    usa la clase predicha. Pon ``model.eval()`` y ``model.zero_grad()``.

    Args:
        model: clasificador. Devuelve Dimensions: [batch, num_classes].
        target_layer: capa convolucional objetivo.
        x: imágenes. Dimensions: [batch, channels, height, width].
        class_idx: clase objetivo por imagen. Dimensions: [batch].

    Returns:
        Mapas en [0, 1]. Dimensions: [batch, height, width].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


# ---------------------------------------------------------------------------
# Ejercicio 5 · guided backprop con hooks en las ReLU
# ---------------------------------------------------------------------------
class GuidedBackprop:
    """
    *Guided backpropagation* (Springenberg et al., 2015) con *backward hooks*.

    En cada ``nn.ReLU`` el gradiente que llega hacia atrás ya está enmascarado por
    la ReLU (solo pasa donde la entrada era > 0); además, guided backprop anula
    los gradientes negativos: ``grad_input = clamp(grad_input, min=0)``.
    """

    def __init__(self, model: nn.Module) -> None:
        """
        Constructor (dado).

        Args:
            model: modelo con ReLU **no in-place** (``inplace=False``).
        """

        self.model = model
        self.handles: list[RemovableHandle] = []

    def _relu_backward_hook(
        self,
        module: nn.Module,
        grad_input: tuple[torch.Tensor | None, ...],
        grad_output: tuple[torch.Tensor | None, ...],
    ) -> tuple[torch.Tensor | None, ...]:
        """
        Hook de la ReLU: devuelve el nuevo ``grad_input`` con los negativos a 0.

        Args:
            module: la ReLU.
            grad_input: gradiente respecto a la entrada de la ReLU (tupla de 1).
            grad_output: gradiente respecto a la salida.

        Returns:
            Tupla con ``grad_input[0].clamp(min=0)``.
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def _register_hooks(self) -> None:
        """Registra ``_relu_backward_hook`` en todas las ``nn.ReLU``."""

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def _remove_hooks(self) -> None:
        """Quita los hooks (dado)."""

        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def attribute(self, x: torch.Tensor, class_idx: torch.Tensor) -> torch.Tensor:
        """
        Gradiente guiado de la puntuación de ``class_idx`` respecto a la entrada.

        Registra los hooks, clona ``x`` con ``requires_grad_()``, hace forward y
        backward de la suma de las puntuaciones de la clase objetivo y quita los
        hooks siempre (``try/finally``).

        Args:
            x: imágenes. Dimensions: [batch, channels, height, width].
            class_idx: clase objetivo por imagen. Dimensions: [batch].

        Returns:
            Mapa de atribución. Dimensions: [batch, channels, height, width].
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")


# ---------------------------------------------------------------------------
# Ejercicio 6 · torchvision.models.feature_extraction
# ---------------------------------------------------------------------------
def extract_features_fx(
    model: nn.Module, return_nodes: dict[str, str], x: torch.Tensor
) -> dict[str, torch.Tensor]:
    """
    Extrae características intermedias con ``create_feature_extractor`` (FX).

    A diferencia de los hooks, reescribe el grafo del modelo y devuelve un
    diccionario sin estado global ni ``remove()``. Los nombres de nodo se
    consultan con ``get_graph_node_names(model)``.

    Args:
        model: modelo trazable con ``torch.fx``.
        return_nodes: ``{nombre del nodo: nombre de la clave de salida}``.
        x: entrada. Dimensions: [batch, channels, height, width].

    Returns:
        ``{nombre de salida: tensor}``.
    """

    from torchvision.models.feature_extraction import create_feature_extractor

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")
