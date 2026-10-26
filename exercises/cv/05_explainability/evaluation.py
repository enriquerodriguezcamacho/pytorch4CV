"""
CV05 · Explicabilidad: evaluación de explicaciones — ejercicios.

Curvas de *deletion* / *insertion* (loyalty e inverse loyalty), AUC, correlación
de rangos y randomización de parámetros para los *sanity checks* de Adebayo.
"""

import copy
import math
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn


def auc(scores: torch.Tensor) -> float:
    """
    Área bajo una curva muestreada en puntos equiespaciados de [0, 1] (regla del
    trapecio).

    Args:
        scores: valores de la curva. Dimensions: [n_points] (``n_points >= 2``).

    Returns:
        Área bajo la curva.
    """

    # TODO: usa torch.trapezoid con dx = 1 / (n_points - 1)
    raise NotImplementedError("TODO")


def _pixel_order(saliency: torch.Tensor, order: Literal["most", "least"]) -> torch.Tensor:
    """
    Orden de los píxeles (índices aplanados) según la saliencia. Ya dado.

    Args:
        saliency: mapa de importancia. Dimensions: [height, width].
        order: ``"most"`` (más importantes primero) o ``"least"``.

    Returns:
        Índices aplanados. Dimensions: [height * width].
    """

    if order not in ("most", "least"):
        raise ValueError("order debe ser 'most' o 'least'.")
    return saliency.flatten().argsort(descending=order == "most")


@torch.no_grad()
def perturbation_curve(
    model: nn.Module,
    x: torch.Tensor,
    saliency: torch.Tensor,
    target_class: int,
    mode: Literal["deletion", "insertion"] = "deletion",
    order: Literal["most", "least"] = "most",
    n_steps: int = 10,
    baseline: float = 0.0,
    use_softmax: bool = True,
) -> torch.Tensor:
    """
    Curva de *deletion* o *insertion* (Petsiuk et al., 2018).

    * ``deletion``: se parte de ``x`` y en cada paso se ponen a ``baseline`` (en
      todos los canales) los siguientes ``ceil(H·W / n_steps)`` píxeles según
      ``order``. Con ``order="most"`` es la curva de **loyalty**; con
      ``"least"`` la de **inverse loyalty**.
    * ``insertion``: se parte de una imagen llena de ``baseline`` y se van
      copiando los píxeles de ``x`` en el mismo orden.

    Todas las imágenes (``n_steps + 1``) se evalúan en un único *forward*.

    Args:
        model: modelo a evaluar.
        x: imagen. Dimensions: [channels, height, width].
        saliency: explicación a evaluar. Dimensions: [height, width].
        target_class: clase explicada.
        mode: ``"deletion"`` o ``"insertion"``.
        order: ``"most"`` o ``"least"`` importantes primero.
        n_steps: número de pasos de la curva.
        baseline: valor de "píxel eliminado".
        use_softmax: si ``True`` se usa la probabilidad; si no, el logit.

    Returns:
        Score de la clase tras 0, 1, ..., ``n_steps`` pasos. Dimensions:
        [n_steps + 1].

    Raises:
        ValueError: si ``mode`` u ``order`` no son válidos.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def deletion_score(
    model: nn.Module,
    x: torch.Tensor,
    saliency: torch.Tensor,
    target_class: int,
    n_steps: int = 10,
    use_softmax: bool = True,
) -> float:
    """
    AUC de la curva de *deletion* (más importantes primero). Cuanto **menor**,
    mejor la explicación. Ya dado a partir de :func:`perturbation_curve`.
    """

    curve = perturbation_curve(
        model, x, saliency, target_class, "deletion", "most", n_steps, 0.0, use_softmax
    )
    return auc(curve)


def insertion_score(
    model: nn.Module,
    x: torch.Tensor,
    saliency: torch.Tensor,
    target_class: int,
    n_steps: int = 10,
    use_softmax: bool = True,
) -> float:
    """
    AUC de la curva de *insertion* (más importantes primero). Cuanto **mayor**,
    mejor la explicación. Ya dado a partir de :func:`perturbation_curve`.
    """

    curve = perturbation_curve(
        model, x, saliency, target_class, "insertion", "most", n_steps, 0.0, use_softmax
    )
    return auc(curve)


def rank_correlation(map_a: torch.Tensor, map_b: torch.Tensor) -> float:
    """
    Correlación de Spearman entre dos mapas (se asume que no hay empates):
    correlación de Pearson entre los **rangos** de sus valores aplanados.

    Se usa en los *sanity checks*: si al randomizar los pesos el mapa apenas
    cambia (correlación alta), el método no depende del modelo.

    Args:
        map_a: primer mapa. Dimensions: cualquiera (p.ej. [height, width]).
        map_b: segundo mapa, mismo número de elementos.

    Returns:
        Correlación en [-1, 1].
    """

    # TODO: rango = argsort de argsort
    raise NotImplementedError("TODO")


def randomize_layers(model: nn.Module, layer_names: list[str]) -> nn.Module:
    """
    *Model parameter randomization test*: devuelve una **copia** del modelo con
    los parámetros de las capas indicadas reinicializados (``reset_parameters``).
    Para la versión en cascada se llama con nombres desde la salida hacia la
    entrada. El modelo original no se modifica.

    Args:
        model: modelo original.
        layer_names: nombres (de ``named_modules``) de las capas a reinicializar.

    Returns:
        Copia del modelo con esas capas aleatorizadas.

    Raises:
        KeyError: si algún nombre no existe en el modelo.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")
