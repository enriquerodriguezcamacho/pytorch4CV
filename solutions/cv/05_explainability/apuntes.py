"""
CV05 · Ejercicios de código del tema 5 de los apuntes (sección 5.7) — soluciones.

Solo incluye lo que no estaba ya en el tema: Shapley por Monte Carlo (Ej. 10),
loyalty según la definición de los apuntes (Ej. 22), ROAR (Ej. 23) y *sanity
checks* (Ej. 24). ``docs/EJERCICIOS_APUNTES.md`` indica dónde está el resto.
"""

from __future__ import annotations

import copy
from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import nn

from .evaluation import auc, perturbation_curve


def monte_carlo_shapley(
    value_fn: Callable[[frozenset[int]], float], n_players: int, n_permutations: int
) -> torch.Tensor:
    """
    Ej. 10 · Aproximación de Shapley con permutaciones aleatorias.

    Para cada permutación (``torch.randperm``) se añaden los jugadores en ese orden y
    se suma a cada uno su contribución marginal ``v(S ∪ {i}) - v(S)``. El resultado
    es la media entre permutaciones. Cuesta ``O(n_permutations · M)`` evaluaciones,
    frente a ``O(2^M)`` del cálculo exacto.

    Args:
        value_fn: función de valor, que recibe la coalición.
        n_players: número de jugadores ``M``.
        n_permutations: número de permutaciones.

    Returns:
        Valores de Shapley estimados (``float32``). Dimensions: [n_players].
    """

    # BEGIN SOLUTION
    totals = torch.zeros(n_players)
    for _ in range(n_permutations):
        coalition: frozenset[int] = frozenset()
        previous = value_fn(coalition)
        for player in torch.randperm(n_players).tolist():
            coalition = coalition | {player}
            current = value_fn(coalition)
            totals[player] += current - previous
            previous = current
    return totals / n_permutations
    # END SOLUTION


def shapley_pixel_map(
    model: nn.Module,
    x: torch.Tensor,
    target_class: int,
    grid_size: int,
    n_permutations: int = 20,
    baseline: float = 0.0,
) -> torch.Tensor:
    """
    Ej. 10 · Mapa de importancia por Shapley: cada jugador es una celda de una rejilla
    ``grid_size × grid_size``. Las celdas fuera de la coalición valen ``baseline`` y
    ``v(S)`` es la probabilidad de la clase objetivo.

    Args:
        model: clasificador. Salida: [batch, num_classes].
        x: imagen. Dimensions: [channels, height, width].
        target_class: clase explicada.
        grid_size: celdas por lado (debe dividir a ``height`` y a ``width``).
        n_permutations: permutaciones de Monte Carlo.
        baseline: valor de las celdas ausentes.

    Returns:
        Mapa con el valor de cada celda repetido en sus píxeles.
        Dimensions: [height, width].
    """

    # BEGIN SOLUTION
    _, height, width = x.shape
    cell_h, cell_w = height // grid_size, width // grid_size

    def value_fn(coalition: frozenset[int]) -> float:
        image = torch.full_like(x, baseline)
        for player in coalition:
            row, col = divmod(player, grid_size)
            top, left = row * cell_h, col * cell_w
            image[:, top : top + cell_h, left : left + cell_w] = x[
                :, top : top + cell_h, left : left + cell_w
            ]
        with torch.no_grad():
            probs = torch.softmax(model(image.unsqueeze(0)), dim=1)
        return probs[0, target_class].item()

    values = monte_carlo_shapley(value_fn, grid_size**2, n_permutations)
    grid = values.view(grid_size, grid_size)
    return grid.repeat_interleave(cell_h, dim=0).repeat_interleave(cell_w, dim=1)
    # END SOLUTION


def loyalty_scores(
    model: nn.Module,
    x: torch.Tensor,
    saliency: torch.Tensor,
    target_class: int,
    inverse: bool = False,
    n_steps: int = 10,
) -> tuple[torch.Tensor, float]:
    """
    Ej. 22 · Curva y AUC de *loyalty* (o *inverse loyalty*), como en los apuntes.

    Se eliminan píxeles de más a menos importantes (de menos a más con
    ``inverse=True``) con :func:`perturbation_curve` y, en cada paso, se registra

    ``1 - |f_c(x) - f_c(x')| / f_c(x)``,

    donde ``f_c`` es la probabilidad de la clase. El AUC se calcula con :func:`auc`.

    Args:
        model: clasificador.
        x: imagen. Dimensions: [channels, height, width].
        saliency: explicación. Dimensions: [height, width].
        target_class: clase explicada.
        inverse: si ``True``, calcula *inverse loyalty*.
        n_steps: pasos de la curva.

    Returns:
        Curva [n_steps + 1] y su AUC.
    """

    # BEGIN SOLUTION
    order = "least" if inverse else "most"
    curve = perturbation_curve(
        model, x, saliency, target_class, order=order, n_steps=n_steps
    )
    scores = 1 - (curve[0] - curve).abs() / curve[0]
    return scores, float(auc(scores))
    # END SOLUTION


def remove_most_important(
    images: torch.Tensor, saliencies: torch.Tensor, fraction: float, fill: float = 0.0
) -> torch.Tensor:
    """
    Ej. 23 · Paso de corrupción de ROAR: por cada imagen, pone a ``fill`` la fracción
    de píxeles más importantes (``round(fraction · H · W)``), en todos los canales.

    Args:
        images: imágenes. Dimensions: [batch, channels, height, width].
        saliencies: explicaciones. Dimensions: [batch, height, width].
        fraction: fracción en [0, 1].
        fill: valor neutro.

    Returns:
        Copia corrupta. Dimensions: [batch, channels, height, width].
    """

    # BEGIN SOLUTION
    batch, _, height, width = images.shape
    k = round(fraction * height * width)
    corrupted = images.clone()
    if k == 0:
        return corrupted
    top = saliencies.view(batch, -1).topk(k, dim=1).indices
    keep = torch.ones(batch, height * width, dtype=torch.bool)
    keep.scatter_(1, top, False)
    keep = keep.view(batch, 1, height, width).expand_as(images)
    corrupted[~keep] = fill
    return corrupted
    # END SOLUTION


def roar_accuracy(
    model_factory: Callable[[], nn.Module],
    train_data: tuple[torch.Tensor, torch.Tensor],
    test_data: tuple[torch.Tensor, torch.Tensor],
    epochs: int = 5,
    lr: float = 1e-2,
) -> float:
    """
    Ej. 23 · Reentrena **un modelo nuevo** con datos ya corruptos, a *batch* completo
    con Adam, y devuelve su *accuracy* en test.

    Se llama con los datos corruptos por la explicación y con los corruptos al azar:
    la explicación es útil si la caída es mayor en el primer caso.

    Args:
        model_factory: crea un modelo sin entrenar.
        train_data: ``(imágenes, etiquetas)`` de entrenamiento, ya corruptas.
        test_data: ``(imágenes, etiquetas)`` de test, ya corruptas.
        epochs: número de épocas.
        lr: *learning rate*.

    Returns:
        *Accuracy* en test.
    """

    # BEGIN SOLUTION
    model = model_factory()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    x_train, y_train = train_data
    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        F.cross_entropy(model(x_train), y_train).backward()
        optimizer.step()
    model.eval()
    x_test, y_test = test_data
    with torch.no_grad():
        return (model(x_test).argmax(dim=1) == y_test).float().mean().item()
    # END SOLUTION


def cascading_randomization(
    model: nn.Module, layer_names: list[str]
) -> list[nn.Module]:
    """
    Ej. 24 · Test de aleatorización **en cascada**: reinicializa las capas de forma
    acumulativa, desde la salida hacia la entrada.

    Args:
        model: modelo original (no se modifica).
        layer_names: nombres de las capas, ordenados de la salida hacia la entrada.

    Returns:
        Una copia por capa; la copia ``k`` tiene aleatorizadas las ``k + 1`` primeras
        capas de ``layer_names``.
    """

    # BEGIN SOLUTION
    copies = []
    current = copy.deepcopy(model)
    modules = dict(current.named_modules())
    for name in layer_names:
        modules[name].reset_parameters()
        copies.append(copy.deepcopy(current))
    return copies
    # END SOLUTION


def shuffle_labels(labels: torch.Tensor) -> torch.Tensor:
    """
    Ej. 24 · *Data randomization test*: permuta las etiquetas de entrenamiento.

    Args:
        labels: etiquetas. Dimensions: [N].

    Returns:
        Etiquetas permutadas con ``torch.randperm``. Dimensions: [N].
    """

    # BEGIN SOLUTION
    return labels[torch.randperm(labels.shape[0])]
    # END SOLUTION


def global_ssim(
    a: torch.Tensor, b: torch.Tensor, data_range: float = 1.0
) -> torch.Tensor:
    """
    Ej. 24 · SSIM global entre dos mapas (una sola ventana, la imagen entera):

    ``(2·μa·μb + C1)(2·σab + C2) / ((μa² + μb² + C1)(σa² + σb² + C2))``,

    con ``C1 = (0.01·L)²`` y ``C2 = (0.03·L)²``, y varianzas poblacionales.

    Args:
        a: primer mapa. Dimensions: [height, width].
        b: segundo mapa. Dimensions: [height, width].
        data_range: rango ``L`` de los valores.

    Returns:
        SSIM escalar; vale 1 si los mapas son idénticos.
    """

    # BEGIN SOLUTION
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    mean_a, mean_b = a.mean(), b.mean()
    var_a, var_b = a.var(unbiased=False), b.var(unbiased=False)
    cov = ((a - mean_a) * (b - mean_b)).mean()
    numerator = (2 * mean_a * mean_b + c1) * (2 * cov + c2)
    denominator = (mean_a**2 + mean_b**2 + c1) * (var_a + var_b + c2)
    return numerator / denominator
    # END SOLUTION
