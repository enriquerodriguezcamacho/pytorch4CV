"""
CV05 · Explicabilidad: visualización de features internas — soluciones.

Filtros de la primera capa, *maximally activating patches*, *gradient ascent*
(class model visualisation), *feature inversion* y DeepDream.
"""

from collections.abc import Callable

import torch
from torch import nn


def normalize_filters(conv: nn.Conv2d) -> torch.Tensor:
    """
    Prepara los filtros de una ``Conv2d`` para visualizarlos como imágenes:
    normalización min-max **de cada filtro** por separado.

    Args:
        conv: capa convolucional (típicamente la primera de la red).

    Returns:
        Filtros en [0, 1] (copia sin gradiente). Dimensions:
        [out_channels, in_channels, kernel_h, kernel_w].
    """

    # BEGIN SOLUTION
    w = conv.weight.detach().clone()
    min_ = w.amin(dim=(1, 2, 3), keepdim=True)
    max_ = w.amax(dim=(1, 2, 3), keepdim=True)
    return (w - min_) / (max_ - min_ + 1e-8)
    # END SOLUTION


def max_activating_indices(
    activations: torch.Tensor, channel: int, k: int
) -> torch.Tensor:
    """
    Índices de las ``k`` imágenes que más activan un canal (*maximally activating
    patches*): se usa el **máximo espacial** de la activación del canal.

    Args:
        activations: activaciones de una capa para N imágenes. Dimensions:
            [N, channels, h, w].
        channel: canal a analizar.
        k: número de imágenes a devolver.

    Returns:
        Índices ordenados de mayor a menor activación. Dimensions: [k].
    """

    # BEGIN SOLUTION
    scores = activations[:, channel].amax(dim=(1, 2))
    return scores.topk(k).indices
    # END SOLUTION


def gradient_ascent_class_visualization(
    model: nn.Module,
    target_class: int,
    input_shape: tuple[int, int, int, int] = (1, 3, 32, 32),
    n_steps: int = 100,
    lr: float = 0.1,
    l2_reg: float = 1e-2,
) -> tuple[torch.Tensor, list[float]]:
    """
    *Class model visualisation* (Simonyan et al., 2014):

    $I^* = \\arg\\max_I\\; S_c(I) - \\lambda \\lVert I \\rVert_2^2$,

    donde $S_c$ es el **logit** (sin softmax: maximizar la probabilidad se puede
    conseguir bajando las otras clases). Se parte de la imagen cero y se hace
    ascenso de gradiente simple: $I \\leftarrow I + lr \\cdot \\nabla_I F(I)$.
    Los parámetros del modelo **no** se modifican.

    Args:
        model: clasificador (en modo ``eval``).
        target_class: clase a visualizar.
        input_shape: forma de la imagen a optimizar.
        n_steps: número de pasos de ascenso.
        lr: tamaño de paso.
        l2_reg: $\\lambda$ de la regularización L2.

    Returns:
        Imagen optimizada (sin gradiente). Dimensions: ``input_shape``.
        Historial del objetivo $F(I)$ en cada paso (antes de actualizar). Longitud:
        ``n_steps``.
    """

    # BEGIN SOLUTION
    model.eval()
    image = torch.zeros(input_shape, requires_grad=True)
    history: list[float] = []
    for _ in range(n_steps):
        score = model(image)[0, target_class]
        objective = score - l2_reg * (image**2).sum()
        (grad,) = torch.autograd.grad(objective, image)
        history.append(objective.item())
        with torch.no_grad():
            image += lr * grad
    return image.detach(), history
    # END SOLUTION


def feature_inversion(
    feature_extractor: Callable[[torch.Tensor], torch.Tensor],
    target_image: torch.Tensor,
    n_steps: int = 200,
    lr: float = 0.05,
    l2_reg: float = 0.0,
) -> tuple[torch.Tensor, list[float]]:
    """
    *Feature inversion* (Mahendran & Vedaldi): busca una imagen cuyas features
    coincidan con las de ``target_image``:

    $x^* = \\arg\\min_x \\lVert \\phi(x) - \\phi(x_0) \\rVert_2^2 + \\lambda
    \\lVert x \\rVert_2^2$.

    Se parte de ruido ``torch.rand_like(target_image)`` y se optimiza con
    ``torch.optim.Adam``.

    Args:
        feature_extractor: función $\\phi$ (p.ej. las primeras capas de una CNN).
        target_image: imagen original $x_0$. Dimensions: [1, channels, height, width].
        n_steps: número de iteraciones.
        lr: *learning rate* de Adam.
        l2_reg: $\\lambda$ de la regularización L2.

    Returns:
        Imagen reconstruida (sin gradiente). Dimensions: igual que ``target_image``.
        Historial de la pérdida. Longitud: ``n_steps``.
    """

    # BEGIN SOLUTION
    with torch.no_grad():
        target_features = feature_extractor(target_image)
    x = torch.rand_like(target_image).requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=lr)
    history: list[float] = []
    for _ in range(n_steps):
        optimizer.zero_grad()
        loss = ((feature_extractor(x) - target_features) ** 2).sum()
        loss = loss + l2_reg * (x**2).sum()
        loss.backward()
        optimizer.step()
        history.append(loss.item())
    return x.detach(), history
    # END SOLUTION


def deep_dream(
    feature_extractor: Callable[[torch.Tensor], torch.Tensor],
    image: torch.Tensor,
    n_steps: int = 20,
    lr: float = 0.05,
) -> torch.Tensor:
    """
    DeepDream: ascenso de gradiente **partiendo de la imagen real** para
    amplificar lo que la capa ya "ve". Objetivo: $\\tfrac{1}{2}\\,\\mathrm{mean}(a^2)$
    de las activaciones $a = \\phi(x)$.

    En cada paso el gradiente se normaliza por su media absoluta
    (``grad / (grad.abs().mean() + 1e-8)``) y la imagen se recorta a [0, 1].

    Args:
        feature_extractor: función $\\phi$ (capa elegida).
        image: imagen de partida en [0, 1]. Dimensions: [1, channels, height, width].
        n_steps: número de pasos.
        lr: tamaño de paso.

    Returns:
        Imagen "soñada" en [0, 1] (sin gradiente). Dimensions: igual que ``image``.
    """

    # BEGIN SOLUTION
    x = image.detach().clone()
    for _ in range(n_steps):
        x.requires_grad_(True)
        objective = 0.5 * (feature_extractor(x) ** 2).mean()
        (grad,) = torch.autograd.grad(objective, x)
        with torch.no_grad():
            x = (x + lr * grad / (grad.abs().mean() + 1e-8)).clamp(0.0, 1.0)
    return x.detach()
    # END SOLUTION
