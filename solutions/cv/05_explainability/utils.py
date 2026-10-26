"""
CV05 · Explicabilidad: utilidades comunes — soluciones.
"""

import torch


def normalize_map(explanation: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Normaliza min-max un mapa de explicación, **cada muestra por separado**.

    Si la entrada es 2D se trata como una única muestra. Un mapa constante
    devuelve ceros (gracias a ``eps`` no se divide por cero).

    Args:
        explanation: mapa a normalizar. Dimensions: [batch, height, width] o
            [height, width].
        eps: constante para evitar la división por cero.

    Returns:
        Mapa con valores en [0, 1] y la misma forma que la entrada.
    """

    # BEGIN SOLUTION usa amin/amax sobre las dos últimas dimensiones con keepdim
    min_ = explanation.amin(dim=(-2, -1), keepdim=True)
    max_ = explanation.amax(dim=(-2, -1), keepdim=True)
    return (explanation - min_) / (max_ - min_ + eps)
    # END SOLUTION


def select_target(logits: torch.Tensor, target_class: int | None) -> torch.Tensor:
    """
    Devuelve el índice de la clase a explicar para cada muestra.

    Args:
        logits: salida del modelo. Dimensions: [batch, num_classes].
        target_class: clase a explicar. Si es ``None`` se usa la clase predicha
            (``argmax``) de cada muestra.

    Returns:
        Índices ``int64``. Dimensions: [batch].

    Raises:
        ValueError: si ``target_class`` no está en ``[0, num_classes)``.
    """

    # BEGIN SOLUTION
    batch, num_classes = logits.shape
    if target_class is None:
        return logits.argmax(dim=1)
    if not 0 <= target_class < num_classes:
        raise ValueError(f"target_class debe estar en [0, {num_classes}).")
    return torch.full((batch,), target_class, dtype=torch.long, device=logits.device)
    # END SOLUTION


def target_scores(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Selecciona el *score* (logit) de la clase objetivo de cada muestra. Ya dada.

    Args:
        logits: salida del modelo. Dimensions: [batch, num_classes].
        targets: índices de clase. Dimensions: [batch].

    Returns:
        Scores seleccionados. Dimensions: [batch].
    """

    return logits.gather(1, targets.view(-1, 1)).squeeze(1)
