"""
CV05 · Explicabilidad: utilidades comunes — ejercicios.
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

    # TODO: usa amin/amax sobre las dos últimas dimensiones con keepdim
    raise NotImplementedError("TODO")


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

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


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
