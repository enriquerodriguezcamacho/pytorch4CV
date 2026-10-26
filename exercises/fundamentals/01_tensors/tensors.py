"""
F01 · Tensores — ejercicios.
"""

import numpy as np
import torch


def to_chw_tensor(image: np.ndarray) -> torch.Tensor:
    """Convierte una imagen ``uint8`` de NumPy en un tensor float de PyTorch.

    Args:
        image: array ``uint8`` de forma ``(H, W, C)`` con valores en [0, 255].

    Returns:
        tensor ``float32`` de forma ``(C, H, W)`` con valores en [0, 1].
    """
    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def normalize_per_channel(
    images: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
) -> torch.Tensor:
    """Normaliza un lote de imágenes canal a canal.

    Args:
        images: tensor ``(B, C, H, W)``.
        mean: tensor ``(C,)`` con la media de cada canal.
        std: tensor ``(C,)`` con la desviación típica de cada canal.

    Returns:
        tensor ``(B, C, H, W)`` con ``(images - mean) / std`` por canal.
    """
    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def pairwise_sq_distances(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Distancias euclídeas al cuadrado entre todas las parejas, sin bucles.

    Args:
        a: tensor ``(N, D)``.
        b: tensor ``(M, D)``.

    Returns:
        tensor ``(N, M)`` donde ``out[i, j] = ||a[i] - b[j]||²``.
    """
    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def select_by_index(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Devuelve el valor de ``logits`` en la clase indicada por ``labels``.

    Args:
        logits: tensor ``(B, K)``.
        labels: tensor ``int64`` ``(B,)`` con valores en ``[0, K)``.

    Returns:
        tensor ``(B,)`` con ``logits[i, labels[i]]``. Usa ``torch.gather``.
    """
    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def one_hot_mask(mask: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Codifica en one-hot una máscara de segmentación.

    Args:
        mask: tensor ``int64`` ``(B, H, W)`` con el índice de clase de cada píxel.
        num_classes: número de clases ``K``.

    Returns:
        tensor ``float32`` ``(B, K, H, W)``. Usa ``Tensor.scatter_`` (no ``F.one_hot``).
    """
    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def image_to_patches(images: torch.Tensor, patch_size: int) -> torch.Tensor:
    """Divide imágenes en parches no solapados y los aplana (como en ViT).

    Args:
        images: tensor ``(B, C, H, W)`` con ``H`` y ``W`` múltiplos de ``patch_size``.
        patch_size: lado ``p`` del parche.

    Returns:
        tensor ``(B, N, C * p * p)`` con ``N = (H / p) * (W / p)``. Los parches se
        ordenan por filas (de izquierda a derecha y de arriba abajo) y cada parche
        se aplana en orden ``(C, p, p)``.
    """
    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def batched_gram_matrix(features: torch.Tensor) -> torch.Tensor:
    """Matriz de Gram normalizada de un lote de mapas de características.

    Args:
        features: tensor ``(B, C, H, W)``.

    Returns:
        tensor ``(B, C, C)`` con ``G[b] = F_b F_bᵀ / (C·H·W)`` siendo ``F_b`` de
        forma ``(C, H·W)``. Usa ``torch.einsum``.
    """
    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    """Exactitud top-k: fracción de ejemplos cuya clase real está en las k mayores.

    Args:
        logits: tensor ``(B, K)``.
        labels: tensor ``int64`` ``(B,)``.
        k: número de predicciones consideradas.

    Returns:
        exactitud como ``float`` en [0, 1].
    """
    # TODO: implementa esta parte
    raise NotImplementedError("TODO")
