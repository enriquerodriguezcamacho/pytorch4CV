"""
F01 · Tensores — soluciones.
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
    # BEGIN SOLUTION
    return torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
    # END SOLUTION


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
    # BEGIN SOLUTION
    return (images - mean.view(1, -1, 1, 1)) / std.view(1, -1, 1, 1)
    # END SOLUTION


def pairwise_sq_distances(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Distancias euclídeas al cuadrado entre todas las parejas, sin bucles.

    Args:
        a: tensor ``(N, D)``.
        b: tensor ``(M, D)``.

    Returns:
        tensor ``(N, M)`` donde ``out[i, j] = ||a[i] - b[j]||²``.
    """
    # BEGIN SOLUTION
    a_sq = (a**2).sum(dim=1, keepdim=True)  # (N, 1)
    b_sq = (b**2).sum(dim=1).unsqueeze(0)  # (1, M)
    return (a_sq + b_sq - 2 * a @ b.T).clamp_min(0.0)
    # END SOLUTION


def select_by_index(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Devuelve el valor de ``logits`` en la clase indicada por ``labels``.

    Args:
        logits: tensor ``(B, K)``.
        labels: tensor ``int64`` ``(B,)`` con valores en ``[0, K)``.

    Returns:
        tensor ``(B,)`` con ``logits[i, labels[i]]``. Usa ``torch.gather``.
    """
    # BEGIN SOLUTION
    return torch.gather(logits, dim=1, index=labels.unsqueeze(1)).squeeze(1)
    # END SOLUTION


def one_hot_mask(mask: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Codifica en one-hot una máscara de segmentación.

    Args:
        mask: tensor ``int64`` ``(B, H, W)`` con el índice de clase de cada píxel.
        num_classes: número de clases ``K``.

    Returns:
        tensor ``float32`` ``(B, K, H, W)``. Usa ``Tensor.scatter_`` (no ``F.one_hot``).
    """
    # BEGIN SOLUTION
    b, h, w = mask.shape
    out = torch.zeros(b, num_classes, h, w, dtype=torch.float32, device=mask.device)
    return out.scatter_(1, mask.unsqueeze(1), 1.0)
    # END SOLUTION


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
    # BEGIN SOLUTION
    b, c, h, w = images.shape
    p = patch_size
    x = images.reshape(b, c, h // p, p, w // p, p)
    x = x.permute(0, 2, 4, 1, 3, 5)  # (B, H/p, W/p, C, p, p)
    return x.reshape(b, (h // p) * (w // p), c * p * p)
    # END SOLUTION


def batched_gram_matrix(features: torch.Tensor) -> torch.Tensor:
    """Matriz de Gram normalizada de un lote de mapas de características.

    Args:
        features: tensor ``(B, C, H, W)``.

    Returns:
        tensor ``(B, C, C)`` con ``G[b] = F_b F_bᵀ / (C·H·W)`` siendo ``F_b`` de
        forma ``(C, H·W)``. Usa ``torch.einsum``.
    """
    # BEGIN SOLUTION
    b, c, h, w = features.shape
    f = features.reshape(b, c, h * w)
    return torch.einsum("bci,bdi->bcd", f, f) / (c * h * w)
    # END SOLUTION


def topk_accuracy(logits: torch.Tensor, labels: torch.Tensor, k: int) -> float:
    """Exactitud top-k: fracción de ejemplos cuya clase real está en las k mayores.

    Args:
        logits: tensor ``(B, K)``.
        labels: tensor ``int64`` ``(B,)``.
        k: número de predicciones consideradas.

    Returns:
        exactitud como ``float`` en [0, 1].
    """
    # BEGIN SOLUTION
    topk = logits.topk(k, dim=1).indices  # (B, k)
    hits = (topk == labels.unsqueeze(1)).any(dim=1)
    return hits.float().mean().item()
    # END SOLUTION


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim: int) -> torch.Tensor:
    """Media de ``x`` en la dimensión ``dim`` contando solo los elementos con ``mask``.

    Típico para promediar una pérdida por píxel solo en los píxeles válidos o para
    hacer *mean pooling* de tokens ignorando el *padding*.

    Args:
        x: tensor de cualquier forma.
        mask: tensor booleano con la misma forma que ``x``.
        dim: dimensión que se reduce.

    Returns:
        tensor con ``dim`` eliminada. Donde no hay ningún elemento válido vale 0
        (divide por ``mask.sum(dim).clamp_min(1)``).
    """
    # BEGIN SOLUTION
    m = mask.to(x.dtype)
    return (x * m).sum(dim) / m.sum(dim).clamp_min(1)
    # END SOLUTION


def patches_to_image(patches: torch.Tensor, patch_size: int, height: int, width: int) -> torch.Tensor:
    """Operación inversa de :func:`image_to_patches`.

    Args:
        patches: tensor ``(B, N, C * p * p)`` con los parches ordenados por filas.
        patch_size: lado ``p`` del parche.
        height: alto ``H`` de la imagen original.
        width: ancho ``W`` de la imagen original.

    Returns:
        tensor ``(B, C, H, W)``.
    """
    # BEGIN SOLUTION
    b, _, d = patches.shape
    p = patch_size
    c = d // (p * p)
    x = patches.reshape(b, height // p, width // p, c, p, p)
    x = x.permute(0, 3, 1, 4, 2, 5)  # (B, C, H/p, p, W/p, p)
    return x.reshape(b, c, height, width)
    # END SOLUTION


def class_histogram(mask: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Número de píxeles de cada clase en cada imagen de un lote.

    Args:
        mask: tensor ``int64`` ``(B, H, W)`` con valores en ``[0, K)``.
        num_classes: número de clases ``K``.

    Returns:
        tensor ``int64`` ``(B, K)``. Usa ``Tensor.scatter_add_`` (sin bucles ni
        ``torch.bincount``).
    """
    # BEGIN SOLUTION
    flat = mask.flatten(1)
    out = torch.zeros(mask.shape[0], num_classes, dtype=torch.int64, device=mask.device)
    return out.scatter_add_(1, flat, torch.ones_like(flat))
    # END SOLUTION


def grid_coordinates(height: int, width: int) -> torch.Tensor:
    """Coordenadas ``(x, y)`` del centro de cada celda de una rejilla ``H × W``.

    Es lo que se usa para pasar de coordenadas relativas a la celda a coordenadas
    de imagen en YOLO o para generar *anchors*.

    Args:
        height: número de filas.
        width: número de columnas.

    Returns:
        tensor ``float32`` ``(H, W, 2)`` con ``out[i, j] = (j + 0.5, i + 0.5)``.
        Usa ``torch.meshgrid(..., indexing="ij")``.
    """
    # BEGIN SOLUTION
    ys = torch.arange(height, dtype=torch.float32) + 0.5
    xs = torch.arange(width, dtype=torch.float32) + 0.5
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([grid_x, grid_y], dim=-1)
    # END SOLUTION


def mask_to_box(mask: torch.Tensor) -> torch.Tensor:
    """Caja mínima ``xyxy`` que contiene los píxeles activos de cada máscara.

    Convención de ``torchvision.ops.masks_to_boxes``: ``x2`` e ``y2`` son el índice
    del último píxel activo (inclusivo).

    Args:
        mask: tensor booleano ``(N, H, W)``; cada máscara tiene al menos un píxel activo.

    Returns:
        tensor ``float32`` ``(N, 4)``. Sin bucles en ``H`` ni ``W`` (usa ``any`` y
        ``argmax`` sobre los ejes).
    """
    # BEGIN SOLUTION
    _, h, w = mask.shape
    rows = mask.any(dim=2)  # (N, H)
    cols = mask.any(dim=1)  # (N, W)
    y1 = rows.int().argmax(dim=1)
    y2 = h - 1 - rows.flip(1).int().argmax(dim=1)
    x1 = cols.int().argmax(dim=1)
    x2 = w - 1 - cols.flip(1).int().argmax(dim=1)
    return torch.stack([x1, y1, x2, y2], dim=1).float()
    # END SOLUTION


def batched_index_select(features: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Selecciona ``K`` vectores de cada elemento del lote.

    Por ejemplo, quedarse con los ``K`` tokens o *anchors* con más puntuación.

    Args:
        features: tensor ``(B, N, D)``.
        indices: tensor ``int64`` ``(B, K)`` con valores en ``[0, N)``.

    Returns:
        tensor ``(B, K, D)`` con ``out[b, k] = features[b, indices[b, k]]``. Usa
        ``torch.gather`` expandiendo los índices a ``(B, K, D)``.
    """
    # BEGIN SOLUTION
    index = indices.unsqueeze(-1).expand(-1, -1, features.shape[-1])
    return torch.gather(features, 1, index)
    # END SOLUTION
