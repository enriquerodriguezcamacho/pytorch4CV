"""
CV03 · Style transfer: funciones auxiliares (Gram, normalización, AdaIN) — ejercicios.
"""

from typing import Literal

import torch
from PIL import Image
from torchvision import transforms

from .config import config


def gram_matrix(
    x: torch.Tensor, normalize: Literal["chw", "hw", "none"] = "chw"
) -> torch.Tensor:
    """
    Matriz de Gram de un mapa de features.

    Con ``X`` de forma ``[channels, height * width]`` (por elemento del lote),
    ``G = X @ X^T``. Normalización:

    * ``"chw"``: se divide por ``channels * height * width`` (la del libro/examen).
    * ``"hw"``: se divide por ``height * width``.
    * ``"none"``: sin normalizar.

    Args:
        x: mapa de features. Dimensions: [batch, channels, height, width].
        normalize: tipo de normalización.

    Returns:
        Matriz de Gram. Dimensions: [batch, channels, channels].

    Raises:
        ValueError: si ``normalize`` no es una de las opciones válidas.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def normalize(
    x: torch.Tensor,
    means: list[float] = config.model.means,
    stds: list[float] = config.model.stds,
) -> torch.Tensor:
    """
    Normaliza una imagen en ``[0, 1]`` con la media y desviación de ImageNet.

    Args:
        x: imagen. Dimensions: [batch, 3, height, width].
        means: media por canal.
        stds: desviación típica por canal.

    Returns:
        Imagen normalizada. Dimensions: [batch, 3, height, width].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def denormalize(
    x: torch.Tensor,
    means: list[float] = config.model.means,
    stds: list[float] = config.model.stds,
) -> torch.Tensor:
    """
    Deshace la normalización de ImageNet y recorta a ``[0, 1]``.

    Args:
        x: imagen normalizada. Dimensions: [batch, 3, height, width].
        means: media por canal.
        stds: desviación típica por canal.

    Returns:
        Imagen en ``[0, 1]``. Dimensions: [batch, 3, height, width].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def adain(
    content_features: torch.Tensor, style_features: torch.Tensor, eps: float = 1e-5
) -> torch.Tensor:
    """
    Adaptive Instance Normalization (Huang & Belongie, 2017).

    ``AdaIN(x, y) = sigma(y) * (x - mu(x)) / sigma(x) + mu(y)``, con media y
    desviación típica (sin corrección de Bessel) calculadas por canal sobre las
    dimensiones espaciales. ``eps`` se suma a la varianza de ``x``.

    Args:
        content_features: features de contenido. Dimensions: [batch, c, h, w].
        style_features: features de estilo. Dimensions: [batch, c, h', w'].
        eps: estabilidad numérica.

    Returns:
        Features con la estadística del estilo. Dimensions: [batch, c, h, w].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def load_image(path: str, size: int = config.model.size_images) -> torch.Tensor:
    """
    Carga una imagen como tensor normalizado para la red (función dada).

    Args:
        path: ruta de la imagen.
        size: alto y ancho al que se redimensiona.

    Returns:
        Imagen normalizada. Dimensions: [1, 3, size, size].
    """

    image = Image.open(path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=config.model.means, std=config.model.stds),
        ]
    )
    return transform(image).unsqueeze(0).to(config.model.device)


def save_image(x: torch.Tensor, save_path: str) -> None:
    """
    Guarda una imagen normalizada en disco (función dada).

    Args:
        x: imagen normalizada. Dimensions: [1, 3, height, width].
        save_path: ruta del fichero ``.png``.
    """

    image = denormalize(x.detach().cpu())[0]
    array = (image.permute(1, 2, 0).numpy() * 255).astype("uint8")
    Image.fromarray(array).save(save_path)
