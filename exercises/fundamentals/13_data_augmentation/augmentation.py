"""
F13 · Aumento de datos — ejercicios.

En clasificación basta con transformar la imagen, pero en segmentación y detección
la máscara y las cajas tienen que transformarse **igual** que la imagen. Aquí se
programan esas transformaciones a mano (deben coincidir con
``torchvision.transforms.v2``), unas transformaciones de color y la
*test-time augmentation* (TTA).

Convenciones: imágenes ``float`` en [0, 1] con forma ``[C, H, W]``, máscaras
``[H, W]`` y cajas ``xyxy`` en píxeles con forma ``[N, 4]``.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

Target = dict[str, torch.Tensor]

# --------------------------------------------------------------------------- #
# Transformaciones geométricas                                                #
# --------------------------------------------------------------------------- #


def hflip_boxes(boxes: torch.Tensor, width: int) -> torch.Tensor:
    """
    Volteo horizontal de cajas: ``x1' = W - x2`` y ``x2' = W - x1`` (``y`` no cambia).

    Args:
        boxes: cajas ``xyxy``. Dimensions: [N, 4].
        width: ancho ``W`` de la imagen.

    Returns:
        Cajas volteadas (sigue cumpliéndose ``x1 <= x2``). Dimensions: [N, 4].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def hflip(
    image: torch.Tensor, target: Target | None = None
) -> tuple[torch.Tensor, Target | None]:
    """
    Volteo horizontal de la imagen y, si existen, de ``target["masks"]``
    (``[H, W]`` o ``[N, H, W]``) y ``target["boxes"]``. El resto de claves se copian.

    Args:
        image: imagen. Dimensions: [C, H, W].
        target: diccionario opcional con ``"boxes"``, ``"masks"``, ``"labels"``...

    Returns:
        Imagen volteada y un **nuevo** diccionario (no se modifica el original).
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def crop_with_boxes(
    image: torch.Tensor,
    boxes: torch.Tensor,
    top: int,
    left: int,
    height: int,
    width: int,
    min_area_ratio: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Recorta la región ``[top, top + height) × [left, left + width)`` y ajusta las cajas.

    1. Las cajas se desplazan ``(-left, -top)`` y se recortan a ``[0, width] × [0, height]``.
    2. Se conservan las que, tras recortar, tienen área ``> 0`` **y** conservan al
       menos ``min_area_ratio`` de su área original.

    Args:
        image: imagen. Dimensions: [C, H, W].
        boxes: cajas ``xyxy``. Dimensions: [N, 4].
        top: fila de inicio del recorte.
        left: columna de inicio del recorte.
        height: alto del recorte.
        width: ancho del recorte.
        min_area_ratio: fracción mínima de área visible para conservar una caja.

    Returns:
        Imagen recortada, cajas conservadas y máscara booleana de las conservadas
        (para filtrar ``labels``). Dimensions: [C, height, width], [K, 4] y [N].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def resize_with_boxes(
    image: torch.Tensor, boxes: torch.Tensor, size: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Redimensiona la imagen a ``size = (new_h, new_w)`` con interpolación bilineal
    (``align_corners=False``, ``antialias=False``) y escala las cajas:
    ``x · new_w / W`` e ``y · new_h / H``.

    Args:
        image: imagen. Dimensions: [C, H, W].
        boxes: cajas ``xyxy``. Dimensions: [N, 4].
        size: tamaño de salida ``(new_h, new_w)``.

    Returns:
        Imagen y cajas redimensionadas. Dimensions: [C, new_h, new_w] y [N, 4].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def resize_mask(mask: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """
    Redimensiona una máscara de clases con vecino más próximo (``mode="nearest"``):
    interpolar etiquetas de forma bilineal inventaría clases que no existen.

    Args:
        mask: máscara ``int64``. Dimensions: [H, W].
        size: tamaño de salida ``(new_h, new_w)``.

    Returns:
        Máscara ``int64``. Dimensions: [new_h, new_w].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


# --------------------------------------------------------------------------- #
# Color                                                                       #
# --------------------------------------------------------------------------- #


def rgb_to_grayscale(image: torch.Tensor) -> torch.Tensor:
    """
    Luminancia ITU-R 601: ``0.2989 R + 0.587 G + 0.114 B``. Ya dada.

    Args:
        image: imagen RGB. Dimensions: [..., 3, H, W].

    Returns:
        Dimensions: [..., 1, H, W].
    """

    r, g, b = image.unbind(-3)
    return (0.2989 * r + 0.587 * g + 0.114 * b).unsqueeze(-3)


def adjust_brightness(image: torch.Tensor, factor: float) -> torch.Tensor:
    """
    Brillo: ``clamp(factor · x, 0, 1)`` (como ``TF.adjust_brightness``).

    Args:
        image: imagen. Dimensions: [..., C, H, W].
        factor: 0 → negro, 1 → igual, > 1 → más brillo.

    Returns:
        Dimensions: igual que ``image``.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def adjust_contrast(image: torch.Tensor, factor: float) -> torch.Tensor:
    """
    Contraste (como ``TF.adjust_contrast``): mezcla con la **media de la imagen en
    escala de grises** (un escalar por imagen):
    ``clamp(factor · x + (1 - factor) · mean(gray(x)), 0, 1)``.

    Args:
        image: imagen RGB. Dimensions: [..., 3, H, W].
        factor: 0 → gris uniforme, 1 → igual, > 1 → más contraste.

    Returns:
        Dimensions: igual que ``image``.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def cutout(image: torch.Tensor, top: int, left: int, size: int, value: float = 0.0) -> torch.Tensor:
    """
    *Cutout* / *random erasing*: rellena un cuadrado ``size × size`` (recortado a la
    imagen) con ``value`` sobre una **copia**.

    Args:
        image: imagen. Dimensions: [C, H, W].
        top: fila de la esquina superior izquierda.
        left: columna de la esquina superior izquierda.
        size: lado del cuadrado.
        value: valor de relleno.

    Returns:
        Dimensions: igual que ``image``.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


# --------------------------------------------------------------------------- #
# Composición aleatoria y TTA                                                 #
# --------------------------------------------------------------------------- #


class RandomHorizontalFlip:
    """
    Voltea imagen y *target* con probabilidad ``p``. La decisión se toma con
    ``torch.rand(1) < p`` (una sola llamada por muestra) para depender de la semilla.
    """

    def __init__(self, p: float = 0.5) -> None:
        """
        Constructor de la clase (ya dado).

        Args:
            p: probabilidad de voltear.
        """

        self.p = p

    def __call__(
        self, image: torch.Tensor, target: Target | None = None
    ) -> tuple[torch.Tensor, Target | None]:
        """
        Aplica la transformación.

        Args:
            image: imagen. Dimensions: [C, H, W].
            target: *target* opcional (ver :func:`hflip`).

        Returns:
            Imagen y *target* (volteados o no).
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")


class PairedCompose:
    """
    Encadena transformaciones que reciben y devuelven ``(image, target)``
    (``torchvision.transforms.Compose`` solo pasa la imagen).
    """

    def __init__(self, transforms: list) -> None:
        """
        Constructor de la clase (ya dado).

        Args:
            transforms: lista de *callables* ``(image, target) -> (image, target)``.
        """

        self.transforms = transforms

    def __call__(
        self, image: torch.Tensor, target: Target | None = None
    ) -> tuple[torch.Tensor, Target | None]:
        """
        Aplica las transformaciones en orden.

        Args:
            image: imagen. Dimensions: [C, H, W].
            target: *target* opcional.

        Returns:
            Imagen y *target* transformados.
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")


@torch.no_grad()
def tta_predict(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    *Test-time augmentation* para clasificación: media de las **probabilidades**
    (softmax) de la imagen original y de su volteo horizontal, con el modelo en
    modo ``eval``.

    Args:
        model: clasificador.
        x: imágenes. Dimensions: [batch, C, H, W].

    Returns:
        Probabilidades medias. Dimensions: [batch, num_classes].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


@torch.no_grad()
def tta_segmentation(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    TTA para segmentación: media de las probabilidades por píxel de la imagen y de
    su volteo, **deshaciendo el volteo** de la segunda predicción antes de promediar.

    Args:
        model: red de segmentación (devuelve logits ``[batch, K, H, W]``).
        x: imágenes. Dimensions: [batch, C, H, W].

    Returns:
        Probabilidades medias. Dimensions: [batch, K, H, W].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")
