"""
CV01 · Ejercicios de código del tema 1 de los apuntes (sección 1.4) — ejercicios.

Cada función indica el ejercicio al que corresponde. Los ejercicios teóricos
están resueltos en ``docs/SOLUCIONES_APUNTES.md``.
"""

from __future__ import annotations

import math

from torch import nn
from torchvision import transforms as T


def conv_output_shape(
    input_shape: tuple[int, int, int],
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> tuple[int, int, int]:
    """
    Ej. 3 · Forma de salida de una ``Conv2d`` con la ecuación estándar:

    ``out = floor((in + 2p - d(k - 1) - 1) / s) + 1``.

    Args:
        input_shape: forma de entrada ``(C, H, W)``.
        out_channels: número de filtros.
        kernel_size: tamaño del kernel.
        stride: paso.
        padding: padding.
        dilation: dilatación.

    Returns:
        Forma de salida ``(out_channels, H_out, W_out)``.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def receptive_field_per_layer(kernel_sizes: list[int]) -> list[int]:
    """
    Ej. 5 · Campo receptivo tras cada capa con ``s = 1`` y ``d = 1``:

    ``RF_0 = 1`` y ``RF_L = RF_{L-1} + (k_L - 1)``.

    Args:
        kernel_sizes: kernel de cada capa, desde la entrada hacia la salida.

    Returns:
        ``[RF_0, RF_1, ..., RF_L]``.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def adapt_resnet_to_grayscale(model: nn.Module, num_classes: int = 10) -> nn.Module:
    """
    Ej. 2 · Adapta una ResNet a imágenes de 1 canal (MNIST), en el sitio.

    * ``conv1`` pasa a tener 1 canal de entrada, conserva sus hiperparámetros y su
      peso es la **media** de los pesos RGB, para reutilizar lo aprendido.
    * ``fc`` se sustituye por una capa nueva de ``num_classes`` salidas.

    Args:
        model: ResNet de torchvision (con ``conv1`` y ``fc``).
        num_classes: número de clases.

    Returns:
        El mismo modelo modificado.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def build_train_transforms(size: int, crop_padding: int = 4) -> T.Compose:
    """
    Ej. 7 · Pipeline de entrenamiento, en este orden: ``RandomCrop`` (con padding),
    ``RandomHorizontalFlip``, ``ColorJitter`` y ``ToTensor``.

    Args:
        size: lado del recorte.
        crop_padding: padding del recorte.

    Returns:
        Transformación compuesta.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def prepare_vgg_feature_extractor(model: nn.Module, num_classes: int = 10) -> nn.Module:
    """
    Ej. 8 · Congela las convoluciones de una VGG y cambia su última capa, en el sitio.

    Solo se congela ``model.features``: las capas densas de ``model.classifier`` y la
    capa final nueva siguen siendo entrenables.

    Args:
        model: VGG de torchvision (``features`` y ``classifier`` secuencial).
        num_classes: número de clases.

    Returns:
        El mismo modelo modificado.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")
