"""
CV04 · 3D y vídeo: modelos de clasificación de vídeo e inflado I3D — ejercicios.

Todos los modelos reciben vídeos en el formato de ``torchvision.io.read_video`` tras
pasar a *float*: ``[batch, T, 3, H, W]`` (tiempo antes que canales). Lo importante
del capítulo es el **manejo de las formas**: ``view(B*T, ...)`` para aplicar una CNN
2D a cada *frame*, ``reshape(B, 3T, H, W)`` para *early fusion* y
``permute(0, 2, 1, 3, 4)`` para las CNN 3D (que esperan ``[B, C, T, H, W]``).
"""

from __future__ import annotations

import torch
from torch import nn


def frame_encoder(out_features: int) -> nn.Module:
    """
    CNN 2D pequeña que se reutiliza en varios modelos. Ya dada.

    ``Conv2d(3, 8, 3, padding=1) → ReLU → AdaptiveAvgPool2d(1) → Flatten →
    Linear(8, out_features)``.

    Args:
        out_features: dimensión del vector de salida.

    Returns:
        Módulo que lleva ``[N, 3, H, W]`` a ``[N, out_features]``.
    """

    return nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(8, out_features),
    )


class SingleFrameModel(nn.Module):
    """
    *Single-frame*: clasifica cada *frame* por separado con una CNN 2D y promedia
    los **logits** de todos los *frames*.
    """

    def __init__(self, num_classes: int) -> None:
        """
        Constructor de la clase (ya dado).

        Args:
            num_classes: número de clases.
        """

        super().__init__()

        self.cnn = frame_encoder(num_classes)

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            video: Dimensions: [batch, T, 3, height, width].

        Returns:
            Logits medios. Dimensions: [batch, num_classes].
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")


class LateFusionModel(nn.Module):
    """
    *Late fusion*: features por *frame* con la CNN 2D, se **concatenan** las ``T``
    features en orden temporal y un MLP ``Linear(T · feat, num_classes)`` clasifica.
    """

    def __init__(self, num_frames: int, num_classes: int, feat_dim: int = 16) -> None:
        """
        Constructor de la clase.

        Args:
            num_frames: número de *frames* ``T`` (fijo).
            num_classes: número de clases.
            feat_dim: dimensión de las features por *frame*.
        """

        super().__init__()

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            video: Dimensions: [batch, T, 3, height, width].

        Returns:
            Logits. Dimensions: [batch, num_classes].
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")


class EarlyFusionModel(nn.Module):
    """
    *Early fusion*: se apilan los ``T`` *frames* en la dimensión de canales
    (``[B, 3T, H, W]``, *frame* 0 en los canales 0–2) y una sola CNN 2D procesa el
    resultado: ``Conv2d(3T, 16, 3, padding=1) → ReLU → AdaptiveAvgPool2d(1) →
    Flatten → Linear(16, num_classes)``.
    """

    def __init__(self, num_frames: int, num_classes: int) -> None:
        """
        Constructor de la clase.

        Args:
            num_frames: número de *frames* ``T``.
            num_classes: número de clases.
        """

        super().__init__()

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            video: Dimensions: [batch, T, 3, height, width].

        Returns:
            Logits. Dimensions: [batch, num_classes].
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")


class C3DModel(nn.Module):
    """
    CNN 3D mínima (estilo C3D): ``Conv3d(3, 8, 3, padding=1) → ReLU →
    MaxPool3d((1, 2, 2)) → Conv3d(8, 16, 3, padding=1) → ReLU →
    AdaptiveAvgPool3d(1) → Flatten → Linear(16, num_classes)``.

    El primer *pooling* no reduce el tiempo (como en C3D, para no perder la
    información temporal demasiado pronto).
    """

    def __init__(self, num_classes: int) -> None:
        """
        Constructor de la clase.

        Args:
            num_classes: número de clases.
        """

        super().__init__()

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            video: Dimensions: [batch, T, 3, height, width].

        Returns:
            Logits. Dimensions: [batch, num_classes].
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")


class CNNLSTMModel(nn.Module):
    """
    CNN + LSTM: features por *frame* con la CNN 2D, una ``nn.LSTM`` con
    ``batch_first=True`` sobre la secuencia y una capa lineal sobre el **último
    estado oculto** ``h_T``.
    """

    def __init__(self, num_classes: int, feat_dim: int = 16, hidden: int = 32) -> None:
        """
        Constructor de la clase.

        Args:
            num_classes: número de clases.
            feat_dim: dimensión de las features por *frame*.
            hidden: tamaño del estado oculto de la LSTM.
        """

        super().__init__()

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Admite cualquier número de *frames* ``T``.

        Args:
            video: Dimensions: [batch, T, 3, height, width].

        Returns:
            Logits. Dimensions: [batch, num_classes].
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")


class SlowFastModel(nn.Module):
    """
    SlowFast simplificado (Feichtenhofer et al., 2019), sin conexiones laterales:

    * Vía **lenta**: ``video[:, ::alpha]`` (uno de cada ``alpha`` *frames*) y
      ``Conv3d(3, slow_channels, 3, padding=1)``.
    * Vía **rápida**: todos los *frames* y ``Conv3d(3, slow_channels // beta_inv,
      3, padding=1)`` (pocos canales).

    Cada vía: conv → ReLU → ``AdaptiveAvgPool3d(1)`` → ``flatten``. Se concatenan
    (lenta primero) y ``Linear`` a ``num_classes``.
    """

    def __init__(
        self, num_classes: int, alpha: int = 4, slow_channels: int = 16, beta_inv: int = 8
    ) -> None:
        """
        Constructor de la clase.

        Args:
            num_classes: número de clases.
            alpha: factor de submuestreo temporal de la vía lenta.
            slow_channels: canales de la vía lenta.
            beta_inv: la vía rápida tiene ``slow_channels // beta_inv`` canales.
        """

        super().__init__()

        self.alpha = alpha
        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            video: Dimensions: [batch, T, 3, height, width].

        Returns:
            Logits. Dimensions: [batch, num_classes].
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")


def inflate_conv2d(conv2d: nn.Conv2d, time_kernel: int) -> nn.Conv3d:
    """
    Inflado I3D: convierte una ``Conv2d`` (preentrenada) en una ``Conv3d`` con
    kernel temporal ``time_kernel``, repitiendo los pesos en el tiempo y
    dividiéndolos entre ``time_kernel``: ``W3D[:, :, t] = W2D / T``.

    Así, sobre un vídeo "aburrido" (el mismo *frame* repetido) la ``Conv3d`` da
    la misma respuesta que la ``Conv2d`` en cada *frame* interior (en los
    extremos el padding temporal con ceros cambia el resultado). El padding temporal es
    ``time_kernel // 2``; *stride* y padding espaciales se copian; el sesgo se
    copia tal cual.

    Args:
        conv2d: convolución 2D de origen (stride, padding y dilation enteros
            o tuplas).
        time_kernel: tamaño ``T`` del kernel temporal (impar).

    Returns:
        Convolución 3D inflada.

    Raises:
        ValueError: si ``conv2d`` usa padding de tipo cadena (``"same"``).
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")
