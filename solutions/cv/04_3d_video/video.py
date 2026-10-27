"""
CV04 · 3D y vídeo: modelos de clasificación de vídeo e inflado I3D — soluciones.

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

        # BEGIN SOLUTION
        b, t, c, h, w = video.shape
        logits = self.cnn(video.reshape(b * t, c, h, w))  # [B*T, num_classes]
        return logits.view(b, t, -1).mean(dim=1)
        # END SOLUTION


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

        # BEGIN SOLUTION
        self.cnn = frame_encoder(feat_dim)
        self.classifier = nn.Linear(num_frames * feat_dim, num_classes)
        # END SOLUTION

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            video: Dimensions: [batch, T, 3, height, width].

        Returns:
            Logits. Dimensions: [batch, num_classes].
        """

        # BEGIN SOLUTION
        b, t, c, h, w = video.shape
        feats = self.cnn(video.reshape(b * t, c, h, w)).view(b, -1)  # [B, T*feat]
        return self.classifier(feats)
        # END SOLUTION


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

        # BEGIN SOLUTION
        self.net = nn.Sequential(
            nn.Conv2d(3 * num_frames, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(16, num_classes),
        )
        # END SOLUTION

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            video: Dimensions: [batch, T, 3, height, width].

        Returns:
            Logits. Dimensions: [batch, num_classes].
        """

        # BEGIN SOLUTION
        b, t, c, h, w = video.shape
        return self.net(video.reshape(b, t * c, h, w))
        # END SOLUTION


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

        # BEGIN SOLUTION
        self.net = nn.Sequential(
            nn.Conv3d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d((1, 2, 2)),
            nn.Conv3d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
            nn.Linear(16, num_classes),
        )
        # END SOLUTION

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            video: Dimensions: [batch, T, 3, height, width].

        Returns:
            Logits. Dimensions: [batch, num_classes].
        """

        # BEGIN SOLUTION
        return self.net(video.permute(0, 2, 1, 3, 4))  # [B, 3, T, H, W]
        # END SOLUTION


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

        # BEGIN SOLUTION
        self.cnn = frame_encoder(feat_dim)
        self.lstm = nn.LSTM(feat_dim, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, num_classes)
        # END SOLUTION

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Forward pass. Admite cualquier número de *frames* ``T``.

        Args:
            video: Dimensions: [batch, T, 3, height, width].

        Returns:
            Logits. Dimensions: [batch, num_classes].
        """

        # BEGIN SOLUTION
        b, t, c, h, w = video.shape
        feats = self.cnn(video.reshape(b * t, c, h, w)).view(b, t, -1)
        _, (h_n, _) = self.lstm(feats)  # h_n: [1, B, hidden]
        return self.fc(h_n[-1])
        # END SOLUTION


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
        # BEGIN SOLUTION
        fast_channels = slow_channels // beta_inv
        self.slow = nn.Sequential(
            nn.Conv3d(3, slow_channels, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool3d(1)
        )
        self.fast = nn.Sequential(
            nn.Conv3d(3, fast_channels, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool3d(1)
        )
        self.fc = nn.Linear(slow_channels + fast_channels, num_classes)
        # END SOLUTION

    def forward(self, video: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            video: Dimensions: [batch, T, 3, height, width].

        Returns:
            Logits. Dimensions: [batch, num_classes].
        """

        # BEGIN SOLUTION
        x = video.permute(0, 2, 1, 3, 4)  # [B, 3, T, H, W]
        slow = self.slow(x[:, :, :: self.alpha]).flatten(1)
        fast = self.fast(x).flatten(1)
        return self.fc(torch.cat([slow, fast], dim=1))
        # END SOLUTION


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

    # BEGIN SOLUTION
    if isinstance(conv2d.padding, str):
        raise ValueError("Solo se admite padding numérico (no 'same'/'valid').")
    kh, kw = conv2d.kernel_size[0], conv2d.kernel_size[1]
    conv3d = nn.Conv3d(
        conv2d.in_channels,
        conv2d.out_channels,
        kernel_size=(time_kernel, kh, kw),
        stride=(1, conv2d.stride[0], conv2d.stride[1]),
        padding=(time_kernel // 2, conv2d.padding[0], conv2d.padding[1]),
        bias=conv2d.bias is not None,
    )
    with torch.no_grad():
        weight = conv2d.weight.unsqueeze(2).repeat(1, 1, time_kernel, 1, 1)
        conv3d.weight.copy_(weight / time_kernel)
        if conv2d.bias is not None:
            conv3d.bias.copy_(conv2d.bias)  # type: ignore[union-attr]
    return conv3d
    # END SOLUTION
