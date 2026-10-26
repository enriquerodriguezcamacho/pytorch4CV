"""
CV07 · Autoencoders convolucionales (y denoising AE) — ejercicios.
"""

import torch
import torch.nn.functional as F
from torch import nn


class ConvAutoencoder(nn.Module):
    """
    Autoencoder convolucional para imágenes 28x28 (tipo MNIST).

    Arquitectura:
        - Encoder: ``Conv2d(in, hidden, 3, stride=2, padding=1)`` + ReLU (28 → 14),
          ``Conv2d(hidden, latent_channels, 3, stride=2, padding=1)`` + ReLU (14 → 7).
        - Decoder: ``ConvTranspose2d(latent_channels, hidden, 4, stride=2, padding=1)``
          + ReLU (7 → 14), ``ConvTranspose2d(hidden, in, 4, stride=2, padding=1)``
          + Sigmoid (14 → 28).

    Args:
        in_channels: canales de la imagen.
        hidden_channels: canales de la capa intermedia.
        latent_channels: canales del código latente (cuello de botella de 7x7).
    """

    def __init__(
        self, in_channels: int = 1, hidden_channels: int = 16, latent_channels: int = 4
    ) -> None:
        super().__init__()

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Codifica la imagen.

        Args:
            x: imágenes. Dimensions: [batch, in_channels, 28, 28].

        Returns:
            Código latente. Dimensions: [batch, latent_channels, 7, 7].
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodifica el código latente.

        Args:
            z: código latente. Dimensions: [batch, latent_channels, 7, 7].

        Returns:
            Reconstrucción en [0, 1]. Dimensions: [batch, in_channels, 28, 28].
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (codifica y decodifica).

        Args:
            x: imágenes. Dimensions: [batch, in_channels, 28, 28].

        Returns:
            Reconstrucción. Dimensions: [batch, in_channels, 28, 28].
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")


def add_gaussian_noise(x: torch.Tensor, std: float) -> torch.Tensor:
    """
    Corrompe una imagen con ruido gaussiano (entrada de un *denoising* AE).

    Args:
        x: imágenes en [0, 1]. Dimensions: [batch, channels, height, width].
        std: desviación típica del ruido.

    Returns:
        ``clamp(x + std * N(0, 1), 0, 1)``. No modifica ``x``. Dimensions: igual que x.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def ae_train_step(
    model: nn.Module,
    x: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    noise_std: float = 0.0,
) -> float:
    """
    Un paso de entrenamiento de un (denoising) autoencoder con MSE.

    Si ``noise_std > 0`` la **entrada** es la imagen con ruido, pero el
    **objetivo** sigue siendo la imagen limpia.

    Args:
        model: autoencoder (se pone en modo ``train``).
        x: imágenes limpias. Dimensions: [batch, channels, height, width].
        optimizer: optimizador de ``model``.
        noise_std: desviación típica del ruido de entrada (0 = AE normal).

    Returns:
        Valor de la pérdida MSE como ``float``.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")
