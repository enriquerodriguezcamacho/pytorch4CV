"""
F11 · Arquitecturas: autoencoder convolucional — soluciones.

Esquema encoder-decoder: el encoder reduce la resolución con convoluciones de
``stride=2`` y el decoder la recupera con convoluciones transpuestas.
"""

import torch
from torch import nn


class ConvAutoencoder(nn.Module):
    """
    Autoencoder convolucional para imágenes en [0, 1] (p.ej. MNIST 28×28).

    * ``self.encoder``: Conv(in, 16, 3, stride 2, padding 1), ReLU,
      Conv(16, latent_channels, 3, stride 2, padding 1), ReLU.
    * ``self.decoder``: ConvTranspose(latent_channels, 16, 3, stride 2, padding 1,
      output_padding 1), ReLU, ConvTranspose(16, in, 3, stride 2, padding 1,
      output_padding 1), Sigmoid.

    Con entrada ``H×W`` (múltiplos de 4) el código tiene tamaño ``H/4 × W/4``.
    """

    def __init__(self, in_channels: int = 1, latent_channels: int = 32) -> None:
        """
        Constructor.

        Args:
            in_channels: canales de la imagen.
            latent_channels: canales del código latente.
        """

        super().__init__()

        # BEGIN SOLUTION
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, latent_channels, 3, stride=2, padding=1),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_channels, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(16, in_channels, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )
        # END SOLUTION

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Codifica.

        Args:
            x: imágenes. Dimensions: [batch, in_channels, height, width].

        Returns:
            Código. Dimensions: [batch, latent_channels, height / 4, width / 4].
        """

        # BEGIN SOLUTION
        return self.encoder(x)
        # END SOLUTION

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodifica.

        Args:
            z: código. Dimensions: [batch, latent_channels, h, w].

        Returns:
            Reconstrucción en [0, 1]. Dimensions: [batch, in_channels, 4h, 4w].
        """

        # BEGIN SOLUTION
        return self.decoder(z)
        # END SOLUTION

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass (codifica y decodifica).

        Args:
            x: imágenes. Dimensions: [batch, in_channels, height, width].

        Returns:
            Reconstrucción. Dimensions: [batch, in_channels, height, width].
        """

        # BEGIN SOLUTION
        return self.decode(self.encode(x))
        # END SOLUTION


def train_autoencoder_step(
    model: ConvAutoencoder, x: torch.Tensor, optimizer: torch.optim.Optimizer
) -> float:
    """
    Un paso de entrenamiento con pérdida MSE de reconstrucción.

    Pone el modelo en modo ``train``, limpia gradientes, calcula
    ``MSE(model(x), x)``, hace ``backward`` y ``step``.

    Args:
        model: autoencoder.
        x: lote de imágenes en [0, 1]. Dimensions: [batch, channels, height, width].
        optimizer: optimizador de los parámetros del modelo.

    Returns:
        Valor de la pérdida (``float``) antes del paso.
    """

    # BEGIN SOLUTION
    model.train()
    optimizer.zero_grad()
    loss = nn.functional.mse_loss(model(x), x)
    loss.backward()
    optimizer.step()
    return loss.item()
    # END SOLUTION
