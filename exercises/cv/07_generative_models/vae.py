"""
CV07 · Variational Autoencoder (VAE y β-VAE) — ejercicios.
"""

import torch
import torch.nn.functional as F
from torch import nn


class VAE(nn.Module):
    """
    VAE con MLP para imágenes ``[batch, 1, 28, 28]`` en [0, 1].

    Arquitectura:
        - Encoder: ``Flatten`` → ``Linear(in_features, hidden_dim)`` → ReLU y dos
          cabezas lineales ``fc_mu`` y ``fc_log_var`` de ``hidden_dim`` a ``latent_dim``.
        - Decoder: ``Linear(latent_dim, hidden_dim)`` → ReLU →
          ``Linear(hidden_dim, in_features)`` → Sigmoid → reshape a la imagen.

    Args:
        image_shape: forma ``(channels, height, width)`` de cada imagen.
        hidden_dim: neuronas de la capa oculta.
        latent_dim: dimensión del espacio latente ``z``.
    """

    def __init__(
        self,
        image_shape: tuple[int, int, int] = (1, 28, 28),
        hidden_dim: int = 128,
        latent_dim: int = 8,
    ) -> None:
        super().__init__()

        self.image_shape = image_shape
        self.latent_dim = latent_dim
        in_features = image_shape[0] * image_shape[1] * image_shape[2]

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parámetros de q(z|x).

        Args:
            x: imágenes. Dimensions: [batch, channels, height, width].

        Returns:
            ``mu`` y ``log_var`` (log σ²). Dimensions: [batch, latent_dim] cada uno.
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def reparameterize(self, mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
        """
        Truco de reparametrización ``z = mu + sigma * eps`` con ``eps ~ N(0, I)``.

        En modo ``eval`` devuelve directamente ``mu`` (inferencia determinista).

        Args:
            mu: media. Dimensions: [batch, latent_dim].
            log_var: log-varianza. Dimensions: [batch, latent_dim].

        Returns:
            Muestra ``z``. Dimensions: [batch, latent_dim].
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodifica ``z`` (media de p(x|z)).

        Args:
            z: códigos latentes. Dimensions: [batch, latent_dim].

        Returns:
            Imágenes en [0, 1]. Dimensions: [batch, channels, height, width].
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass completo.

        Args:
            x: imágenes. Dimensions: [batch, channels, height, width].

        Returns:
            Reconstrucción ``[batch, channels, height, width]``, ``mu`` y ``log_var``
            ``[batch, latent_dim]``.
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    @torch.no_grad()
    def sample(self, n: int) -> torch.Tensor:
        """
        Genera ``n`` imágenes nuevas muestreando el prior ``z ~ N(0, I)``.

        Args:
            n: número de muestras.

        Returns:
            Imágenes generadas. Dimensions: [n, channels, height, width].
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")


def kl_divergence(mu: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    """
    KL(q(z|x) || N(0, I)) en forma cerrada, sumada en ``latent_dim`` y promediada
    en el batch: ``-0.5 * sum(1 + log_var - mu² - exp(log_var))``.

    Args:
        mu: media. Dimensions: [batch, latent_dim].
        log_var: log-varianza. Dimensions: [batch, latent_dim].

    Returns:
        Escalar (tensor de 0 dimensiones).
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def vae_loss(
    recon: torch.Tensor,
    x: torch.Tensor,
    mu: torch.Tensor,
    log_var: torch.Tensor,
    beta: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Pérdida del (β-)VAE = -ELBO: ``recon + beta * kl``.

    El término de reconstrucción es la BCE **sumada** sobre los píxeles y
    **promediada** en el batch (así es comparable con la KL, que también se suma
    sobre las dimensiones latentes).

    Args:
        recon: reconstrucción en [0, 1]. Dimensions: [batch, channels, height, width].
        x: imágenes originales en [0, 1]. Dimensions: igual que ``recon``.
        mu: media. Dimensions: [batch, latent_dim].
        log_var: log-varianza. Dimensions: [batch, latent_dim].
        beta: peso de la KL (β = 1 es el VAE estándar).

    Returns:
        Tupla ``(recon_loss, kl, total)`` de escalares.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


@torch.no_grad()
def interpolate(
    model: VAE, x1: torch.Tensor, x2: torch.Tensor, steps: int
) -> torch.Tensor:
    """
    Interpolación lineal en el espacio latente entre dos imágenes.

    Codifica ``x1`` y ``x2`` (usa las medias ``mu``), genera
    ``z_t = (1 - t) * mu1 + t * mu2`` para ``t = linspace(0, 1, steps)`` y decodifica.

    Args:
        model: VAE entrenado (no cambies su modo).
        x1: primera imagen. Dimensions: [1, channels, height, width].
        x2: segunda imagen. Dimensions: [1, channels, height, width].
        steps: número de puntos de la interpolación (incluye los extremos).

    Returns:
        Imágenes decodificadas. Dimensions: [steps, channels, height, width].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def vae_train_step(
    model: VAE, x: torch.Tensor, optimizer: torch.optim.Optimizer, beta: float = 1.0
) -> tuple[float, float, float]:
    """
    Un paso de entrenamiento del β-VAE.

    Args:
        model: VAE (se pone en modo ``train``).
        x: imágenes en [0, 1]. Dimensions: [batch, channels, height, width].
        optimizer: optimizador del VAE.
        beta: peso de la KL.

    Returns:
        ``(recon_loss, kl, total)`` como ``float``.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")
