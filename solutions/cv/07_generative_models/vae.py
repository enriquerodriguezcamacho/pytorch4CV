"""
CV07 · Variational Autoencoder (VAE y β-VAE) — soluciones.
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

        # BEGIN SOLUTION
        self.encoder = nn.Sequential(
            nn.Flatten(), nn.Linear(in_features, hidden_dim), nn.ReLU()
        )
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_log_var = nn.Linear(hidden_dim, latent_dim)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, in_features),
            nn.Sigmoid(),
        )
        # END SOLUTION

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parámetros de q(z|x).

        Args:
            x: imágenes. Dimensions: [batch, channels, height, width].

        Returns:
            ``mu`` y ``log_var`` (log σ²). Dimensions: [batch, latent_dim] cada uno.
        """

        # BEGIN SOLUTION
        h = self.encoder(x)
        return self.fc_mu(h), self.fc_log_var(h)
        # END SOLUTION

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

        # BEGIN SOLUTION
        if not self.training:
            return mu
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + std * eps
        # END SOLUTION

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decodifica ``z`` (media de p(x|z)).

        Args:
            z: códigos latentes. Dimensions: [batch, latent_dim].

        Returns:
            Imágenes en [0, 1]. Dimensions: [batch, channels, height, width].
        """

        # BEGIN SOLUTION
        return self.decoder(z).view(-1, *self.image_shape)
        # END SOLUTION

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

        # BEGIN SOLUTION
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var
        # END SOLUTION

    @torch.no_grad()
    def sample(self, n: int) -> torch.Tensor:
        """
        Genera ``n`` imágenes nuevas muestreando el prior ``z ~ N(0, I)``.

        Args:
            n: número de muestras.

        Returns:
            Imágenes generadas. Dimensions: [n, channels, height, width].
        """

        # BEGIN SOLUTION
        device = next(self.parameters()).device
        z = torch.randn(n, self.latent_dim, device=device)
        return self.decode(z)
        # END SOLUTION


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

    # BEGIN SOLUTION
    kl = -0.5 * (1 + log_var - mu.pow(2) - log_var.exp())
    return kl.sum(dim=1).mean()
    # END SOLUTION


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

    # BEGIN SOLUTION
    batch = x.shape[0]
    recon_loss = F.binary_cross_entropy(recon, x, reduction="sum") / batch
    kl = kl_divergence(mu, log_var)
    return recon_loss, kl, recon_loss + beta * kl
    # END SOLUTION


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

    # BEGIN SOLUTION
    mu1, _ = model.encode(x1)
    mu2, _ = model.encode(x2)
    t = torch.linspace(0, 1, steps, device=mu1.device).unsqueeze(1)
    z = (1 - t) * mu1 + t * mu2
    return model.decode(z)
    # END SOLUTION


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

    # BEGIN SOLUTION
    model.train()
    recon, mu, log_var = model(x)
    recon_loss, kl, total = vae_loss(recon, x, mu, log_var, beta)
    optimizer.zero_grad()
    total.backward()
    optimizer.step()
    return recon_loss.item(), kl.item(), total.item()
    # END SOLUTION
