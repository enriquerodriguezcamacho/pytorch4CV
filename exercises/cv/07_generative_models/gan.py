"""
CV07 · GAN (DCGAN pequeña), pérdidas, paso de entrenamiento y WGAN-GP — ejercicios.
"""

import torch
import torch.nn.functional as F
from torch import nn


class Generator(nn.Module):
    """
    Generador DCGAN para imágenes 28x28 con valores en [-1, 1].

    Arquitectura (sigue las *guidelines* de DCGAN: sin pooling ni capas densas
    ocultas, BatchNorm + ReLU en el generador y Tanh a la salida):
        - ``z`` se ve como ``[batch, latent_dim, 1, 1]``.
        - ``ConvTranspose2d(latent_dim, hidden * 2, 7, 1, 0)`` + BN + ReLU → 7x7.
        - ``ConvTranspose2d(hidden * 2, hidden, 4, 2, 1)`` + BN + ReLU → 14x14.
        - ``ConvTranspose2d(hidden, out_channels, 4, 2, 1)`` + Tanh → 28x28.

    Args:
        latent_dim: dimensión del ruido ``z``.
        hidden: canales base.
        out_channels: canales de la imagen generada.
    """

    def __init__(self, latent_dim: int = 16, hidden: int = 16, out_channels: int = 1) -> None:
        super().__init__()

        self.latent_dim = latent_dim

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            z: ruido latente. Dimensions: [batch, latent_dim].

        Returns:
            Imágenes en [-1, 1]. Dimensions: [batch, out_channels, 28, 28].
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")


class Discriminator(nn.Module):
    """
    Discriminador DCGAN para imágenes 28x28. Devuelve **logits** (sin sigmoide).

    Arquitectura (LeakyReLU(0.2), convoluciones con stride en vez de pooling y sin
    BatchNorm en la primera capa):
        - ``Conv2d(in_channels, hidden, 4, 2, 1)`` + LeakyReLU(0.2) → 14x14.
        - ``Conv2d(hidden, hidden * 2, 4, 2, 1)`` + BN + LeakyReLU(0.2) → 7x7.
        - ``Conv2d(hidden * 2, 1, 7, 1, 0)`` → 1x1 → aplanar a ``[batch, 1]``.

    Args:
        in_channels: canales de la imagen.
        hidden: canales base.
    """

    def __init__(self, in_channels: int = 1, hidden: int = 16) -> None:
        super().__init__()

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: imágenes. Dimensions: [batch, in_channels, 28, 28].

        Returns:
            Logits real/falso. Dimensions: [batch, 1].
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")


def weights_init(module: nn.Module) -> None:
    """
    Inicialización DCGAN (usar con ``model.apply(weights_init)``).

    * ``Conv2d`` y ``ConvTranspose2d``: pesos ~ N(0, 0.02).
    * ``BatchNorm2d``: pesos ~ N(1, 0.02) y bias = 0.

    Args:
        module: submódulo que recibe ``nn.Module.apply``.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def discriminator_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    """
    Pérdida del discriminador: ``BCE(real, 1) + BCE(fake, 0)`` (con logits).

    Args:
        real_logits: salida de D sobre imágenes reales. Dimensions: [batch, 1].
        fake_logits: salida de D sobre imágenes generadas. Dimensions: [batch, 1].

    Returns:
        Escalar (suma de las dos BCE medias).
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def generator_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    """
    Pérdida **no saturante** del generador: ``BCE(D(G(z)), 1) = -log D(G(z))``.

    Args:
        fake_logits: salida de D sobre imágenes generadas. Dimensions: [batch, 1].

    Returns:
        Escalar.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def generator_loss_saturating(fake_logits: torch.Tensor) -> torch.Tensor:
    """
    Pérdida **minimax original** (saturante) del generador: ``mean(log(1 - D(G(z))))``.

    Se minimiza; equivale a ``-BCE(D(G(z)), 0)``. Úsala solo para comparar
    gradientes con la versión no saturante.

    Args:
        fake_logits: salida de D sobre imágenes generadas. Dimensions: [batch, 1].

    Returns:
        Escalar (≤ 0).
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def gan_train_step(
    generator: nn.Module,
    discriminator: nn.Module,
    real: torch.Tensor,
    g_optimizer: torch.optim.Optimizer,
    d_optimizer: torch.optim.Optimizer,
    latent_dim: int,
) -> tuple[float, float]:
    """
    Un paso de entrenamiento alternado de la GAN.

    1. **Discriminador**: genera ``fake = G(z)`` y calcula
       ``discriminator_loss(D(real), D(fake.detach()))`` → backward → ``d_optimizer``.
       (El ``detach`` evita propagar gradientes al generador en este paso.)
    2. **Generador**: ``generator_loss(D(fake))`` → backward → ``g_optimizer``.

    Args:
        generator: generador (modo ``train``).
        discriminator: discriminador (modo ``train``).
        real: imágenes reales. Dimensions: [batch, channels, height, width].
        g_optimizer: optimizador del generador.
        d_optimizer: optimizador del discriminador.
        latent_dim: dimensión de ``z``.

    Returns:
        ``(d_loss, g_loss)`` como ``float``.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def wasserstein_critic_loss(real_scores: torch.Tensor, fake_scores: torch.Tensor) -> torch.Tensor:
    """
    Pérdida del crítico de WGAN (a minimizar): ``mean(c(fake)) - mean(c(real))``.

    Args:
        real_scores: puntuaciones del crítico sobre reales. Dimensions: [batch, 1].
        fake_scores: puntuaciones del crítico sobre falsas. Dimensions: [batch, 1].

    Returns:
        Escalar (puede ser negativo).
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def wasserstein_generator_loss(fake_scores: torch.Tensor) -> torch.Tensor:
    """
    Pérdida del generador de WGAN: ``-mean(c(G(z)))``.

    Args:
        fake_scores: puntuaciones del crítico sobre falsas. Dimensions: [batch, 1].

    Returns:
        Escalar.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


@torch.no_grad()
def clip_weights(critic: nn.Module, clip_value: float) -> None:
    """
    *Weight clipping* de la WGAN original: recorta **in-place** todos los
    parámetros del crítico a ``[-clip_value, clip_value]``.

    Args:
        critic: crítico.
        clip_value: valor ``c`` del recorte.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


def gradient_penalty(critic: nn.Module, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    """
    Penalización de gradiente de WGAN-GP: ``E[(||∇_x̂ c(x̂)||₂ - 1)²]``.

    ``x̂ = eps * real + (1 - eps) * fake`` con un ``eps ~ U[0, 1]`` **por muestra**.
    Usa ``torch.autograd.grad(..., create_graph=True)`` para que la penalización
    sea derivable respecto a los parámetros del crítico.

    Args:
        critic: crítico (devuelve ``[batch, 1]``).
        real: imágenes reales. Dimensions: [batch, channels, height, width].
        fake: imágenes generadas (ya sin gradiente hacia G). Dimensions: igual que real.

    Returns:
        Escalar.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")
