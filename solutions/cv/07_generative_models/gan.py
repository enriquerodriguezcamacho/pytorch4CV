"""
CV07 · GAN (DCGAN pequeña), pérdidas, paso de entrenamiento y WGAN-GP — soluciones.
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

        # BEGIN SOLUTION
        self.net = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, hidden * 2, 7, 1, 0, bias=False),
            nn.BatchNorm2d(hidden * 2),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden * 2, hidden, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden, out_channels, 4, 2, 1),
            nn.Tanh(),
        )
        # END SOLUTION

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            z: ruido latente. Dimensions: [batch, latent_dim].

        Returns:
            Imágenes en [-1, 1]. Dimensions: [batch, out_channels, 28, 28].
        """

        # BEGIN SOLUTION
        return self.net(z.view(z.shape[0], self.latent_dim, 1, 1))
        # END SOLUTION


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

        # BEGIN SOLUTION
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden, hidden * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(hidden * 2),
            nn.LeakyReLU(0.2),
            nn.Conv2d(hidden * 2, 1, 7, 1, 0),
        )
        # END SOLUTION

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: imágenes. Dimensions: [batch, in_channels, 28, 28].

        Returns:
            Logits real/falso. Dimensions: [batch, 1].
        """

        # BEGIN SOLUTION
        return self.net(x).view(x.shape[0], 1)
        # END SOLUTION


def weights_init(module: nn.Module) -> None:
    """
    Inicialización DCGAN (usar con ``model.apply(weights_init)``).

    * ``Conv2d`` y ``ConvTranspose2d``: pesos ~ N(0, 0.02).
    * ``BatchNorm2d``: pesos ~ N(1, 0.02) y bias = 0.

    Args:
        module: submódulo que recibe ``nn.Module.apply``.
    """

    # BEGIN SOLUTION
    if isinstance(module, (nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.normal_(module.weight, 0.0, 0.02)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.zeros_(module.bias)
    # END SOLUTION


def discriminator_loss(real_logits: torch.Tensor, fake_logits: torch.Tensor) -> torch.Tensor:
    """
    Pérdida del discriminador: ``BCE(real, 1) + BCE(fake, 0)`` (con logits).

    Args:
        real_logits: salida de D sobre imágenes reales. Dimensions: [batch, 1].
        fake_logits: salida de D sobre imágenes generadas. Dimensions: [batch, 1].

    Returns:
        Escalar (suma de las dos BCE medias).
    """

    # BEGIN SOLUTION
    real = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
    fake = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
    return real + fake
    # END SOLUTION


def generator_loss(fake_logits: torch.Tensor) -> torch.Tensor:
    """
    Pérdida **no saturante** del generador: ``BCE(D(G(z)), 1) = -log D(G(z))``.

    Args:
        fake_logits: salida de D sobre imágenes generadas. Dimensions: [batch, 1].

    Returns:
        Escalar.
    """

    # BEGIN SOLUTION
    return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))
    # END SOLUTION


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

    # BEGIN SOLUTION
    return -F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
    # END SOLUTION


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

    # BEGIN SOLUTION
    generator.train()
    discriminator.train()
    z = torch.randn(real.shape[0], latent_dim, device=real.device)
    fake = generator(z)

    d_optimizer.zero_grad()
    d_loss = discriminator_loss(discriminator(real), discriminator(fake.detach()))
    d_loss.backward()
    d_optimizer.step()

    g_optimizer.zero_grad()
    g_loss = generator_loss(discriminator(fake))
    g_loss.backward()
    g_optimizer.step()
    return d_loss.item(), g_loss.item()
    # END SOLUTION


def wasserstein_critic_loss(real_scores: torch.Tensor, fake_scores: torch.Tensor) -> torch.Tensor:
    """
    Pérdida del crítico de WGAN (a minimizar): ``mean(c(fake)) - mean(c(real))``.

    Args:
        real_scores: puntuaciones del crítico sobre reales. Dimensions: [batch, 1].
        fake_scores: puntuaciones del crítico sobre falsas. Dimensions: [batch, 1].

    Returns:
        Escalar (puede ser negativo).
    """

    # BEGIN SOLUTION
    return fake_scores.mean() - real_scores.mean()
    # END SOLUTION


def wasserstein_generator_loss(fake_scores: torch.Tensor) -> torch.Tensor:
    """
    Pérdida del generador de WGAN: ``-mean(c(G(z)))``.

    Args:
        fake_scores: puntuaciones del crítico sobre falsas. Dimensions: [batch, 1].

    Returns:
        Escalar.
    """

    # BEGIN SOLUTION
    return -fake_scores.mean()
    # END SOLUTION


@torch.no_grad()
def clip_weights(critic: nn.Module, clip_value: float) -> None:
    """
    *Weight clipping* de la WGAN original: recorta **in-place** todos los
    parámetros del crítico a ``[-clip_value, clip_value]``.

    Args:
        critic: crítico.
        clip_value: valor ``c`` del recorte.
    """

    # BEGIN SOLUTION
    for p in critic.parameters():
        p.clamp_(-clip_value, clip_value)
    # END SOLUTION


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

    # BEGIN SOLUTION
    batch = real.shape[0]
    eps = torch.rand(batch, *([1] * (real.dim() - 1)), device=real.device)
    x_hat = (eps * real + (1 - eps) * fake).requires_grad_(True)
    scores = critic(x_hat)
    (grads,) = torch.autograd.grad(
        outputs=scores,
        inputs=x_hat,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
    )
    norms = grads.reshape(batch, -1).norm(2, dim=1)
    return ((norms - 1) ** 2).mean()
    # END SOLUTION
