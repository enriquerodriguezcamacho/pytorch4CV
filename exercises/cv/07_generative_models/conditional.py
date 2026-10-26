"""
CV07 · Generación condicional: CGAN y CVAE — ejercicios.
"""

import torch
import torch.nn.functional as F
from torch import nn


def one_hot_label_maps(
    labels: torch.Tensor, num_classes: int, height: int, width: int
) -> torch.Tensor:
    """
    Convierte etiquetas en mapas one-hot espaciales (canales extra para D o CVAE).

    El canal ``labels[b]`` del ejemplo ``b`` vale 1 en todos los píxeles y el
    resto vale 0.

    Args:
        labels: clases. Dimensions: [batch] (``int64``).
        num_classes: número de clases ``K``.
        height: alto del mapa.
        width: ancho del mapa.

    Returns:
        Mapas ``float32``. Dimensions: [batch, num_classes, height, width].
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


class ConditionalGenerator(nn.Module):
    """
    Generador condicional (CGAN) para imágenes 28x28 en [-1, 1].

    Arquitectura:
        - ``label_emb = nn.Embedding(num_classes, embed_dim)``; se concatena con el
          ruido: entrada de tamaño ``latent_dim + embed_dim``.
        - ``Linear(latent_dim + embed_dim, hidden * 2 * 7 * 7)`` → reshape a
          ``[batch, hidden * 2, 7, 7]`` → BN → ReLU.
        - ``ConvTranspose2d(hidden * 2, hidden, 4, 2, 1)`` → BN → ReLU (14x14).
        - ``ConvTranspose2d(hidden, 1, 4, 2, 1)`` → Tanh (28x28).

    Args:
        latent_dim: dimensión del ruido.
        num_classes: número de clases.
        embed_dim: dimensión del embedding de la etiqueta.
        hidden: canales base.
    """

    def __init__(
        self, latent_dim: int = 16, num_classes: int = 10, embed_dim: int = 8, hidden: int = 16
    ) -> None:
        super().__init__()

        self.hidden = hidden

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def forward(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            z: ruido. Dimensions: [batch, latent_dim].
            labels: clases. Dimensions: [batch].

        Returns:
            Imágenes en [-1, 1]. Dimensions: [batch, 1, 28, 28].
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")


class ConditionalDiscriminator(nn.Module):
    """
    Discriminador condicional: la etiqueta entra como ``num_classes`` mapas one-hot
    concatenados a la imagen (entrada de ``1 + num_classes`` canales).

    Arquitectura:
        - ``Conv2d(1 + num_classes, hidden, 4, 2, 1)`` + LeakyReLU(0.2) → 14x14.
        - ``Conv2d(hidden, hidden * 2, 4, 2, 1)`` + BN + LeakyReLU(0.2) → 7x7.
        - ``Flatten`` + ``Linear(hidden * 2 * 7 * 7, 1)`` → logit.

    Args:
        num_classes: número de clases.
        hidden: canales base.
    """

    def __init__(self, num_classes: int = 10, hidden: int = 16) -> None:
        super().__init__()

        self.num_classes = num_classes

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def forward(self, x: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: imágenes. Dimensions: [batch, 1, 28, 28].
            labels: clases. Dimensions: [batch].

        Returns:
            Logits. Dimensions: [batch, 1].
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")


def cgan_train_step(
    generator: ConditionalGenerator,
    discriminator: ConditionalDiscriminator,
    real: torch.Tensor,
    labels: torch.Tensor,
    g_optimizer: torch.optim.Optimizer,
    d_optimizer: torch.optim.Optimizer,
    latent_dim: int,
) -> tuple[float, float]:
    """
    Un paso de entrenamiento de la CGAN con ``BCEWithLogits``.

    Las imágenes falsas se generan con **las mismas etiquetas** que las reales.
    D: ``BCE(D(real, y), 1) + BCE(D(fake.detach(), y), 0)``.
    G: ``BCE(D(fake, y), 1)`` (no saturante).

    Args:
        generator: generador condicional.
        discriminator: discriminador condicional.
        real: imágenes reales en [-1, 1]. Dimensions: [batch, 1, 28, 28].
        labels: clases de ``real``. Dimensions: [batch].
        g_optimizer: optimizador de G.
        d_optimizer: optimizador de D.
        latent_dim: dimensión del ruido.

    Returns:
        ``(d_loss, g_loss)`` como ``float``.
    """

    # TODO: implementa esta parte
    raise NotImplementedError("TODO")


class CVAE(nn.Module):
    """
    VAE condicional (MLP) para imágenes ``[batch, 1, 28, 28]``.

    La etiqueta se codifica en one-hot y se **concatena** tanto a la entrada del
    encoder (``q(z|x, y)``) como a ``z`` en el decoder (``p(x|z, y)``).

    Arquitectura:
        - Encoder: ``Linear(784 + K, hidden)`` → ReLU → ``fc_mu`` / ``fc_log_var``.
        - Decoder: ``Linear(latent_dim + K, hidden)`` → ReLU →
          ``Linear(hidden, 784)`` → Sigmoid → reshape a ``[batch, 1, 28, 28]``.

    Args:
        num_classes: número de clases ``K``.
        hidden_dim: neuronas ocultas.
        latent_dim: dimensión latente.
    """

    def __init__(self, num_classes: int = 10, hidden_dim: int = 128, latent_dim: int = 8) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.latent_dim = latent_dim

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def encode(
        self, x: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parámetros de q(z|x, y).

        Args:
            x: imágenes. Dimensions: [batch, 1, 28, 28].
            labels: clases. Dimensions: [batch].

        Returns:
            ``mu`` y ``log_var``. Dimensions: [batch, latent_dim].
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def decode(self, z: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Decodifica ``z`` condicionado a la clase.

        Args:
            z: códigos. Dimensions: [batch, latent_dim].
            labels: clases. Dimensions: [batch].

        Returns:
            Imágenes en [0, 1]. Dimensions: [batch, 1, 28, 28].
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    def forward(
        self, x: torch.Tensor, labels: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass con reparametrización (siempre muestrea, también en eval).

        Args:
            x: imágenes. Dimensions: [batch, 1, 28, 28].
            labels: clases. Dimensions: [batch].

        Returns:
            Reconstrucción ``[batch, 1, 28, 28]``, ``mu`` y ``log_var``
            ``[batch, latent_dim]``.
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")

    @torch.no_grad()
    def generate(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Genera una imagen por etiqueta muestreando ``z ~ N(0, I)``.

        Args:
            labels: clases a generar. Dimensions: [n].

        Returns:
            Imágenes. Dimensions: [n, 1, 28, 28].
        """

        # TODO: implementa esta parte
        raise NotImplementedError("TODO")
