"""
CV07 · VQ-VAE: cuantización vectorial con *straight-through* — soluciones.
"""

import torch
import torch.nn.functional as F
from torch import nn


def nearest_codebook_indices(z: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """
    Índice del vector del codebook más cercano (distancia euclídea) a cada fila.

    Args:
        z: vectores a cuantizar. Dimensions: [n, embedding_dim].
        codebook: vectores del codebook. Dimensions: [num_embeddings, embedding_dim].

    Returns:
        Índices ``int64``. Dimensions: [n].
    """

    # BEGIN SOLUTION usa torch.cdist + argmin
    return torch.cdist(z, codebook).argmin(dim=1)
    # END SOLUTION


def vq_losses(z_e: torch.Tensor, z_q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pérdidas de VQ-VAE con *stop-gradient* (``detach``).

    * ``codebook_loss = ||sg[z_e] - z_q||²`` (solo mueve el codebook).
    * ``commitment_loss = ||z_e - sg[z_q]||²`` (solo mueve el encoder).

    Ambas con ``F.mse_loss`` (media).

    Args:
        z_e: salida del encoder. Dimensions: [batch, embedding_dim, height, width].
        z_q: vectores del codebook elegidos (con gradiente al codebook). Dimensions:
            igual que ``z_e``.

    Returns:
        ``(codebook_loss, commitment_loss)``, escalares.
    """

    # BEGIN SOLUTION
    codebook_loss = F.mse_loss(z_q, z_e.detach())
    commitment_loss = F.mse_loss(z_e, z_q.detach())
    return codebook_loss, commitment_loss
    # END SOLUTION


class VectorQuantizer(nn.Module):
    """
    Capa de cuantización vectorial de VQ-VAE.

    Args:
        num_embeddings: tamaño ``K`` del codebook.
        embedding_dim: dimensión ``D`` de cada código (= canales de ``z_e``).
        beta: peso de la *commitment loss* (0.25 en el paper).
    """

    def __init__(self, num_embeddings: int, embedding_dim: int, beta: float = 0.25) -> None:
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.beta = beta
        self.codebook = nn.Embedding(num_embeddings, embedding_dim)
        nn.init.uniform_(self.codebook.weight, -1 / num_embeddings, 1 / num_embeddings)

    def forward(
        self, z_e: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Cuantiza cada vector espacial de ``z_e`` a su código más cercano.

        Pasos: ``[B, D, H, W] → [B·H·W, D]`` → índices más cercanos → ``z_q`` →
        volver a ``[B, D, H, W]`` → pérdida ``codebook + beta * commitment`` →
        *straight-through*: ``z_q_st = z_e + (z_q - z_e).detach()``.

        Args:
            z_e: salida del encoder. Dimensions: [batch, embedding_dim, height, width].

        Returns:
            ``z_q_st`` (mismo valor que ``z_q`` pero gradiente identidad hacia
            ``z_e``) ``[batch, embedding_dim, height, width]``, la pérdida VQ
            (escalar) y los índices ``[batch, height, width]``.
        """

        # BEGIN SOLUTION
        b, d, h, w = z_e.shape
        flat = z_e.permute(0, 2, 3, 1).reshape(-1, d)
        indices = nearest_codebook_indices(flat, self.codebook.weight)
        z_q = self.codebook(indices).view(b, h, w, d).permute(0, 3, 1, 2)
        codebook_loss, commitment_loss = vq_losses(z_e, z_q)
        loss = codebook_loss + self.beta * commitment_loss
        z_q_st = z_e + (z_q - z_e).detach()
        return z_q_st, loss, indices.view(b, h, w)
        # END SOLUTION
