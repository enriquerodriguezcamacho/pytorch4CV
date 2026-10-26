"""
CV07 · Redes neuronales bayesianas: MC Dropout y Bayes by Backprop — soluciones.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn


def enable_dropout(model: nn.Module) -> None:
    """
    Pone en modo ``train`` **solo** las capas de dropout (``nn.Dropout``,
    ``nn.Dropout2d``...). El resto (p. ej. BatchNorm) se queda como esté.

    Args:
        model: red a modificar in-place.
    """

    # BEGIN SOLUTION
    for module in model.modules():
        if isinstance(module, nn.modules.dropout._DropoutNd):
            module.train()
    # END SOLUTION


@torch.no_grad()
def mc_dropout_predict(
    model: nn.Module, x: torch.Tensor, n_samples: int = 20
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Predicción con MC Dropout (aproximación bayesiana de Gal & Ghahramani).

    Pone el modelo en ``eval``, activa solo el dropout, hace ``n_samples``
    pasadas, aplica ``softmax`` a cada una y devuelve media y varianza (sin
    corrección, ``unbiased=False``) de las probabilidades. Al terminar, **restaura**
    el modo original (``model.training``) del modelo.

    Args:
        model: clasificador que devuelve logits ``[batch, num_classes]``.
        x: entrada. Dimensions: [batch, ...].
        n_samples: número ``T`` de pasadas estocásticas.

    Returns:
        ``(mean, var)`` de las probabilidades. Dimensions: [batch, num_classes].
    """

    # BEGIN SOLUTION
    was_training = model.training
    model.eval()
    enable_dropout(model)
    probs = torch.stack([F.softmax(model(x), dim=-1) for _ in range(n_samples)])
    model.train(was_training)
    return probs.mean(dim=0), probs.var(dim=0, unbiased=False)
    # END SOLUTION


def predictive_entropy(probs: torch.Tensor) -> torch.Tensor:
    """
    Entropía de la predicción media ``H = -Σ_c p_c log p_c`` (incertidumbre total).

    Usa ``clamp_min(1e-12)`` dentro del logaritmo para evitar ``log(0)``.

    Args:
        probs: probabilidades. Dimensions: [batch, num_classes].

    Returns:
        Entropía (nats). Dimensions: [batch].
    """

    # BEGIN SOLUTION
    return -(probs * probs.clamp_min(1e-12).log()).sum(dim=-1)
    # END SOLUTION


class BayesLinear(nn.Module):
    """
    Capa lineal bayesiana (Bayes by Backprop, Blundell et al. 2015).

    Cada peso tiene una posterior ``N(mu, sigma²)`` con ``sigma = softplus(rho)``
    y prior ``N(0, prior_sigma²)``. En cada forward se muestrea
    ``w = mu + sigma * eps`` (reparametrización), también para el bias.

    Args:
        in_features: entradas.
        out_features: salidas.
        prior_sigma: desviación típica del prior.
    """

    def __init__(self, in_features: int, out_features: int, prior_sigma: float = 1.0) -> None:
        super().__init__()

        self.prior_sigma = prior_sigma
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features).uniform_(-0.1, 0.1))
        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), -5.0))
        self.bias_mu = nn.Parameter(torch.zeros(out_features))
        self.bias_rho = nn.Parameter(torch.full((out_features,), -5.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward con pesos muestreados.

        Args:
            x: entrada. Dimensions: [batch, in_features].

        Returns:
            Salida. Dimensions: [batch, out_features].
        """

        # BEGIN SOLUTION
        w_sigma = F.softplus(self.weight_rho)
        b_sigma = F.softplus(self.bias_rho)
        w = self.weight_mu + w_sigma * torch.randn_like(w_sigma)
        b = self.bias_mu + b_sigma * torch.randn_like(b_sigma)
        return F.linear(x, w, b)
        # END SOLUTION

    def kl_divergence(self) -> torch.Tensor:
        """
        KL(q || p) total (pesos + bias) entre gaussianas diagonales:
        ``Σ [log(σ_p / σ) + (σ² + μ²) / (2 σ_p²) - 1/2]``.

        Returns:
            Escalar.
        """

        # BEGIN SOLUTION
        total = torch.zeros((), device=self.weight_mu.device)
        for mu, rho in ((self.weight_mu, self.weight_rho), (self.bias_mu, self.bias_rho)):
            sigma = F.softplus(rho)
            kl = (
                math.log(self.prior_sigma)
                - sigma.log()
                + (sigma.pow(2) + mu.pow(2)) / (2 * self.prior_sigma**2)
                - 0.5
            )
            total = total + kl.sum()
        return total
        # END SOLUTION
