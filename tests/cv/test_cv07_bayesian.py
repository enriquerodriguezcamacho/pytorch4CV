"""
Tests de CV07 · Redes bayesianas (MC Dropout y BayesLinear).
"""

import math

import pytest
import torch
from helpers import load
from torch import nn

m = load("cv/07_generative_models/bayesian.py")

BATCH_SIZE = 5
IN_FEATURES = 6
NUM_CLASSES = 3
N_SAMPLES = 30


def _model(p: float = 0.5) -> nn.Module:
    return nn.Sequential(
        nn.Linear(IN_FEATURES, 16),
        nn.BatchNorm1d(16),
        nn.ReLU(),
        nn.Dropout(p),
        nn.Linear(16, NUM_CLASSES),
    )


def test_enable_dropout() -> None:
    """Solo Dropout pasa a train; BatchNorm sigue en eval."""

    model = _model().eval()
    m.enable_dropout(model)
    assert model[3].training, "El dropout debe quedar en modo train."
    assert not model[1].training, "BatchNorm debe seguir en eval."
    assert not model[0].training and not model[4].training, "El resto no cambia."


def test_mc_dropout_predict_shapes_and_range() -> None:
    """Media y varianza [batch, classes]; la media es una distribución."""

    model = _model()
    model.eval()
    x = torch.randn(BATCH_SIZE, IN_FEATURES)
    mean, var = m.mc_dropout_predict(model, x, N_SAMPLES)
    assert mean.shape == var.shape == (BATCH_SIZE, NUM_CLASSES), "Formas incorrectas."
    assert torch.allclose(mean.sum(dim=1), torch.ones(BATCH_SIZE), atol=1e-5), "Softmax."
    assert torch.all(var >= 0), "La varianza es no negativa."
    assert var.sum() > 0, "Con dropout activo las pasadas deben diferir."
    assert not mean.requires_grad, "Predicción sin grafo (no_grad)."


@pytest.mark.parametrize("initial_mode", [True, False])
def test_mc_dropout_restores_mode(initial_mode: bool) -> None:
    """model.training vuelve a su estado original."""

    model = _model()
    model.train(initial_mode)
    m.mc_dropout_predict(model, torch.randn(BATCH_SIZE, IN_FEATURES), 3)
    assert model.training == initial_mode, "Restaura el modo original del modelo."
    assert model[3].training == initial_mode, "Restaura también el dropout."


def test_mc_dropout_without_dropout_is_deterministic() -> None:
    """Sin dropout (p=0) todas las pasadas coinciden: varianza 0 y media = softmax."""

    model = _model(p=0.0).eval()
    x = torch.randn(BATCH_SIZE, IN_FEATURES)
    mean, var = m.mc_dropout_predict(model, x, 5)
    assert torch.allclose(var, torch.zeros_like(var), atol=1e-10), "Varianza ≠ 0."
    assert torch.allclose(mean, torch.softmax(model(x), dim=1), atol=1e-6), "Media mal."


def test_mc_dropout_hand_variance() -> None:
    """Con un modelo que alterna dos salidas conocidas, la varianza es la poblacional."""

    class Alternating(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.calls = 0

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            self.calls += 1
            logits = torch.tensor([[0.0, 0.0]]) if self.calls % 2 else torch.tensor([[10.0, -10.0]])
            return logits.expand(x.shape[0], 2)

    model = Alternating()
    mean, var = m.mc_dropout_predict(model, torch.zeros(1, 1), 4)
    p_hi = torch.softmax(torch.tensor([10.0, -10.0]), 0)
    expected_mean = (torch.tensor([0.5, 0.5]) + p_hi) / 2
    assert torch.allclose(mean[0], expected_mean, atol=1e-6), "Media incorrecta."
    expected_var = ((p_hi - 0.5) / 2) ** 2
    assert torch.allclose(var[0], expected_var, atol=1e-6), "Usa var con unbiased=False."
    assert model.calls == 4, "Debe haber exactamente n_samples pasadas."


def test_predictive_entropy() -> None:
    """Uniforme → log K; one-hot → 0 (sin NaN)."""

    probs = torch.tensor([[1 / 3, 1 / 3, 1 / 3], [1.0, 0.0, 0.0]])
    h = m.predictive_entropy(probs)
    assert h.shape == (2,), "Una entropía por ejemplo."
    assert torch.isclose(h[0], torch.tensor(math.log(3.0))), "Uniforme → log 3."
    assert torch.isclose(h[1], torch.tensor(0.0), atol=1e-6), "One-hot → 0."
    assert not torch.isnan(h).any(), "Evita log(0)."


def test_bayes_linear_forward_is_stochastic() -> None:
    """Dos pasadas dan salidas distintas y el gradiente llega a mu y rho."""

    layer = m.BayesLinear(IN_FEATURES, 4)
    x = torch.randn(BATCH_SIZE, IN_FEATURES)
    out1, out2 = layer(x), layer(x)
    assert out1.shape == (BATCH_SIZE, 4), "Forma incorrecta."
    assert not torch.allclose(out1, out2), "Los pesos se muestrean en cada forward."
    out1.sum().backward()
    assert layer.weight_mu.grad is not None and layer.weight_rho.grad is not None, (
        "La reparametrización debe propagar gradiente a mu y rho."
    )


def test_bayes_linear_kl() -> None:
    """KL = 0 si q = prior; valor a mano con mu ≠ 0."""

    layer = m.BayesLinear(2, 1, prior_sigma=1.0)
    rho_for_sigma_one = math.log(math.e - 1)  # softplus(rho) = 1
    with torch.no_grad():
        layer.weight_mu.zero_()
        layer.bias_mu.zero_()
        layer.weight_rho.fill_(rho_for_sigma_one)
        layer.bias_rho.fill_(rho_for_sigma_one)
    assert torch.isclose(layer.kl_divergence(), torch.tensor(0.0), atol=1e-6), "KL(p||p)=0"
    with torch.no_grad():
        layer.weight_mu.fill_(2.0)
    # dos pesos con mu = 2: cada uno aporta mu² / 2 = 2
    assert torch.isclose(layer.kl_divergence(), torch.tensor(4.0), atol=1e-5), "KL ≠ 4."
