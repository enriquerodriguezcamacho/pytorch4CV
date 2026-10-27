"""
Tests de CV07 · GAN (DCGAN), pérdidas y WGAN-GP.
"""

import math

import pytest
import torch
import torch.nn.functional as F
from helpers import load
from torch import nn

m = load("cv/07_generative_models/gan.py")

LATENT_DIM = 8
HIDDEN = 8
BATCH_SIZE = 4
LN2 = math.log(2.0)


@pytest.mark.parametrize("batch", [1, 4])
def test_generator(batch: int) -> None:
    """G: [batch, latent] → [batch, 1, 28, 28] en [-1, 1] (Tanh)."""

    gen = m.Generator(LATENT_DIM, HIDDEN).eval()
    out = gen(torch.randn(batch, LATENT_DIM))
    assert out.shape == (batch, 1, 28, 28), f"Forma {out.shape}"
    assert out.min() >= -1 and out.max() <= 1, "La salida de G debe ir en [-1, 1]."
    assert any(isinstance(mod, nn.Tanh) for mod in gen.modules()), "Usa Tanh al final."
    assert any(isinstance(mod, nn.BatchNorm2d) for mod in gen.modules()), "G lleva BN."


def test_discriminator() -> None:
    """D: logits [batch, 1], LeakyReLU(0.2) y sin pooling."""

    disc = m.Discriminator(1, HIDDEN)
    out = disc(torch.randn(BATCH_SIZE, 1, 28, 28) * 10)
    assert out.shape == (BATCH_SIZE, 1), f"Forma {out.shape}"
    assert not (out.min() >= 0 and out.max() <= 1), "D debe devolver logits."
    leaky = [mod for mod in disc.modules() if isinstance(mod, nn.LeakyReLU)]
    assert leaky and all(mod.negative_slope == 0.2 for mod in leaky), "LeakyReLU(0.2)."
    assert not any(
        isinstance(mod, (nn.MaxPool2d, nn.AvgPool2d)) for mod in disc.modules()
    ), "DCGAN no usa pooling: usa convoluciones con stride."


def test_weights_init() -> None:
    """Pesos conv ~ N(0, 0.02) y BN ~ N(1, 0.02) con bias 0."""

    gen = m.Generator(LATENT_DIM, 32)
    gen.apply(m.weights_init)
    convs = [mod for mod in gen.modules() if isinstance(mod, nn.ConvTranspose2d)]
    w = torch.cat([c.weight.flatten() for c in convs])
    assert abs(w.mean().item()) < 0.005, "Media de los pesos ≈ 0."
    assert abs(w.std().item() - 0.02) < 0.003, "Desviación ≈ 0.02."
    bns = [mod for mod in gen.modules() if isinstance(mod, nn.BatchNorm2d)]
    assert all(torch.all(bn.bias == 0) for bn in bns), "Bias de BN a 0."
    assert all(abs(bn.weight.mean().item() - 1) < 0.02 for bn in bns), "Pesos BN ≈ 1."


def test_discriminator_loss_hand_values() -> None:
    """Con logits 0: ln2 + ln2; con D perfecto la pérdida ≈ 0."""

    zeros = torch.zeros(BATCH_SIZE, 1)
    loss = m.discriminator_loss(zeros, zeros)
    assert loss.dim() == 0, "Debe ser un escalar."
    assert torch.isclose(loss, torch.tensor(2 * LN2)), f"Esperado 2·ln2, obtenido {loss}"
    good = m.discriminator_loss(torch.full((4, 1), 10.0), torch.full((4, 1), -10.0))
    assert good < 1e-3, "Si D acierta con confianza la pérdida es ~0."
    real, fake = torch.randn(4, 1), torch.randn(4, 1)
    ref = F.binary_cross_entropy(torch.sigmoid(real), torch.ones(4, 1)) + (
        F.binary_cross_entropy(torch.sigmoid(fake), torch.zeros(4, 1))
    )
    assert torch.isclose(m.discriminator_loss(real, fake), ref, atol=1e-5), "BCE real+fake."


def test_generator_losses() -> None:
    """No saturante = -log D(G(z)); saturante = log(1 - D(G(z)))."""

    zeros = torch.zeros(BATCH_SIZE, 1)
    assert torch.isclose(m.generator_loss(zeros), torch.tensor(LN2)), "ln2 en logits 0."
    assert torch.isclose(m.generator_loss_saturating(zeros), torch.tensor(-LN2)), (
        "log(1 - 0.5) = -ln2."
    )
    logits = torch.tensor([[1.5], [-0.3]])
    d = torch.sigmoid(logits)
    assert torch.isclose(m.generator_loss(logits), -d.log().mean()), "-mean(log D)."
    assert torch.isclose(m.generator_loss_saturating(logits), (1 - d).log().mean()), (
        "mean(log(1 - D))."
    )


def test_non_saturating_has_stronger_gradient_early() -> None:
    """Cuando D rechaza bien las falsas (logit muy negativo), la versión
    no saturante da un gradiente mucho mayor que la minimax original."""

    logit_ns = torch.full((1, 1), -6.0, requires_grad=True)
    m.generator_loss(logit_ns).backward()
    logit_sat = torch.full((1, 1), -6.0, requires_grad=True)
    m.generator_loss_saturating(logit_sat).backward()
    assert logit_ns.grad.abs() > 50 * logit_sat.grad.abs(), (
        "La pérdida no saturante evita el desvanecimiento del gradiente."
    )


def test_gan_train_step_updates_both() -> None:
    """Ambas redes cambian y las pérdidas son floats finitos."""

    gen, disc = m.Generator(LATENT_DIM, HIDDEN), m.Discriminator(1, HIDDEN)
    g_opt = torch.optim.Adam(gen.parameters(), lr=2e-4, betas=(0.5, 0.999))
    d_opt = torch.optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.999))
    real = torch.rand(BATCH_SIZE, 1, 28, 28) * 2 - 1
    g_before = [p.clone() for p in gen.parameters()]
    d_before = [p.clone() for p in disc.parameters()]
    d_loss, g_loss = m.gan_train_step(gen, disc, real, g_opt, d_opt, LATENT_DIM)
    assert isinstance(d_loss, float) and isinstance(g_loss, float), "Devuelve floats."
    assert math.isfinite(d_loss) and math.isfinite(g_loss), "Pérdidas finitas."
    assert d_loss >= 0 and g_loss >= 0, "Las BCE son no negativas."
    assert any(torch.norm(p - q) > 0 for p, q in zip(gen.parameters(), g_before)), (
        "G debe actualizarse."
    )
    assert any(torch.norm(p - q) > 0 for p, q in zip(disc.parameters(), d_before)), (
        "D debe actualizarse."
    )
    assert gen.training and disc.training, "Ambas redes en modo train."


def test_gan_train_step_detach() -> None:
    """Con detach en el paso de D, el gradiente solo atraviesa G una vez."""

    gen, disc = m.Generator(LATENT_DIM, HIDDEN), m.Discriminator(1, HIDDEN)
    g_opt = torch.optim.SGD(gen.parameters(), lr=0.01)
    d_opt = torch.optim.SGD(disc.parameters(), lr=0.01)
    calls = []
    first_param = next(gen.parameters())
    handle = first_param.register_hook(lambda g: calls.append(1))
    m.gan_train_step(gen, disc, torch.randn(BATCH_SIZE, 1, 28, 28), g_opt, d_opt, LATENT_DIM)
    handle.remove()
    assert len(calls) == 1, (
        f"El gradiente pasó {len(calls)} veces por G: usa fake.detach() al entrenar D."
    )


def test_gan_train_step_learning_rate_zero() -> None:
    """Con lr=0 en G solo cambia D (y viceversa): cada optimizador mueve su red."""

    gen, disc = m.Generator(LATENT_DIM, HIDDEN), m.Discriminator(1, HIDDEN)
    g_opt = torch.optim.SGD(gen.parameters(), lr=0.0)
    d_opt = torch.optim.SGD(disc.parameters(), lr=0.1)
    g_before = [p.clone() for p in gen.parameters()]
    m.gan_train_step(gen, disc, torch.randn(BATCH_SIZE, 1, 28, 28), g_opt, d_opt, LATENT_DIM)
    assert all(torch.equal(p, q) for p, q in zip(gen.parameters(), g_before)), (
        "G no debe cambiar si su lr es 0 (¿usas el optimizador correcto?)."
    )


def test_wasserstein_losses() -> None:
    """Crítico: E[c(fake)] - E[c(real)]; generador: -E[c(fake)]."""

    real = torch.tensor([[1.0], [3.0]])
    fake = torch.tensor([[-1.0], [0.0]])
    assert torch.isclose(m.wasserstein_critic_loss(real, fake), torch.tensor(-2.5)), (
        "-0.5 - 2 = -2.5"
    )
    assert torch.isclose(m.wasserstein_generator_loss(fake), torch.tensor(0.5)), "0.5"


def test_clip_weights() -> None:
    """Todos los parámetros quedan en [-c, c] y in-place."""

    critic = m.Discriminator(1, HIDDEN)
    with torch.no_grad():
        for p in critic.parameters():
            p.mul_(100)
    ref = next(critic.parameters())
    m.clip_weights(critic, 0.01)
    assert ref is next(critic.parameters()), "Recorta in-place (no crees parámetros nuevos)."
    assert all(p.abs().max() <= 0.01 + 1e-8 for p in critic.parameters()), "Fuera de rango."


def _linear_critic(weight_norm: float) -> nn.Module:
    critic = nn.Sequential(nn.Flatten(), nn.Linear(4, 1))
    with torch.no_grad():
        critic[1].weight.copy_(torch.tensor([[1.0, 1.0, 1.0, 1.0]]) / 2 * weight_norm)
    return critic


@pytest.mark.parametrize(("weight_norm", "expected"), [(1.0, 0.0), (2.0, 1.0), (3.0, 4.0)])
def test_gradient_penalty_linear_critic(weight_norm: float, expected: float) -> None:
    """Para c(x) = wᵀx el gradiente es w en todo punto: GP = (||w|| - 1)²."""

    critic = _linear_critic(weight_norm)
    real, fake = torch.randn(BATCH_SIZE, 1, 2, 2), torch.randn(BATCH_SIZE, 1, 2, 2)
    gp = m.gradient_penalty(critic, real, fake)
    assert gp.dim() == 0, "La penalización es un escalar."
    assert torch.isclose(gp, torch.tensor(expected), atol=1e-5), f"GP={gp.item()}"


def test_gradient_penalty_create_graph() -> None:
    """La GP debe ser derivable respecto a los pesos del crítico (create_graph)."""

    critic = _linear_critic(2.0)
    real, fake = torch.randn(BATCH_SIZE, 1, 2, 2), torch.randn(BATCH_SIZE, 1, 2, 2)
    gp = m.gradient_penalty(critic, real, fake)
    assert gp.requires_grad, "Usa create_graph=True en torch.autograd.grad."
    gp.backward()
    w_grad = critic[1].weight.grad
    # d/dw (||w|| - 1)² = 2 (||w|| - 1) w / ||w||  → con w = [1,1,1,1] (||w|| = 2): w = 1
    assert w_grad is not None and torch.allclose(w_grad, torch.ones(1, 4)), (
        f"Gradiente de la GP respecto a w incorrecto: {w_grad}"
    )
    assert not real.requires_grad, "No modifiques los tensores de entrada."
