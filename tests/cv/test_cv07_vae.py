"""
Tests de CV07 · VAE y β-VAE.
"""

import math

import pytest
import torch
import torch.nn.functional as F
from helpers import assert_close, load

m = load("cv/07_generative_models/vae.py")

BATCH_SIZE = 4
LATENT_DIM = 3
HIDDEN_DIM = 32


def _model() -> torch.nn.Module:
    return m.VAE((1, 28, 28), HIDDEN_DIM, LATENT_DIM)


def test_encode_decode_shapes() -> None:
    """encode → (mu, log_var) y decode → imagen en [0, 1]."""

    model = _model()
    x = torch.rand(BATCH_SIZE, 1, 28, 28)
    mu, log_var = model.encode(x)
    assert mu.shape == (BATCH_SIZE, LATENT_DIM), f"mu con forma {mu.shape}"
    assert log_var.shape == (BATCH_SIZE, LATENT_DIM), "log_var con forma incorrecta."
    out = model.decode(mu)
    assert out.shape == x.shape, f"decode devuelve {out.shape}"
    assert out.min() >= 0 and out.max() <= 1, "El decoder termina en sigmoide."


def test_forward() -> None:
    """forward devuelve (recon, mu, log_var) y el gradiente llega a fc_log_var."""

    model = _model()
    x = torch.rand(BATCH_SIZE, 1, 28, 28)
    recon, mu, log_var = model(x)
    assert recon.shape == x.shape, "Reconstrucción con forma incorrecta."
    assert mu.shape == log_var.shape == (BATCH_SIZE, LATENT_DIM), "mu/log_var mal."
    recon.sum().backward()
    grads = [p.grad for p in model.fc_log_var.parameters()]
    assert all(g is not None and g.abs().sum() > 0 for g in grads), (
        "El gradiente debe llegar a fc_log_var (reparametrización)."
    )


def test_reparameterize_train_vs_eval() -> None:
    """En train z = mu + sigma·eps (aleatorio); en eval z = mu."""

    model = _model()
    mu = torch.tensor([[1.0, -2.0, 0.5]])
    log_var = torch.log(torch.tensor([[4.0, 1.0, 0.25]]))
    sigma = torch.tensor([[2.0, 1.0, 0.5]])
    model.train()
    torch.manual_seed(0)
    z = model.reparameterize(mu, log_var)
    torch.manual_seed(0)
    eps = torch.randn_like(mu)
    assert_close(z, mu + sigma * eps)
    zs = torch.stack([model.reparameterize(mu, log_var) for _ in range(4000)])
    assert_close(zs.mean(0), mu, atol=0.1)
    assert_close(zs.std(0), sigma, atol=0.1)
    model.eval()
    assert torch.equal(model.reparameterize(mu, log_var), mu), "En eval z = mu."


def test_sample() -> None:
    """sample genera imágenes nuevas a partir del prior."""

    model = _model()
    out = model.sample(5)
    assert out.shape == (5, 1, 28, 28), f"sample devuelve {out.shape}"
    assert not out.requires_grad, "sample no debe construir grafo (no_grad)."


def test_kl_divergence_hand_values() -> None:
    """KL = 0 para N(0, I); y valores calculados a mano."""

    zeros = torch.zeros(2, LATENT_DIM)
    assert torch.isclose(m.kl_divergence(zeros, zeros), torch.tensor(0.0)), "KL(N(0,I))=0"
    mu = torch.tensor([[1.0, 2.0, 0.0]])
    # -0.5 * sum(1 + 0 - mu² - 1) = 0.5 * (1 + 4) = 2.5
    assert torch.isclose(m.kl_divergence(mu, torch.zeros(1, 3)), torch.tensor(2.5)), (
        "KL incorrecta con media no nula."
    )
    log_var = torch.tensor([[math.log(2.0), 0.0, 0.0]])
    expected = -0.5 * (1 + math.log(2.0) - 0 - 2.0)
    assert torch.isclose(
        m.kl_divergence(torch.zeros(1, 3), log_var), torch.tensor(expected)
    ), "KL incorrecta con varianza 2."
    batch_mu = torch.tensor([[1.0, 2.0, 0.0], [0.0, 0.0, 0.0]])
    assert torch.isclose(m.kl_divergence(batch_mu, torch.zeros(2, 3)), torch.tensor(1.25)), (
        "La KL se suma en latent_dim y se promedia en el batch."
    )


def test_vae_loss_hand_values() -> None:
    """recon = BCE sumada por imagen y media en el batch; total = recon + beta·kl."""

    x = torch.zeros(2, 1, 2, 2)
    recon = torch.full((2, 1, 2, 2), 0.5)
    mu = torch.tensor([[1.0, 2.0, 0.0], [0.0, 0.0, 0.0]])
    log_var = torch.zeros(2, 3)
    r, kl, total = m.vae_loss(recon, x, mu, log_var, beta=1.0)
    assert r.dim() == kl.dim() == total.dim() == 0, "Deben ser escalares."
    assert torch.isclose(r, torch.tensor(4 * math.log(2.0))), f"recon={r.item()}"
    assert torch.isclose(kl, torch.tensor(1.25)), f"kl={kl.item()}"
    assert torch.isclose(total, r + kl), "total = recon + kl con beta=1."
    ref = F.binary_cross_entropy(recon, x, reduction="sum") / 2
    assert torch.isclose(r, ref), "Usa BCE con reduction='sum' / batch."


@pytest.mark.parametrize("beta", [0.0, 0.5, 4.0])
def test_vae_loss_scales_with_beta(beta: float) -> None:
    """Solo el término KL escala linealmente con beta (β-VAE)."""

    x = torch.rand(BATCH_SIZE, 1, 28, 28)
    recon = torch.rand(BATCH_SIZE, 1, 28, 28) * 0.98 + 0.01
    mu, log_var = torch.randn(BATCH_SIZE, 3), torch.randn(BATCH_SIZE, 3)
    r1, kl1, _ = m.vae_loss(recon, x, mu, log_var, beta=1.0)
    r, kl, total = m.vae_loss(recon, x, mu, log_var, beta=beta)
    assert torch.isclose(r, r1) and torch.isclose(kl, kl1), "recon y kl no dependen de beta."
    assert torch.isclose(total, r1 + beta * kl1), "total = recon + beta * kl."


def test_interpolate() -> None:
    """Los extremos de la interpolación son decode(mu1) y decode(mu2)."""

    model = _model().eval()
    x1, x2 = torch.rand(1, 1, 28, 28), torch.rand(1, 1, 28, 28)
    out = m.interpolate(model, x1, x2, steps=5)
    assert out.shape == (5, 1, 28, 28), f"Forma {out.shape}"
    mu1, _ = model.encode(x1)
    mu2, _ = model.encode(x2)
    assert_close(out[0], model.decode(mu1)[0])
    assert_close(out[-1], model.decode(mu2)[0])
    assert_close(out[2], model.decode((mu1 + mu2) / 2)[0])


def test_vae_train_step() -> None:
    """Los parámetros cambian, se usa modo train y la pérdida baja."""

    model = _model().eval()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    x = (torch.rand(BATCH_SIZE, 1, 28, 28) > 0.5).float()
    before = [p.clone() for p in model.parameters()]
    r, kl, total = m.vae_train_step(model, x, opt, beta=1.0)
    assert all(isinstance(v, float) for v in (r, kl, total)), "Devuelve floats."
    assert total == pytest.approx(r + kl, rel=1e-4), "total = recon + kl."
    assert model.training, "El modelo debe quedar en modo train."
    assert any(torch.norm(p - q) > 0 for p, q in zip(model.parameters(), before)), (
        "Los parámetros deben cambiar."
    )
    for _ in range(20):
        *_, last = m.vae_train_step(model, x, opt, beta=1.0)
    assert last < total, "La pérdida debe bajar en un batch fijo."
