"""
Tests de CV07 · Generación condicional (CGAN y CVAE).
"""

import math

import pytest
import torch
from helpers import load
from torch import nn

m = load("cv/07_generative_models/conditional.py")

LATENT_DIM = 8
NUM_CLASSES = 10
EMBED_DIM = 4
HIDDEN = 8
BATCH_SIZE = 6


def test_one_hot_label_maps() -> None:
    """Un canal constante a 1 por ejemplo; el resto a 0."""

    labels = torch.tensor([2, 0])
    maps = m.one_hot_label_maps(labels, 3, 4, 5)
    assert maps.shape == (2, 3, 4, 5), f"Forma {maps.shape}"
    assert maps.dtype == torch.float32, "Los mapas deben ser float."
    assert torch.all(maps[0, 2] == 1) and torch.all(maps[0, :2] == 0), "Ejemplo 0 mal."
    assert torch.all(maps[1, 0] == 1) and maps[1].sum() == 20, "Ejemplo 1 mal."


@pytest.mark.parametrize("batch", [1, 6])
def test_conditional_generator(batch: int) -> None:
    """Forma, rango [-1, 1], uso de nn.Embedding y condicionamiento efectivo."""

    gen = m.ConditionalGenerator(LATENT_DIM, NUM_CLASSES, EMBED_DIM, HIDDEN).eval()
    out = gen(torch.randn(batch, LATENT_DIM), torch.randint(0, NUM_CLASSES, (batch,)))
    assert out.shape == (batch, 1, 28, 28), f"Forma {out.shape}"
    assert out.min() >= -1 and out.max() <= 1, "Salida con Tanh."
    assert any(isinstance(mod, nn.Embedding) for mod in gen.modules()), "Usa nn.Embedding."
    z = torch.randn(1, LATENT_DIM)
    assert not torch.allclose(gen(z, torch.tensor([0])), gen(z, torch.tensor([1]))), (
        "Etiquetas distintas deben dar imágenes distintas."
    )
    assert torch.allclose(gen(z, torch.tensor([3])), gen(z, torch.tensor([3]))), (
        "En eval, mismo (z, y) → misma imagen."
    )


def test_conditional_discriminator() -> None:
    """Logits [batch, 1], primera conv con 1 + K canales y condicionamiento."""

    disc = m.ConditionalDiscriminator(NUM_CLASSES, HIDDEN).eval()
    x = torch.randn(BATCH_SIZE, 1, 28, 28)
    out = disc(x, torch.randint(0, NUM_CLASSES, (BATCH_SIZE,)))
    assert out.shape == (BATCH_SIZE, 1), f"Forma {out.shape}"
    first_conv = next(mod for mod in disc.modules() if isinstance(mod, nn.Conv2d))
    assert first_conv.in_channels == 1 + NUM_CLASSES, (
        "La etiqueta entra como K mapas one-hot concatenados a la imagen."
    )
    x1 = x[:1]
    assert not torch.allclose(disc(x1, torch.tensor([0])), disc(x1, torch.tensor([7]))), (
        "D debe depender de la etiqueta."
    )
    x_grad = x.clone().requires_grad_(True)
    disc(x_grad, torch.zeros(BATCH_SIZE, dtype=torch.long)).sum().backward()
    assert x_grad.grad is not None and x_grad.grad.abs().sum() > 0, "Sin gradiente."


def test_cgan_train_step() -> None:
    """Ambas redes cambian, las pérdidas son finitas y se usa detach en D."""

    gen = m.ConditionalGenerator(LATENT_DIM, NUM_CLASSES, EMBED_DIM, HIDDEN)
    disc = m.ConditionalDiscriminator(NUM_CLASSES, HIDDEN)
    g_opt = torch.optim.Adam(gen.parameters(), lr=1e-3)
    d_opt = torch.optim.Adam(disc.parameters(), lr=1e-3)
    real = torch.rand(BATCH_SIZE, 1, 28, 28) * 2 - 1
    labels = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))
    g_before = [p.clone() for p in gen.parameters()]
    d_before = [p.clone() for p in disc.parameters()]
    calls = []
    handle = gen.label_emb.weight.register_hook(lambda g: calls.append(1))
    d_loss, g_loss = m.cgan_train_step(gen, disc, real, labels, g_opt, d_opt, LATENT_DIM)
    handle.remove()
    assert isinstance(d_loss, float) and isinstance(g_loss, float), "Devuelve floats."
    assert math.isfinite(d_loss) and math.isfinite(g_loss), "Pérdidas finitas."
    assert len(calls) == 1, "Usa fake.detach() en el paso del discriminador."
    assert any(torch.norm(p - q) > 0 for p, q in zip(gen.parameters(), g_before)), "G fijo."
    assert any(torch.norm(p - q) > 0 for p, q in zip(disc.parameters(), d_before)), "D fijo."
    for _ in range(5):
        d_loss, g_loss = m.cgan_train_step(gen, disc, real, labels, g_opt, d_opt, LATENT_DIM)
        assert math.isfinite(d_loss) and math.isfinite(g_loss), "La pérdida explota."


def test_cvae_shapes_and_conditioning() -> None:
    """encode/decode/forward con etiqueta y generate por clase."""

    model = m.CVAE(NUM_CLASSES, 32, 4)
    x = torch.rand(BATCH_SIZE, 1, 28, 28)
    labels = torch.randint(0, NUM_CLASSES, (BATCH_SIZE,))
    mu, log_var = model.encode(x, labels)
    assert mu.shape == log_var.shape == (BATCH_SIZE, 4), "mu/log_var con forma incorrecta."
    mu_other, _ = model.encode(x, (labels + 1) % NUM_CLASSES)
    assert not torch.allclose(mu, mu_other), "q(z|x, y) debe depender de y."
    recon, mu2, _ = model(x, labels)
    assert recon.shape == x.shape, "Reconstrucción con forma incorrecta."
    assert torch.allclose(mu, mu2), "forward usa encode."
    z = torch.zeros(2, 4)
    imgs = model.decode(z, torch.tensor([1, 5]))
    assert imgs.shape == (2, 1, 28, 28), "decode con forma incorrecta."
    assert not torch.allclose(imgs[0], imgs[1]), "p(x|z, y) debe depender de y."
    gen = model.generate(torch.arange(NUM_CLASSES))
    assert gen.shape == (NUM_CLASSES, 1, 28, 28), "Una imagen por etiqueta."
    assert gen.min() >= 0 and gen.max() <= 1, "Salida con sigmoide."
