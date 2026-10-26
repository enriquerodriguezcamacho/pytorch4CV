"""
Tests de F11 · Arquitecturas: autoencoder, red siamesa y multi-tarea.
"""

import pytest
import torch
import torch.nn.functional as F
from helpers import assert_close, load
from torch import nn

ae = load("fundamentals/11_architectures/autoencoder.py")
sm = load("fundamentals/11_architectures/siamese.py")

BATCH = 4


# ---------------------------------------------------------------- autoencoder


@pytest.mark.parametrize("in_channels, size", [(1, 28), (3, 32)])
@pytest.mark.parametrize("latent", [8, 32])
def test_autoencoder_shapes(in_channels: int, size: int, latent: int) -> None:
    """Código de tamaño H/4 y reconstrucción del tamaño de la entrada."""

    model = ae.ConvAutoencoder(in_channels, latent)
    x = torch.rand(BATCH, in_channels, size, size)
    z = model.encode(x)
    assert z.shape == (BATCH, latent, size // 4, size // 4), "Forma del código."
    rec = model.decode(z)
    assert rec.shape == x.shape, "Forma de la reconstrucción."
    assert_close(model(x), rec)


def test_autoencoder_output_range() -> None:
    """La sigmoide final deja la reconstrucción en [0, 1]."""

    out = ae.ConvAutoencoder()(torch.randn(BATCH, 1, 28, 28) * 10)
    assert ((out >= 0) & (out <= 1)).all(), "La salida debe estar en [0, 1]."


def test_autoencoder_uses_transposed_convs() -> None:
    """El decoder sube resolución con ``ConvTranspose2d``."""

    model = ae.ConvAutoencoder()
    tconvs = [mod for mod in model.decoder.modules() if isinstance(mod, nn.ConvTranspose2d)]
    assert len(tconvs) == 2, "El decoder debe tener 2 ConvTranspose2d."
    assert all(t.stride == (2, 2) for t in tconvs), "Con stride 2."


def test_train_autoencoder_step() -> None:
    """El paso entrena (parámetros cambian, modo train) y la pérdida baja."""

    model = ae.ConvAutoencoder(1, 8)
    model.eval()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    x = torch.rand(BATCH, 1, 16, 16)
    old = [p.detach().clone() for p in model.parameters()]
    first = ae.train_autoencoder_step(model, x, opt)
    assert isinstance(first, float), "Debe devolver un float."
    assert model.training, "El paso debe poner el modelo en modo train."
    for o, p in zip(old, model.parameters(), strict=True):
        assert torch.norm(p.detach() - o) > 0, "Algún parámetro no cambió."
    for _ in range(30):
        last = ae.train_autoencoder_step(model, x, opt)
    assert last < first, "La pérdida debería bajar al sobreajustar un lote."


# ---------------------------------------------------------------- siamesa


def test_siamese_embed_unit_norm() -> None:
    """Embeddings de norma 1."""

    net = sm.SiameseNetwork(1, 16)
    emb = net.embed(torch.randn(BATCH, 1, 16, 16))
    assert emb.shape == (BATCH, 16), "Forma incorrecta."
    assert_close(emb.norm(dim=1), torch.ones(BATCH))


def test_siamese_distance_properties() -> None:
    """d(x, x) = 0, simetría y rango [0, 2]."""

    net = sm.SiameseNetwork(3, 8).eval()
    a, b = torch.randn(BATCH, 3, 16, 16), torch.randn(BATCH, 3, 16, 16)
    d = net(a, b)
    assert d.shape == (BATCH,), "Forma incorrecta."
    assert_close(net(a, a), torch.zeros(BATCH), atol=1e-3)
    assert_close(d, net(b, a))
    assert ((d >= 0) & (d <= 2 + 1e-5)).all(), "Distancia entre embeddings unitarios."


def test_siamese_shares_weights() -> None:
    """Las dos ramas comparten pesos: el gradiente acumula ambas ramas en un solo módulo."""

    net = sm.SiameseNetwork(1, 8)
    a, b = torch.randn(2, 1, 8, 8), torch.randn(2, 1, 8, 8)
    net(a, b).sum().backward()
    n_modules = sum(1 for mod in net.modules() if isinstance(mod, nn.Conv2d))
    assert n_modules == 2, "Debe haber una única embedding_net (2 convs)."
    expected = (net.embed(a) - net.embed(b)).norm(dim=1)
    assert_close(net(a, b), expected)


def test_contrastive_loss_hand_values() -> None:
    """Valores a mano con margin = 1."""

    d = torch.tensor([0.5, 0.5, 2.0, 0.2])
    y = torch.tensor([1, 0, 0, 1])
    # [0.25, (1-0.5)^2=0.25, 0, 0.04] → media = 0.135
    loss = sm.contrastive_loss(d, y, margin=1.0)
    assert loss.dim() == 0, "Debe ser un escalar."
    assert torch.isclose(loss, torch.tensor(0.135)), f"Valor incorrecto: {loss.item()}."
    assert torch.isclose(sm.contrastive_loss(d, y, margin=3.0), torch.tensor((0.25 + 6.25 + 1 + 0.04) / 4)), (
        "El margen afecta solo a las parejas distintas."
    )


# ---------------------------------------------------------------- multi-tarea


def test_multitask_shapes() -> None:
    """Dos salidas: logits y caja en [0, 1]."""

    net = sm.MultiTaskNet(3, 5, 32)
    logits, boxes = net(torch.randn(BATCH, 3, 16, 16))
    assert logits.shape == (BATCH, 5), "Forma de los logits."
    assert boxes.shape == (BATCH, 4), "Forma de las cajas."
    assert ((boxes >= 0) & (boxes <= 1)).all(), "Cajas normalizadas con sigmoide."


@pytest.mark.parametrize("weight", [0.0, 1.0, 2.0])
def test_multitask_loss(weight: float) -> None:
    """CE + peso · SmoothL1, y la parte de regresión escala linealmente."""

    logits, labels = torch.randn(BATCH, 5), torch.tensor([0, 1, 2, 3])
    boxes, target = torch.rand(BATCH, 4), torch.rand(BATCH, 4)
    loss = sm.multitask_loss(logits, boxes, labels, target, weight)
    expected = F.cross_entropy(logits, labels) + weight * F.smooth_l1_loss(boxes, target)
    assert_close(loss, expected)


def test_multitask_backbone_gets_both_gradients() -> None:
    """El backbone compartido recibe gradiente de las dos cabezas."""

    net = sm.MultiTaskNet(3, 5, 32)
    x = torch.randn(BATCH, 3, 16, 16)
    _, boxes = net(x)
    boxes.sum().backward()
    g_box = net.backbone[-1].weight.grad.clone()
    assert net.cls_head.weight.grad is None, "La caja no debe tocar la cabeza de clase."
    net.zero_grad()
    logits, _ = net(x)
    logits[:, 0].sum().backward()
    assert g_box.abs().sum() > 0 and net.backbone[-1].weight.grad.abs().sum() > 0, (
        "El backbone debe recibir gradiente de ambas tareas."
    )
