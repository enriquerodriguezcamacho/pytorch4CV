"""
Tests de F11 · Arquitecturas: MLP, LeNet-5, SmallResNet y CNN desde configuración.
"""

import pytest
import torch
from helpers import load
from torch import nn

m = load("fundamentals/11_architectures/classic_cnns.py")

BATCH = 3


def n_params(module: nn.Module) -> int:
    """Número de parámetros entrenables."""

    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# ---------------------------------------------------------------- MLP


@pytest.mark.parametrize("hidden", [[], [16], [32, 16]])
@pytest.mark.parametrize("dropout", [0.0, 0.5])
def test_mlp_shape(hidden: list[int], dropout: float) -> None:
    """Aplana la imagen y devuelve logits."""

    model = m.MLP(28 * 28, hidden, 10, dropout)
    out = model(torch.randn(BATCH, 1, 28, 28))
    assert out.shape == (BATCH, 10), "Forma incorrecta."
    linears = [mod for mod in model.modules() if isinstance(mod, nn.Linear)]
    assert len(linears) == len(hidden) + 1, "Número de capas lineales incorrecto."
    has_dropout = any(isinstance(mod, nn.Dropout) for mod in model.modules())
    assert has_dropout == (dropout > 0 and len(hidden) > 0), "Dropout incorrecto."


def test_mlp_params() -> None:
    """784·128+128 + 128·10+10 = 101 770."""

    assert n_params(m.MLP(784, [128], 10)) == 101770, "Parámetros incorrectos."


def test_mlp_no_final_activation() -> None:
    """La salida son logits: puede haber valores negativos."""

    out = m.MLP(4, [8], 3)(torch.randn(64, 4))
    assert (out < 0).any(), "No apliques activación en la capa final."


# ---------------------------------------------------------------- LeNet-5


def test_lenet5_shape_and_params() -> None:
    """32×32 → 10 logits con 61 706 parámetros."""

    model = m.LeNet5()
    out = model(torch.randn(BATCH, 1, 32, 32))
    assert out.shape == (BATCH, 10), "Forma incorrecta."
    assert n_params(model) == 61706, f"Parámetros incorrectos: {n_params(model)}."


def test_lenet5_features_shape() -> None:
    """Las features tienen forma [B, 16, 5, 5]."""

    model = m.LeNet5(in_channels=3, num_classes=4)
    feats = model.features(torch.randn(BATCH, 3, 32, 32))
    assert feats.shape == (BATCH, 16, 5, 5), "Forma de las features incorrecta."
    assert model(torch.randn(BATCH, 3, 32, 32)).shape == (BATCH, 4), "Forma incorrecta."


def test_lenet5_train_step_changes_params() -> None:
    """Un paso de SGD modifica los pesos."""

    model = m.LeNet5()
    old = [p.detach().clone() for p in model.parameters()]
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    loss = nn.functional.cross_entropy(model(torch.randn(4, 1, 32, 32)), torch.arange(4))
    loss.backward()
    opt.step()
    for o, p in zip(old, model.parameters(), strict=True):
        assert torch.norm(p - o) > 0, "Algún parámetro no se ha actualizado."


# ---------------------------------------------------------------- build_cnn_from_config


def test_build_cnn_tiny_config() -> None:
    """Configuración tipo YOLO: nº de bloques, canales y forma de salida."""

    net = m.build_cnn_from_config(m.TINY_CONFIG, in_channels=3)
    assert isinstance(net, nn.Sequential), "Debe devolver nn.Sequential."
    blocks = [mod for mod in net if isinstance(mod, m.CNNBlock)]
    pools = [mod for mod in net if isinstance(mod, nn.MaxPool2d)]
    assert len(blocks) == 1 + 1 + 2 + 2 * 2 + 1, "Número de CNNBlock incorrecto."
    assert len(pools) == 3, "Número de MaxPool incorrecto."
    out = net(torch.randn(2, 3, 64, 64))
    # 64 →(s2) 32 →M 16 →M 8 →M 4 →(s2) 2
    assert out.shape == (2, 64, 2, 2), f"Forma incorrecta: {tuple(out.shape)}."


def test_build_cnn_conv_hyperparameters() -> None:
    """Cada tupla se traduce a (kernel, out_channels, stride, padding) en orden."""

    net = m.build_cnn_from_config([(5, 6, 2, 2), "M", (1, 4, 1, 0)], in_channels=2)
    conv0, conv1 = net[0].conv, net[2].conv
    assert (conv0.in_channels, conv0.out_channels) == (2, 6), "Canales del primer bloque."
    assert conv0.kernel_size == (5, 5) and conv0.stride == (2, 2), "Kernel/stride."
    assert conv0.padding == (2, 2), "Padding incorrecto."
    assert (conv1.in_channels, conv1.out_channels) == (6, 4), "Encadena los canales."
    assert net[1].kernel_size == 2 and net[1].stride == 2, "MaxPool 2×2 / 2."


def test_build_cnn_repeats_chain_channels() -> None:
    """Las listas se repiten ``num_repeats`` veces encadenando canales."""

    net = m.build_cnn_from_config([[(1, 4, 1, 0), (3, 8, 1, 1), 3]], in_channels=8)
    chans = [(b.conv.in_channels, b.conv.out_channels) for b in net]
    assert chans == [(8, 4), (4, 8)] * 3, f"Canales incorrectos: {chans}."


@pytest.mark.parametrize("bad", ["X", 3, {"k": 3}])
def test_build_cnn_raises(bad: object) -> None:
    """Elementos no válidos → ``ValueError``."""

    with pytest.raises(ValueError):
        m.build_cnn_from_config([(3, 4, 1, 1), bad])


# ---------------------------------------------------------------- SmallResNet


@pytest.mark.parametrize("size", [16, 32])
def test_small_resnet(size: int) -> None:
    """Forma de salida independiente del tamaño gracias al GAP."""

    model = m.SmallResNet(3, 7, (8, 16))
    out = model(torch.randn(2, 3, size, size))
    assert out.shape == (2, 7), "Forma incorrecta."
    assert len(model.stages) == 4, "Dos bloques residuales por etapa."
    assert isinstance(model.pool, nn.AdaptiveAvgPool2d), "Usa global average pooling."
    feats = model.stages(model.stem(torch.randn(1, 3, size, size)))
    assert feats.shape == (1, 16, size // 2, size // 2), "Solo la 2ª etapa reduce."


def test_small_resnet_train_eval_mode() -> None:
    """``model.eval()`` se propaga a todos los BatchNorm y es determinista."""

    model = m.SmallResNet(3, 4, (8,))
    model.eval()
    assert all(not mod.training for mod in model.modules()), "Todo debe estar en eval."
    x = torch.randn(2, 3, 16, 16)
    assert torch.equal(model(x), model(x)), "En eval la salida es determinista."
