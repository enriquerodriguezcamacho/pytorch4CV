"""
Tests de F11 · Arquitecturas: bloques (VGG, ResNet, Inception, DenseNet, SE).
"""

import pytest
import torch
from helpers import assert_close, load
from torch import nn

m = load("fundamentals/11_architectures/blocks.py")

BATCH = 2
SIZE = 8


def n_params(module: nn.Module) -> int:
    """Número de parámetros entrenables."""

    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# ---------------------------------------------------------------- VGG


@pytest.mark.parametrize("num_convs", [1, 2, 3])
@pytest.mark.parametrize("batch_norm", [False, True])
def test_vgg_block_shape(num_convs: int, batch_norm: bool) -> None:
    """Mantiene canales de salida y divide la resolución entre 2."""

    block = m.VGGBlock(3, 8, num_convs, batch_norm)
    out = block(torch.randn(BATCH, 3, SIZE, SIZE))
    assert out.shape == (BATCH, 8, SIZE // 2, SIZE // 2), "Forma incorrecta."
    convs = [mod for mod in block.modules() if isinstance(mod, nn.Conv2d)]
    assert len(convs) == num_convs, "Número de convoluciones incorrecto."
    assert all(c.kernel_size == (3, 3) and c.padding == (1, 1) for c in convs), "Usa 3×3/p1."
    bns = [mod for mod in block.modules() if isinstance(mod, nn.BatchNorm2d)]
    assert len(bns) == (num_convs if batch_norm else 0), "BatchNorm incorrecto."


def test_vgg_block_params() -> None:
    """Primer bloque de VGG-16: (3·9·64+64) + (64·9·64+64) = 38 720."""

    assert n_params(m.VGGBlock(3, 64, 2)) == 38720, "Parámetros incorrectos."


# ---------------------------------------------------------------- ResidualBlock


@pytest.mark.parametrize(
    "cin, cout, stride", [(8, 8, 1), (8, 16, 1), (8, 8, 2), (8, 16, 2)]
)
def test_residual_block_shape(cin: int, cout: int, stride: int) -> None:
    """Formas y uso de proyección solo cuando hace falta."""

    block = m.ResidualBlock(cin, cout, stride)
    out = block(torch.randn(BATCH, cin, SIZE, SIZE))
    assert out.shape == (BATCH, cout, SIZE // stride, SIZE // stride), "Forma incorrecta."
    needs_proj = cin != cout or stride != 1
    is_identity = isinstance(block.shortcut, nn.Identity)
    assert is_identity != needs_proj, "Proyección 1×1 solo si cambian canales o stride."
    if needs_proj:
        conv = next(mod for mod in block.shortcut.modules() if isinstance(mod, nn.Conv2d))
        assert conv.kernel_size == (1, 1), "La proyección es una conv 1×1."
        assert conv.stride == (stride, stride), "La proyección debe usar el stride."


@pytest.mark.parametrize("train", [True, False])
def test_residual_block_zero_weights_is_relu_identity(train: bool) -> None:
    """Con la rama residual a cero, el bloque es ``ReLU(x)``."""

    block = m.ResidualBlock(4, 4)
    nn.init.zeros_(block.conv1.weight)
    nn.init.zeros_(block.conv2.weight)
    block.train(train)
    x = torch.randn(BATCH, 4, SIZE, SIZE)
    assert_close(block(x), torch.relu(x))


def test_residual_block_non_negative_and_grad() -> None:
    """Salida ≥ 0 (ReLU final) y el gradiente llega a la entrada."""

    block = m.ResidualBlock(4, 8, 2)
    x = torch.randn(BATCH, 4, SIZE, SIZE, requires_grad=True)
    out = block(x)
    assert (out >= 0).all(), "Falta la ReLU tras la suma."
    out.sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0, "No fluye el gradiente."


def test_residual_block_params() -> None:
    """Bloque 64→64 de ResNet-18: 2·(64·64·9) + 2·(2·64) = 73 984."""

    assert n_params(m.ResidualBlock(64, 64)) == 73984, "Parámetros incorrectos."


# ---------------------------------------------------------------- Bottleneck


@pytest.mark.parametrize("cin, mid, stride", [(16, 4, 1), (8, 4, 1), (16, 4, 2)])
def test_bottleneck_shape(cin: int, mid: int, stride: int) -> None:
    """Salida con ``mid * 4`` canales y resolución dividida por el stride."""

    block = m.BottleneckBlock(cin, mid, stride)
    out = block(torch.randn(BATCH, cin, SIZE, SIZE))
    assert out.shape == (BATCH, mid * 4, SIZE // stride, SIZE // stride), "Forma incorrecta."
    assert block.conv1.kernel_size == (1, 1), "conv1 es 1×1."
    assert block.conv2.kernel_size == (3, 3), "conv2 es 3×3."
    assert block.conv3.kernel_size == (1, 1), "conv3 es 1×1."
    assert isinstance(block.shortcut, nn.Identity) == (cin == mid * 4 and stride == 1), (
        "Shortcut incorrecto."
    )


def test_bottleneck_params_vs_torchvision() -> None:
    """Mismo número de parámetros que un bottleneck de torchvision (ResNet-50)."""

    from torchvision.models.resnet import Bottleneck

    down = nn.Sequential(nn.Conv2d(64, 256, 1, bias=False), nn.BatchNorm2d(256))
    ref = Bottleneck(64, 64, downsample=down)
    assert n_params(m.BottleneckBlock(64, 64)) == n_params(ref), "Parámetros incorrectos."


def test_bottleneck_cheaper_than_basic() -> None:
    """El bottleneck 256→256 es más barato que dos 3×3 de 256 canales."""

    assert n_params(m.BottleneckBlock(256, 64)) < n_params(m.ResidualBlock(256, 256)) / 4, (
        "El bottleneck debería ahorrar parámetros."
    )


# ---------------------------------------------------------------- Inception

INCEPTION_3A = (192, 64, 96, 128, 16, 32, 32)


def test_inception_shape_3a() -> None:
    """Inception 3a de GoogLeNet: 192 → 64 + 128 + 32 + 32 = 256 canales."""

    block = m.InceptionModule(*INCEPTION_3A)
    out = block(torch.randn(1, 192, 7, 7))
    assert out.shape == (1, 256, 7, 7), "Forma incorrecta."


def test_inception_params_3a() -> None:
    """Calculado a mano: 12352 + 18528 + 110720 + 3088 + 12832 + 6176 = 163 696."""

    assert n_params(m.InceptionModule(*INCEPTION_3A)) == 163696, "Parámetros incorrectos."


def test_inception_concat_order() -> None:
    """Los primeros ``c1`` canales son exactamente la rama 1×1."""

    block = m.InceptionModule(4, 2, 3, 5, 1, 3, 2).eval()
    x = torch.randn(BATCH, 4, SIZE, SIZE)
    out = block(x)
    assert out.shape == (BATCH, 2 + 5 + 3 + 2, SIZE, SIZE), "Forma incorrecta."
    assert_close(out[:, :2], block.branch1(x))
    assert_close(out[:, -2:], block.branch_pool(x))
    assert (out >= 0).all(), "Todas las ramas acaban en ReLU."


# ---------------------------------------------------------------- DenseBlock


@pytest.mark.parametrize("cin, k, num_layers", [(4, 2, 1), (4, 3, 3), (8, 4, 4)])
def test_dense_block_shape(cin: int, k: int, num_layers: int) -> None:
    """Canales de salida ``C + L·k`` y entrada copiada al principio."""

    block = m.DenseBlock(cin, k, num_layers)
    x = torch.randn(BATCH, cin, SIZE, SIZE)
    out = block(x)
    assert out.shape == (BATCH, cin + num_layers * k, SIZE, SIZE), "Forma incorrecta."
    assert torch.equal(out[:, :cin], x), "La entrada debe ir concatenada primero."
    assert len(block.layers) == num_layers, "Número de capas incorrecto."
    for i, layer in enumerate(block.layers):
        assert layer.conv.in_channels == cin + i * k, f"Canales de entrada de la capa {i}."


def test_dense_block_params() -> None:
    """Parámetros a mano: Σ_l [2(C+lk) + (C+lk)·k·9]."""

    cin, k, num_layers = 4, 3, 3
    expected = sum(2 * (cin + i * k) + (cin + i * k) * k * 9 for i in range(num_layers))
    assert n_params(m.DenseBlock(cin, k, num_layers)) == expected, "Parámetros incorrectos."


# ---------------------------------------------------------------- SE


def test_se_block_shape_and_range() -> None:
    """Misma forma y factor de escala en (0, 1)."""

    block = m.SEBlock(16, reduction=4)
    x = torch.rand(BATCH, 16, SIZE, SIZE) + 0.1
    out = block(x)
    ratio = out / x
    assert out.shape == x.shape, "Forma incorrecta."
    assert ((ratio > 0) & (ratio < 1)).all(), "El factor debe salir de una sigmoide."
    assert_close(ratio, ratio[:, :, :1, :1].expand_as(ratio))  # un factor por canal


def test_se_block_zero_params_halves() -> None:
    """Con todos los parámetros a cero, ``s = sigmoid(0) = 0.5``."""

    block = m.SEBlock(8, reduction=2)
    for p in block.parameters():
        nn.init.zeros_(p)
    x = torch.randn(BATCH, 8, SIZE, SIZE)
    assert_close(block(x), 0.5 * x)


def test_se_block_params() -> None:
    """C=64, r=16: (64·4+4) + (4·64+64) = 580."""

    assert n_params(m.SEBlock(64, 16)) == 580, "Parámetros incorrectos."
