"""
Tests de CV01 · bloques de ResNet, GoogLeNet y DenseNet.
"""

import pytest
import torch
from helpers import load
from torch import nn

b = load("cv/01_introduction_transfer_learning/blocks.py")

BATCH = 2


@pytest.mark.parametrize("in_ch, out_ch, stride", [(4, 4, 1), (4, 8, 2), (4, 8, 1)])
@pytest.mark.parametrize("size", [8, 16])
def test_residual_block_shapes(in_ch: int, out_ch: int, stride: int, size: int) -> None:
    """La salida tiene ``out_ch`` canales y resolución ``size / stride``."""

    block = b.ResidualBlock(in_ch, out_ch, stride)
    out = block(torch.randn(BATCH, in_ch, size, size))
    assert out.shape == (BATCH, out_ch, size // stride, size // stride), "Forma."
    assert (out >= 0).all(), "La salida pasa por una ReLU final."


def test_residual_block_shortcut_type() -> None:
    """Identidad si la forma no cambia; proyección si cambia."""

    assert isinstance(b.ResidualBlock(4, 4, 1).shortcut, nn.Identity), "Identidad."
    proj = b.ResidualBlock(4, 8, 2).shortcut
    assert not isinstance(proj, nn.Identity), "Debe haber projection shortcut."
    assert any(isinstance(mod, nn.Conv2d) for mod in proj.modules()), "Falta la 1×1."


def test_residual_block_zero_branch_is_identity() -> None:
    """Si la rama residual vale 0, el bloque devuelve ``relu(x)``."""

    block = b.ResidualBlock(3, 3, 1).eval()
    nn.init.zeros_(block.bn2.weight)
    nn.init.zeros_(block.bn2.bias)
    x = torch.randn(BATCH, 3, 5, 5)
    assert torch.allclose(block(x), torch.relu(x)), "F(x)=0 ⇒ salida = relu(x)."


def test_residual_block_gradients() -> None:
    """El gradiente llega a la entrada."""

    x = torch.randn(BATCH, 4, 8, 8, requires_grad=True)
    b.ResidualBlock(4, 8, 2)(x).sum().backward()
    assert x.grad is not None and x.grad.abs().sum() > 0, "No fluye el gradiente."


def test_inception_module() -> None:
    """Canales = c1 + c3 + c5 + pool_proj y misma resolución."""

    module = b.InceptionModule(8, 4, 3, 6, 2, 5, 7)
    out = module(torch.randn(BATCH, 8, 10, 10))
    assert out.shape == (BATCH, 4 + 6 + 5 + 7, 10, 10), "Forma incorrecta."
    convs = [mod for mod in module.modules() if isinstance(mod, nn.Conv2d)]
    kernels = sorted(c.kernel_size[0] for c in convs)
    assert kernels == [1, 1, 1, 1, 3, 5], "Debe haber cuatro 1×1, una 3×3 y una 5×5."


def test_inception_branch_order() -> None:
    """Los primeros ``c1`` canales son la rama 1×1."""

    module = b.InceptionModule(3, 2, 2, 2, 2, 2, 2)
    x = torch.randn(1, 3, 6, 6)
    assert torch.allclose(module(x)[:, :2], module.branch1(x)), "Orden de las ramas."


@pytest.mark.parametrize("num_layers, k", [(1, 4), (3, 2), (4, 8)])
def test_dense_block(num_layers: int, k: int) -> None:
    """Salida con ``in + L·k`` canales y la entrada copiada al principio."""

    block = b.DenseBlock(5, k, num_layers)
    x = torch.randn(BATCH, 5, 6, 6)
    out = block(x)
    assert out.shape == (BATCH, 5 + num_layers * k, 6, 6), "Forma incorrecta."
    assert torch.equal(out[:, :5], x), "Los primeros canales deben ser la entrada."
    in_chs = [layer[2].in_channels for layer in block.layers]
    assert in_chs == [5 + i * k for i in range(num_layers)], "Conexiones densas."
