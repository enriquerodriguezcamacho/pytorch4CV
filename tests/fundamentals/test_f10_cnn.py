"""
Tests de F10 · Capas convolucionales.
"""

import pytest
import torch
import torch.nn.functional as F
from helpers import assert_close, load
from torch import nn

m = load("fundamentals/10_cnn_layers/cnn_layers.py")

BATCH = 2
IN_CHANNELS = 3
OUT_CHANNELS = 4
SIZE = 9


# ---------------------------------------------------------------- tamaños


@pytest.mark.parametrize("n", [7, 8, 32])
@pytest.mark.parametrize("k", [1, 3, 5])
@pytest.mark.parametrize("s", [1, 2])
@pytest.mark.parametrize("p", [0, 1, 2])
@pytest.mark.parametrize("d", [1, 2])
def test_conv_output_size_vs_pytorch(n: int, k: int, s: int, p: int, d: int) -> None:
    """La fórmula coincide con la forma real de ``F.conv2d``."""

    if n + 2 * p < d * (k - 1) + 1:
        pytest.skip("Configuración imposible.")
    x = torch.zeros(1, 1, n, n)
    w = torch.zeros(1, 1, k, k)
    expected = F.conv2d(x, w, stride=s, padding=p, dilation=d).shape[-1]
    got = m.conv_output_size(n, k, s, p, d)
    assert got == expected, f"Esperado {expected}, obtenido {got}."
    assert isinstance(got, int), "Debe devolver un int."


def test_conv_output_size_hand_values() -> None:
    """Casos clásicos calculados a mano."""

    assert m.conv_output_size(224, 7, 2, 3) == 112, "Primera conv de ResNet."
    assert m.conv_output_size(32, 3, 1, 1) == 32, "Padding 'same' con k=3."
    assert m.conv_output_size(32, 5) == 28, "Padding 'valid' con k=5 (LeNet)."
    assert m.conv_output_size(7, 3, 1, 0, 2) == 3, "Dilatación 2: k_eff = 5."


def test_conv_output_size_raises() -> None:
    """El kernel no cabe → ``ValueError``."""

    with pytest.raises(ValueError):
        m.conv_output_size(2, 5)


@pytest.mark.parametrize("n", [3, 4, 7])
@pytest.mark.parametrize("k", [2, 3, 4])
@pytest.mark.parametrize("s", [1, 2, 3])
@pytest.mark.parametrize("p", [0, 1])
@pytest.mark.parametrize("op", [0, 1])
def test_conv_transpose_output_size(n: int, k: int, s: int, p: int, op: int) -> None:
    """La fórmula coincide con ``nn.ConvTranspose2d``."""

    if op >= s:
        pytest.skip("output_padding debe ser menor que stride.")
    layer = nn.ConvTranspose2d(1, 1, k, stride=s, padding=p, output_padding=op)
    expected = layer(torch.zeros(1, 1, n, n)).shape[-1]
    got = m.conv_transpose_output_size(n, k, s, p, op)
    assert got == expected, f"Esperado {expected}, obtenido {got}."


def test_conv_transpose_doubles_size() -> None:
    """k=4, s=2, p=1 duplica exactamente el tamaño (DCGAN / U-Net)."""

    assert m.conv_transpose_output_size(16, 4, 2, 1) == 32, "Debería duplicar."
    assert m.conv_transpose_output_size(16, 3, 2, 1, 1) == 32, "k=3,s=2,p=1,op=1."


# ---------------------------------------------------------------- parámetros


@pytest.mark.parametrize(
    "cin, cout, k, groups, bias",
    [(3, 64, 3, 1, True), (64, 128, 1, 1, False), (32, 32, 3, 32, True), (8, 16, 5, 4, True)],
)
def test_count_conv_params(cin: int, cout: int, k: int, groups: int, bias: bool) -> None:
    """Coincide con el número de parámetros de ``nn.Conv2d``."""

    layer = nn.Conv2d(cin, cout, k, groups=groups, bias=bias)
    expected = sum(p.numel() for p in layer.parameters())
    assert m.count_conv_params(cin, cout, k, groups, bias) == expected, "Mal contado."


def test_count_conv_params_hand_value() -> None:
    """3·3·3·64 + 64 = 1792 (primera conv de VGG)."""

    assert m.count_conv_params(3, 64, 3) == 1792, "Debe ser 1792."


def test_count_conv_params_raises() -> None:
    """Canales no divisibles entre grupos → ``ValueError``."""

    with pytest.raises(ValueError):
        m.count_conv_params(6, 8, 3, groups=4)


def test_count_conv_macs() -> None:
    """MACs a mano: 32·32 · 16 · 3 · 3·3 = 442368; con groups divide entre g."""

    assert m.count_conv_macs(3, 16, 3, 32, 32) == 442368, "MACs incorrectas."
    assert m.count_conv_macs(16, 16, 3, 8, 8, groups=16) == 8 * 8 * 16 * 9, "Depthwise."


# ---------------------------------------------------------------- convolución


def test_conv2d_naive_hand_value() -> None:
    """Correlación cruzada (sin voltear el kernel) en un caso 3×3 / 2×2."""

    x = torch.arange(9.0).view(1, 1, 3, 3)
    w = torch.tensor([[[[1.0, 0.0], [0.0, -1.0]]]])
    out = m.conv2d_naive(x, w)
    expected = torch.full((1, 1, 2, 2), -4.0)
    assert out.shape == (1, 1, 2, 2), "Forma incorrecta."
    assert torch.isclose(out, expected).all(), "Valores incorrectos (¿volteaste el kernel?)."


@pytest.mark.parametrize("stride", [1, 2])
@pytest.mark.parametrize("padding", [0, 1, 2])
@pytest.mark.parametrize("use_bias", [False, True])
@pytest.mark.parametrize("fn", ["conv2d_naive", "conv2d_im2col"])
def test_conv2d_vs_pytorch(stride: int, padding: int, use_bias: bool, fn: str) -> None:
    """Ambas implementaciones coinciden con ``F.conv2d``."""

    x = torch.randn(BATCH, IN_CHANNELS, SIZE, SIZE - 1)
    w = torch.randn(OUT_CHANNELS, IN_CHANNELS, 3, 2)
    b = torch.randn(OUT_CHANNELS) if use_bias else None
    out = getattr(m, fn)(x, w, b, stride, padding)
    ref = F.conv2d(x, w, b, stride=stride, padding=padding)
    assert out.shape == ref.shape, f"Forma {tuple(out.shape)} != {tuple(ref.shape)}."
    assert_close(out, ref, atol=1e-4)


def test_conv2d_im2col_gradients() -> None:
    """im2col es diferenciable y da los mismos gradientes que ``F.conv2d``."""

    x = torch.randn(1, 2, 5, 5, requires_grad=True)
    w = torch.randn(3, 2, 3, 3, requires_grad=True)
    m.conv2d_im2col(x, w, padding=1).sum().backward()
    gx, gw = x.grad.clone(), w.grad.clone()
    x.grad, w.grad = None, None
    F.conv2d(x, w, padding=1).sum().backward()
    assert_close(gx, x.grad, atol=1e-4)
    assert_close(gw, w.grad, atol=1e-4)


# ---------------------------------------------------------------- pooling


@pytest.mark.parametrize("k, s", [(2, None), (3, 1), (3, 2), (2, 1)])
def test_max_pool2d(k: int, s: int | None) -> None:
    """Coincide con ``F.max_pool2d``."""

    x = torch.randn(BATCH, IN_CHANNELS, 8, 7)
    out = m.max_pool2d(x, k, s)
    ref = F.max_pool2d(x, k, s)
    assert out.shape == ref.shape, "Forma incorrecta."
    assert_close(out, ref)


def test_max_pool2d_hand_value() -> None:
    """Pooling 2×2 sobre una imagen 4×4 conocida."""

    x = torch.arange(16.0).view(1, 1, 4, 4)
    out = m.max_pool2d(x, 2)
    expected = torch.tensor([[[[5.0, 7.0], [13.0, 15.0]]]])
    assert torch.isclose(out, expected).all(), "Valores incorrectos."


# ---------------------------------------------------------------- campo receptivo


@pytest.mark.parametrize(
    "layers, expected",
    [
        ([(3, 1)], 3),
        ([(3, 1), (3, 1)], 5),  # dos 3×3 = un 5×5
        ([(3, 1), (3, 1), (3, 1)], 7),  # tres 3×3 = un 7×7 (VGG)
        ([(3, 1), (2, 2), (3, 1)], 8),
        ([(7, 2), (3, 2)], 11),  # stem de ResNet: conv 7×7/2 + maxpool 3×3/2
        ([(3, 1, 2)], 5),  # dilatación 2
        ([(3, 1, 1), (3, 1, 2), (3, 1, 4)], 15),  # WaveNet / ASPP
    ],
)
def test_receptive_field(layers: list[tuple[int, ...]], expected: int) -> None:
    """Valores calculados a mano."""

    got = m.receptive_field(layers)
    assert got == expected, f"Esperado {expected}, obtenido {got}."


def test_receptive_field_empirical() -> None:
    """El gradiente de un píxel de salida solo toca ``r×r`` píxeles de entrada."""

    layers = [(3, 1), (3, 2), (3, 1)]
    net = nn.Sequential(*[nn.Conv2d(1, 1, k, s) for k, s in layers])
    for p in net.parameters():
        nn.init.constant_(p, 1.0)
    x = torch.ones(1, 1, 21, 21, requires_grad=True)
    out = net(x)
    out[0, 0, out.shape[-2] // 2, out.shape[-1] // 2].backward()
    rows = (x.grad[0, 0].abs().sum(1) > 0).sum().item()
    assert rows == m.receptive_field(layers), "No coincide con el campo receptivo real."


# ---------------------------------------------------------------- depthwise separable


@pytest.mark.parametrize("stride", [1, 2])
def test_depthwise_separable_shape(stride: int) -> None:
    """Forma de salida con stride 1 y 2."""

    block = m.DepthwiseSeparableConv(IN_CHANNELS, OUT_CHANNELS, 3, stride)
    out = block(torch.randn(BATCH, IN_CHANNELS, 8, 8))
    assert out.shape == (BATCH, OUT_CHANNELS, 8 // stride, 8 // stride), "Forma incorrecta."


def test_depthwise_separable_params() -> None:
    """Parámetros = C·k² + C·C' (sin bias) y muchos menos que una conv normal."""

    cin, cout, k = 32, 64, 3
    block = m.DepthwiseSeparableConv(cin, cout, k, bias=False)
    n_sep = sum(p.numel() for p in block.parameters())
    n_std = sum(p.numel() for p in nn.Conv2d(cin, cout, k, bias=False).parameters())
    assert n_sep == cin * k * k + cin * cout, f"Parámetros incorrectos: {n_sep}."
    assert n_sep < n_std / 7, "La separable debería ahorrar ~1/C' + 1/k² del coste."
    assert block.depthwise.groups == cin, "La depthwise debe usar groups=in_channels."
    assert block.pointwise.kernel_size == (1, 1), "La pointwise es una 1×1."


# ---------------------------------------------------------------- transpuesta


def test_transposed_conv2d_hand_value() -> None:
    """Entrada 2×2, kernel de unos 2×2, stride 1 → solapes visibles."""

    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    w = torch.ones(2, 2)
    out = m.transposed_conv2d(x, w)
    expected = torch.tensor([[1.0, 3.0, 2.0], [4.0, 10.0, 6.0], [3.0, 7.0, 4.0]])
    assert out.shape == (3, 3), "Forma incorrecta."
    assert torch.isclose(out, expected).all(), "Valores incorrectos."


@pytest.mark.parametrize("k", [2, 3, 4])
@pytest.mark.parametrize("s", [1, 2, 3])
@pytest.mark.parametrize("p", [0, 1])
@pytest.mark.parametrize("op", [0, 1])
def test_transposed_conv2d_vs_pytorch(k: int, s: int, p: int, op: int) -> None:
    """Coincide con ``F.conv_transpose2d`` en un canal."""

    if op >= s:
        pytest.skip("output_padding debe ser menor que stride.")
    x, w = torch.randn(4, 3), torch.randn(k, k)
    out = m.transposed_conv2d(x, w, s, p, op)
    ref = F.conv_transpose2d(x[None, None], w[None, None], stride=s, padding=p, output_padding=op)[0, 0]
    assert out.shape == ref.shape, f"Forma {tuple(out.shape)} != {tuple(ref.shape)}."
    assert_close(out, ref, atol=1e-4)


def test_transposed_conv_checkerboard() -> None:
    """Con k=3, s=2 y entrada/kernel de unos aparece el patrón de tablero."""

    out = m.transposed_conv2d(torch.ones(4, 4), torch.ones(3, 3), stride=2)
    assert out[2, 2] > out[1, 1], "Las zonas de solape reciben más contribuciones."
    assert len(torch.unique(out[1:-1, 1:-1])) > 1, "Debería haber valores desiguales."


# ---------------------------------------------------------------- upsampling


@pytest.mark.parametrize("r", [2, 3])
def test_pixel_shuffle(r: int) -> None:
    """Coincide con ``F.pixel_shuffle``."""

    x = torch.randn(BATCH, 2 * r * r, 4, 5)
    out = m.pixel_shuffle(x, r)
    assert out.shape == (BATCH, 2, 4 * r, 5 * r), "Forma incorrecta."
    assert_close(out, F.pixel_shuffle(x, r))


@pytest.mark.parametrize("mode", ["nearest", "bilinear"])
@pytest.mark.parametrize("scale", [2, 4])
def test_upsample_conv(mode: str, scale: int) -> None:
    """Forma correcta y equivalencia con interpolate + conv."""

    block = m.UpsampleConv(IN_CHANNELS, OUT_CHANNELS, scale, mode)
    x = torch.randn(BATCH, IN_CHANNELS, 5, 6)
    out = block(x)
    assert out.shape == (BATCH, OUT_CHANNELS, 5 * scale, 6 * scale), "Forma incorrecta."
    kwargs = {"align_corners": False} if mode == "bilinear" else {}
    ref = block.conv(F.interpolate(x, scale_factor=scale, mode=mode, **kwargs))
    assert_close(out, ref)
