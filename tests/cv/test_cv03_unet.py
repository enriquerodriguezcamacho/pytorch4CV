"""
Tests de CV03 · Segmentación: convolución transpuesta, U-Net y cabeza FCN.
"""

import pytest
import torch
import torch.nn.functional as F
from helpers import assert_close, load
from torch import nn

m = load("cv/03_segmentation_style_transfer/segmentation/unet.py")

BATCH = 2
HEIGHT = 24
WIDTH = 16

X_SMALL = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
K_SMALL = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
EXPECTED_TCONV = torch.tensor([[0.0, 0.0, 1.0], [0.0, 4.0, 6.0], [4.0, 12.0, 9.0]])


@pytest.mark.parametrize(
    ("size", "kernel", "stride", "padding", "output_padding", "dilation"),
    [(2, 2, 1, 0, 0, 1), (4, 2, 2, 0, 0, 1), (5, 3, 2, 1, 1, 1), (7, 4, 2, 1, 0, 2)],
)
def test_transposed_conv_output_size(
    size: int, kernel: int, stride: int, padding: int, output_padding: int, dilation: int
) -> None:
    """
    El tamaño coincide con el de ``nn.ConvTranspose2d``.
    """

    layer = nn.ConvTranspose2d(
        1, 1, kernel, stride, padding, output_padding=output_padding, dilation=dilation
    )
    expected = layer(torch.zeros(1, 1, size, size)).shape[-1]
    got = m.transposed_conv_output_size(size, kernel, stride, padding, output_padding, dilation)
    assert got == expected, f"Tamaño incorrecto: {got} != {expected}."


def test_conv_transpose2d_manual_book_example() -> None:
    """
    Ejemplo del libro: [[0,1],[2,3]] * [[0,1],[2,3]] y su versión con padding 1.
    """

    out = m.conv_transpose2d_manual(X_SMALL, K_SMALL)
    assert out.shape == (3, 3), "Forma incorrecta (stride 1, padding 0)."
    assert torch.allclose(out, EXPECTED_TCONV), "Valores incorrectos."
    out_p = m.conv_transpose2d_manual(X_SMALL, K_SMALL, padding=1)
    assert torch.allclose(out_p, torch.tensor([[4.0]])), "El padding recorta la salida."


@pytest.mark.parametrize("stride", [1, 2, 3])
@pytest.mark.parametrize("padding", [0, 1])
def test_conv_transpose2d_manual_matches_torch(stride: int, padding: int) -> None:
    """
    Coincide con ``F.conv_transpose2d`` para strides y paddings generales.
    """

    x, k = torch.randn(4, 5), torch.randn(3, 3)
    out = m.conv_transpose2d_manual(x, k, stride=stride, padding=padding)
    expected = F.conv_transpose2d(x[None, None], k[None, None], stride=stride, padding=padding)
    assert_close(out, expected[0, 0])


def test_classic_upsampling_kernels() -> None:
    """
    Kernels [[1,1],[1,1]] y [[1,0],[0,0]] con stride 2 = nearest neighbor y bed of nails.
    """

    x = torch.arange(1.0, 7.0).reshape(2, 3)
    nn_out = m.conv_transpose2d_manual(x, torch.ones(2, 2), stride=2)
    expected_nn = F.interpolate(x[None, None], scale_factor=2, mode="nearest")[0, 0]
    assert torch.allclose(nn_out, expected_nn), "No equivale a nearest neighbor."

    bed = m.conv_transpose2d_manual(x, torch.tensor([[1.0, 0.0], [0.0, 0.0]]), stride=2)
    expected_bed = torch.zeros(4, 6)
    expected_bed[::2, ::2] = x
    assert torch.allclose(bed, expected_bed), "No equivale a bed of nails."


def test_center_crop() -> None:
    """
    Recorta el centro con las dimensiones pedidas.
    """

    x = torch.arange(36.0).reshape(1, 1, 6, 6)
    out = m.center_crop(x, (2, 4))
    assert out.shape == (1, 1, 2, 4), "Forma incorrecta."
    assert torch.equal(out[0, 0], x[0, 0, 2:4, 1:5]), "No es el centro."


@pytest.mark.parametrize("in_channels", [1, 8])
@pytest.mark.parametrize("out_channels", [4, 16])
@pytest.mark.parametrize("use_bn", [True, False])
def test_conv_block(in_channels: int, out_channels: int, use_bn: bool) -> None:
    """
    Mantiene el tamaño, cambia los canales, aplica ReLU y usa BN solo si se pide.
    """

    block = m.ConvBlock(in_channels, out_channels, use_bn=use_bn)
    out = block(torch.randn(BATCH, in_channels, HEIGHT, WIDTH))
    assert out.shape == (BATCH, out_channels, HEIGHT, WIDTH), "Forma incorrecta."
    assert torch.all(out >= 0), "La salida debe pasar por una ReLU."
    n_convs = sum(isinstance(x, nn.Conv2d) for x in block.modules())
    n_bn = sum(isinstance(x, nn.BatchNorm2d) for x in block.modules())
    assert n_convs == 2, "Debe haber exactamente dos Conv2d."
    assert n_bn == (2 if use_bn else 0), "Número de BatchNorm2d incorrecto."


@pytest.mark.parametrize("in_channels", [3, 16])
@pytest.mark.parametrize("out_channels", [8, 32])
def test_encoder_block(in_channels: int, out_channels: int) -> None:
    """
    Devuelve (skip, pooled) con las resoluciones correctas.
    """

    enc = m.EncoderBlock(in_channels, out_channels)
    skip, pooled = enc(torch.randn(BATCH, in_channels, HEIGHT, WIDTH))
    assert skip.shape == (BATCH, out_channels, HEIGHT, WIDTH), "Skip incorrecta."
    assert pooled.shape == (BATCH, out_channels, HEIGHT // 2, WIDTH // 2), (
        "Salida tras el pooling incorrecta."
    )
    assert torch.allclose(pooled, F.max_pool2d(skip, 2)), "Usa MaxPool2d(2) sobre la skip."


@pytest.mark.parametrize(("in_c", "skip_c", "out_c"), [(32, 16, 16), (16, 4, 8)])
def test_decoder_block(in_c: int, skip_c: int, out_c: int) -> None:
    """
    Sube x2 con ConvTranspose2d(k=2, s=2), concatena y aplica un ConvBlock.
    """

    dec = m.DecoderBlock(in_c, skip_c, out_c)
    tconvs = [x for x in dec.modules() if isinstance(x, nn.ConvTranspose2d)]
    assert len(tconvs) == 1, "Debe haber una ConvTranspose2d."
    assert tconvs[0].kernel_size == (2, 2) and tconvs[0].stride == (2, 2), (
        "La convolución transpuesta debe tener kernel 2 y stride 2."
    )
    x = torch.randn(BATCH, in_c, HEIGHT // 2, WIDTH // 2)
    skip = torch.randn(BATCH, skip_c, HEIGHT, WIDTH)
    out = dec(x, skip)
    assert out.shape == (BATCH, out_c, HEIGHT, WIDTH), "Forma incorrecta."


def test_decoder_block_odd_size_padding() -> None:
    """
    Si la skip tiene tamaño impar, el tensor subido se rellena hasta igualarlo.
    """

    dec = m.DecoderBlock(8, 4, 4)
    out = dec(torch.randn(1, 8, 3, 5), torch.randn(1, 4, 7, 11))
    assert out.shape == (1, 4, 7, 11), "Hay que igualar tamaños con F.pad."


@pytest.mark.parametrize("channels", [[8, 16], [8, 16, 32], [4, 8, 16, 32]])
@pytest.mark.parametrize("n_classes", [1, 5])
def test_unet_shapes(channels: list[int], n_classes: int) -> None:
    """
    La U-Net devuelve un logit por clase y píxel, con el número de bloques correcto.
    """

    net = m.UNet(in_channels=3, n_classes=n_classes, channels=channels)
    assert len(net.encoders) == len(channels) - 1, "Número de encoders incorrecto."
    assert len(net.decoders) == len(channels) - 1, "Número de decoders incorrecto."
    out = net(torch.randn(BATCH, 3, HEIGHT, WIDTH))
    assert out.shape == (BATCH, n_classes, HEIGHT, WIDTH), "Forma de salida incorrecta."
    assert (out < 0).any(), "La U-Net debe devolver logits (sin ReLU/sigmoide final)."


@pytest.mark.parametrize(("height", "width"), [(30, 30), (21, 17)])
def test_unet_non_divisible_sizes(height: int, width: int) -> None:
    """
    Funciona con tamaños no divisibles por 2^profundidad.
    """

    net = m.UNet(in_channels=1, n_classes=3, channels=[4, 8, 16, 32])
    out = net(torch.randn(1, 1, height, width))
    assert out.shape == (1, 3, height, width), "Forma de salida incorrecta."


def test_unet_skip_concat_channels() -> None:
    """
    El ConvBlock del decoder recibe out + skip canales (concatenación, no suma).
    """

    net = m.UNet(in_channels=3, n_classes=2, channels=[16, 32, 64])
    first_conv = next(x for x in net.decoders[0].conv.modules() if isinstance(x, nn.Conv2d))
    assert first_conv.in_channels == 64, "Decoder 64->32 con skip de 32: entran 64 canales."


def test_unet_train_step_and_predict() -> None:
    """
    Un paso de entrenamiento cambia todos los parámetros; ``predict`` da una máscara.
    """

    net = m.UNet(in_channels=3, n_classes=3, channels=[4, 8, 16])
    x = torch.randn(2, 3, 16, 16)
    y = torch.randint(0, 3, (2, 16, 16))
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    old = [p.detach().clone() for p in net.parameters()]
    net.train()
    loss = F.cross_entropy(net(x), y)
    opt.zero_grad()
    loss.backward()
    opt.step()
    for i, (o, p) in enumerate(zip(old, net.parameters())):
        assert torch.norm(p.detach() - o) > 0, f"El parámetro {i} no se ha actualizado."
    mask = net.predict(x)
    assert mask.shape == (2, 16, 16) and mask.dtype == torch.int64, "Máscara incorrecta."
    assert not net.training, "predict debe poner el modelo en eval."


@pytest.mark.parametrize("output_size", [(32, 32), (50, 30)])
def test_fcn_head(output_size: tuple[int, int]) -> None:
    """
    Conv 1x1 + interpolación bilineal hasta el tamaño de la imagen.
    """

    head = m.FCNHead(in_channels=16, n_classes=4)
    feats = torch.randn(BATCH, 16, 4, 4)
    out = head(feats, output_size)
    assert out.shape == (BATCH, 4, *output_size), "Forma incorrecta."
    conv = next(x for x in head.modules() if isinstance(x, nn.Conv2d))
    assert conv.kernel_size == (1, 1), "La cabeza usa una convolución 1x1."
    expected = F.interpolate(conv(feats), size=output_size, mode="bilinear", align_corners=False)
    assert_close(out, expected)
