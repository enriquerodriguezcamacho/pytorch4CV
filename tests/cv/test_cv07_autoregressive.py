"""
Tests de CV07 · Modelos autorregresivos (MaskedConv2d y PixelCNN).
"""

import math

import pytest
import torch
from helpers import load

m = load("cv/07_generative_models/autoregressive.py")

HEIGHT = WIDTH = 6
N_LEVELS = 4


def test_build_causal_mask_hand_values() -> None:
    """Máscaras 3x3 de tipo A y B escritas a mano."""

    mask_a = torch.tensor([[1.0, 1, 1], [1, 0, 0], [0, 0, 0]])
    mask_b = torch.tensor([[1.0, 1, 1], [1, 1, 0], [0, 0, 0]])
    assert torch.equal(m.build_causal_mask(3, "A"), mask_a), "Máscara A incorrecta."
    assert torch.equal(m.build_causal_mask(3, "B"), mask_b), "Máscara B incorrecta."
    mask5 = m.build_causal_mask(5, "A")
    assert mask5.sum() == 12, "En 5x5 tipo A hay 2 filas completas + 2 píxeles = 12."


def test_build_causal_mask_errors() -> None:
    """Tipos inválidos o kernels pares lanzan ValueError."""

    with pytest.raises(ValueError):
        m.build_causal_mask(3, "C")
    with pytest.raises(ValueError):
        m.build_causal_mask(4, "A")


def test_masked_conv_is_buffer() -> None:
    """La máscara es un buffer (no se entrena) y la salida conserva el tamaño."""

    conv = m.MaskedConv2d(2, 3, 5, "B")
    assert "mask" in dict(conv.named_buffers()), "Registra la máscara con register_buffer."
    assert "mask" not in dict(conv.named_parameters()), "La máscara no es un parámetro."
    out = conv(torch.randn(1, 2, HEIGHT, WIDTH))
    assert out.shape == (1, 3, HEIGHT, WIDTH), f"Forma {out.shape}"


def _raster_index(i: int, j: int) -> int:
    return i * WIDTH + j


@pytest.mark.parametrize("mask_type", ["A", "B"])
@pytest.mark.parametrize(("pi", "pj"), [(2, 3), (0, 0), (5, 5)])
def test_masked_conv_causality(mask_type: str, pi: int, pj: int) -> None:
    """Perturbar el píxel (pi, pj) no cambia salidas anteriores en orden raster;
    con máscara A tampoco cambia la salida en el propio píxel, con B sí."""

    conv = m.MaskedConv2d(1, 4, 3, mask_type)
    x = torch.randn(1, 1, HEIGHT, WIDTH)
    x2 = x.clone()
    x2[0, 0, pi, pj] += 5.0
    diff = (conv(x) - conv(x2)).abs().sum(dim=1)[0]
    for i in range(HEIGHT):
        for j in range(WIDTH):
            if _raster_index(i, j) < _raster_index(pi, pj):
                assert diff[i, j] < 1e-6, f"La salida en {(i, j)} ve el futuro {(pi, pj)}."
    if mask_type == "A":
        assert diff[pi, pj] < 1e-6, "Con máscara A la salida no ve su propio píxel."
    else:
        assert diff[pi, pj] > 1e-4, "Con máscara B la salida sí ve el píxel actual."


def test_pixelcnn_is_causal() -> None:
    """La red completa (A + varias B) es causal estricta: la salida en (i, j) no
    depende de x[i, j] ni de píxeles posteriores."""

    model = m.PixelCNN(hidden=8, n_layers=2, n_levels=N_LEVELS)
    x = torch.rand(1, 1, HEIGHT, WIDTH, requires_grad=True)
    logits = model(x)
    assert logits.shape == (1, N_LEVELS, HEIGHT, WIDTH), f"Forma {logits.shape}"
    i, j = 3, 2
    logits[0, :, i, j].sum().backward()
    grad = x.grad[0, 0]
    flat = grad.flatten()
    assert torch.all(flat[_raster_index(i, j) :] == 0), "Depende de píxeles no causales."
    assert flat[: _raster_index(i, j)].abs().sum() > 0, "Debe depender del pasado."


def test_pixelcnn_nll_and_bits() -> None:
    """Logits uniformes → NLL = ln(n_levels) y log2(n_levels) bits por dimensión."""

    logits = torch.zeros(2, 256, 3, 3)
    x_int = torch.randint(0, 256, (2, 1, 3, 3))
    nll = m.pixelcnn_nll(logits, x_int)
    assert nll.dim() == 0, "Escalar."
    assert torch.isclose(nll, torch.tensor(math.log(256.0))), f"NLL={nll.item()}"
    assert torch.isclose(m.bits_per_dim(nll), torch.tensor(8.0)), "8 bits por píxel."


def test_pixelcnn_trains() -> None:
    """La NLL baja al entrenar unos pasos sobre una imagen fija."""

    model = m.PixelCNN(hidden=8, n_layers=1, n_levels=N_LEVELS)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    x_int = torch.randint(0, N_LEVELS, (2, 1, HEIGHT, WIDTH))
    x = x_int.float() / (N_LEVELS - 1)
    losses = []
    for _ in range(30):
        loss = m.pixelcnn_nll(model(x), x_int)
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0], "La NLL debe disminuir."


def test_sample() -> None:
    """La generación secuencial devuelve enteros válidos y pone el modelo en eval."""

    model = m.PixelCNN(hidden=4, n_layers=1, n_levels=N_LEVELS)
    out = m.sample(model, 4, 4, n=2)
    assert out.shape == (2, 1, 4, 4), f"Forma {out.shape}"
    assert out.dtype == torch.long, "Devuelve niveles enteros (int64)."
    assert out.min() >= 0 and out.max() < N_LEVELS, "Niveles fuera de rango."
    assert not model.training, "El muestreo se hace en modo eval."
