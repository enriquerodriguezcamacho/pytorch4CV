"""
Tests de CV07 · Autoencoders.
"""

import pytest
import torch
from helpers import load

m = load("cv/07_generative_models/autoencoders.py")

BATCH_SIZE = 4
HIDDEN = 8
LATENT = 4


@pytest.mark.parametrize("in_channels", [1, 3])
def test_conv_autoencoder_shapes(in_channels: int) -> None:
    """Encoder a 7x7, decoder de vuelta a 28x28 y salida en [0, 1]."""

    model = m.ConvAutoencoder(in_channels, HIDDEN, LATENT)
    x = torch.rand(BATCH_SIZE, in_channels, 28, 28)
    z = model.encode(x)
    assert z.shape == (BATCH_SIZE, LATENT, 7, 7), f"Código latente incorrecto: {z.shape}"
    out = model.decode(z)
    assert out.shape == x.shape, f"Reconstrucción con forma {out.shape}"
    assert torch.all(out >= 0) and torch.all(out <= 1), "La salida debe ir en [0, 1]."
    assert torch.allclose(model(x), out), "forward debe ser decode(encode(x))."


def test_add_gaussian_noise() -> None:
    """El ruido mantiene la forma, el rango [0, 1] y no modifica la entrada."""

    x = torch.rand(BATCH_SIZE, 1, 28, 28)
    x_copy = x.clone()
    noisy = m.add_gaussian_noise(x, 0.5)
    assert noisy.shape == x.shape, "Forma incorrecta."
    assert torch.equal(x, x_copy), "No modifiques la entrada in-place."
    assert noisy.min() >= 0 and noisy.max() <= 1, "Haz clamp a [0, 1]."
    assert not torch.allclose(noisy, x), "Debe añadirse ruido."
    assert torch.allclose(m.add_gaussian_noise(x, 0.0), x), "std=0 no cambia nada."


@pytest.mark.parametrize("noise_std", [0.0, 0.3])
def test_ae_train_step(noise_std: float) -> None:
    """La pérdida es un float, los parámetros cambian y baja tras varios pasos."""

    model = m.ConvAutoencoder(1, HIDDEN, LATENT)
    model.eval()
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    x = torch.rand(BATCH_SIZE, 1, 28, 28)
    before = [p.clone() for p in model.parameters()]
    first = m.ae_train_step(model, x, opt, noise_std)
    assert isinstance(first, float), "Devuelve loss.item()."
    assert model.training, "ae_train_step debe poner el modelo en modo train."
    changed = any(torch.norm(p - q) > 0 for p, q in zip(model.parameters(), before))
    assert changed, "Los parámetros deben cambiar tras un paso."
    for _ in range(15):
        last = m.ae_train_step(model, x, opt, noise_std)
    assert last < first, "La pérdida debería bajar en un batch fijo."


def test_denoising_target_is_clean() -> None:
    """Con ruido, el objetivo de la MSE es la imagen limpia (no la ruidosa)."""

    class Identity(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = torch.nn.Parameter(torch.ones(()))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x * self.w

    model = Identity()
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    x = torch.full((2, 1, 4, 4), 0.5)
    assert m.ae_train_step(model, x, opt, 0.0) == pytest.approx(0.0), (
        "Sin ruido la identidad reconstruye perfectamente."
    )
    assert m.ae_train_step(model, x, opt, 0.3) > 0, (
        "Con ruido, la identidad ya no coincide con el objetivo limpio."
    )
